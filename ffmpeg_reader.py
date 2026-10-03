"""Sequential RGB decoding through an ffmpeg pipe with explicit colour handling.

Why this exists: OpenCV's bundled FFmpeg converts YUV to RGB without telling
swscale which colour matrix the stream is tagged with. On some builds that
means BT.601 coefficients for BT.709 material. This reader reads the stream
tags itself (matrix, range) and passes them to ffmpeg explicitly, together
with exact rounding, so the result is identical on every ffmpeg build.

No ComfyUI, Torch or OpenCV dependency; comfy interruption is used if present.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import threading

import numpy as np

LOGGER = logging.getLogger("FlashVSR-SAnodes")

try:  # inside ComfyUI
    import comfy.model_management as _mm
except Exception:  # standalone tests
    _mm = None

_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "smpte170m", "bt470bg": "bt470bg", "bt601": "smpte170m",
    "bt2020nc": "bt2020", "bt2020c": "bt2020", "bt2020": "bt2020",
    "smpte240m": "smpte240m", "fcc": "fcc",
}


def find_ffmpeg(preferred=None):
    """Prefer VideoHelperSuite's ffmpeg, then PATH, then imageio-ffmpeg."""
    candidates = [preferred]
    try:
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
        if cls is not None:
            import importlib
            module = importlib.import_module(cls.__module__)
            candidates.append(getattr(module, "ffmpeg_path", None))
            utils = sys.modules.get(cls.__module__.rsplit(".", 1)[0] + ".utils")
            if utils is not None:
                candidates.append(getattr(utils, "ffmpeg_path", None))
    except Exception:
        pass
    candidates.append(shutil.which("ffmpeg"))
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("ffmpeg not found. Install VideoHelperSuite's ffmpeg or put ffmpeg on PATH.")


def _no_window():
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def probe_video(ffmpeg, video):
    """Return width, height, fps, nb_frames, pix_fmt, matrix, range from the
    stream tags. ffprobe (if it sits next to ffmpeg) gives exact numbers;
    otherwise ffmpeg's own stream dump is parsed."""
    info = {"width": 0, "height": 0, "fps": 0.0, "nb_frames": 0,
            "pix_fmt": "?", "matrix": "unknown", "range": "unknown", "source": ""}
    ffprobe = os.path.join(os.path.dirname(ffmpeg) or ".",
                           "ffprobe.exe" if os.name == "nt" else "ffprobe")
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt,"
               "color_space,color_range", "-of", "default=noprint_wrappers=1", video]
        proc = subprocess.run(cmd, capture_output=True, text=True, **_no_window())
        if proc.returncode == 0:
            kv = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
            info["width"] = int(kv.get("width", 0) or 0)
            info["height"] = int(kv.get("height", 0) or 0)
            rate = kv.get("avg_frame_rate") or kv.get("r_frame_rate") or "0/1"
            if rate in ("0/0", "N/A", ""):
                rate = kv.get("r_frame_rate") or "0/1"
            num, _, den = rate.partition("/")
            info["fps"] = float(num) / float(den or 1) if float(den or 1) else 0.0
            frames = kv.get("nb_frames", "")
            info["nb_frames"] = int(frames) if frames.isdigit() else 0
            info["pix_fmt"] = kv.get("pix_fmt", "?")
            info["matrix"] = kv.get("color_space", "unknown")
            info["range"] = kv.get("color_range", "unknown")
            info["source"] = "ffprobe"
            return info
    # Fallback: parse "Stream #0:0 ... Video: hevc (...), yuv444p10le(tv, bt709), 1280x720 ..., 24 fps"
    proc = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", video],
                          capture_output=True, text=True, **_no_window())
    text = proc.stderr
    match = re.search(r"Video: .*?, ([a-z0-9]+)(?:\(([^)]*)\))?, (\d+)x(\d+)", text)
    if not match:
        raise RuntimeError(f"Could not read the video stream of {video}:\n{text[-600:]}")
    info["pix_fmt"] = match.group(1)
    info["width"], info["height"] = int(match.group(3)), int(match.group(4))
    tags = [t.strip() for t in (match.group(2) or "").split(",")]
    for tag in tags:
        if tag in ("tv", "pc"):
            info["range"] = tag
        elif "/" in tag:
            info["matrix"] = tag.split("/")[-1]  # primaries/transfer/matrix
        elif tag in _MATRIX:
            info["matrix"] = tag
    fps = re.search(r"([\d.]+) fps", text)
    info["fps"] = float(fps.group(1)) if fps else 0.0
    info["source"] = "ffmpeg -i"
    return info


def choose_matrix(info):
    """Explicit swscale matrix name, with the HD/SD convention for untagged files."""
    tagged = _MATRIX.get(info["matrix"])
    if tagged:
        return tagged, "tagged"
    return ("bt709" if info["height"] >= 720 else "smpte170m"), "untagged, by resolution"


class FFmpegRGBReader:
    """Decode frames in order into uint8 RGB with the stream's own colour tags."""

    def __init__(self, video, ffmpeg=None, log=True):
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.video = video
        self.info = probe_video(self.ffmpeg, video)
        if self.info["width"] <= 0 or self.info["height"] <= 0:
            raise RuntimeError(f"Could not determine the frame size of {video}.")
        self.fps = float(self.info["fps"])
        if not self.fps or self.fps < 1:
            raise RuntimeError("Invalid video frame rate. Convert the source to CFR first.")
        self.total_frames = int(self.info["nb_frames"])
        self.width, self.height = self.info["width"], self.info["height"]
        self.dimensions = (self.height, self.width, 3)
        self.frame_bytes = self.width * self.height * 3
        self.matrix, why = choose_matrix(self.info)
        self.range = "pc" if self.info["range"] == "pc" else "tv"
        vf = (f"scale=in_color_matrix={self.matrix}:in_range={self.range}:out_range=pc:"
              f"flags=accurate_rnd+full_chroma_int+bitexact,format=rgb24")
        cmd = [self.ffmpeg, "-v", "error", "-nostdin", "-i", video, "-map", "0:v:0",
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._stderr = []
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=self.frame_bytes * 4,
                                     **_no_window())
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self.closed = False
        if log:
            LOGGER.info("[SAnodes] ffmpeg decode: %s | %dx%d | %.6f fps | %s frames | "
                        "pix_fmt=%s matrix=%s (%s) range=%s | tags via %s",
                        os.path.basename(video), self.width, self.height, self.fps,
                        self.total_frames or "unknown", self.info["pix_fmt"], self.matrix, why,
                        self.range, self.info["source"])

    def _drain_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr.append(line.decode("utf-8", "replace"))
        except Exception:
            pass

    def read_frame(self):
        if _mm is not None:
            _mm.throw_exception_if_processing_interrupted()
        if self.closed:
            return None
        chunks, remaining = [], self.frame_bytes
        while remaining:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == self.frame_bytes:  # clean EOF
            code = self.proc.wait()
            if code not in (0, None):
                raise RuntimeError(f"ffmpeg ended with code {code}:\n{''.join(self._stderr)[-800:]}")
            return None
        if remaining:
            raise RuntimeError(f"ffmpeg delivered a truncated frame ({self.frame_bytes - remaining} "
                               f"of {self.frame_bytes} bytes):\n{''.join(self._stderr)[-800:]}")
        data = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.dimensions)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except Exception:
                pass
