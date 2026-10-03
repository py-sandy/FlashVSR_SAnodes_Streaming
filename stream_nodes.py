"""Two ComfyUI adapters; existing FlashVSR and VHS nodes remain installed."""

import importlib
import logging
import math
import os
from pathlib import Path

import numpy as np

from .ffmpeg_reader import FFmpegRGBReader
from .stream_core import OverlapBlend, WindowStream, close_guard, validate_window

LOGGER = logging.getLogger("FlashVSR-SAnodes")

DECODER_FFMPEG = "ffmpeg (Farbmatrix aus den Stream-Tags, exakte Rundung)"
DECODER_OPENCV = "opencv (wie bis v1.2)"


def vhs_load_module():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if cls is None:
        raise RuntimeError("Install/enable VideoHelperSuite and restart ComfyUI.")
    module = importlib.import_module(cls.__module__)
    if not callable(getattr(module, "lazy_get_audio", None)):
        raise RuntimeError("This VideoHelperSuite version lacks lazy_get_audio.")
    return module


def resolve_video_path(value):
    import folder_paths
    value = os.path.expanduser(os.path.expandvars(value.strip().strip('"')))
    candidates = [Path(value)]
    root = Path(getattr(folder_paths, "base_path", os.getcwd()))
    candidates.append(root / value)
    normalized = value.replace("\\", "/")
    if normalized.startswith("ComfyUI/"):
        candidates.append(root / normalized[len("ComfyUI/"):])
    # Also allow the normal ComfyUI input/output path annotations.
    try:
        candidates.append(Path(folder_paths.get_annotated_filepath(value)))
    except (AttributeError, ValueError, KeyError):
        pass
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Video not found: {value}. Set its full path in the stream loader.")


class RGBVideoReader:
    """OpenCV sequential decoder, using the same RGB conversion as VHS.

    Kept as a fallback. OpenCV does not tell swscale which colour matrix the
    stream carries; on some builds BT.709 material is converted with BT.601
    coefficients. The ffmpeg reader is the default since v1.3.
    """

    def __init__(self, video):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(video)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"OpenCV could not open the video: {video}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps < 1:
            self.close()
            raise RuntimeError("Invalid video frame rate. Convert the source to CFR first.")
        count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(count) if math.isfinite(count) and count > 0 else 0
        self.dimensions = None

    def read_frame(self):
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
        success, frame = self.capture.read()
        if not success:
            return None
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError("The decoder returned an invalid video frame.")
        if self.dimensions is None:
            self.dimensions = frame.shape
        elif self.dimensions != frame.shape:
            raise RuntimeError("Video dimensions changed while decoding.")
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.capture.release()


def open_reader(path, decoder):
    if decoder == DECODER_OPENCV:
        LOGGER.info("[SAnodes] decoder: OpenCV (fallback)")
        return RGBVideoReader(path)
    ffmpeg = None
    try:
        module = vhs_load_module()
        ffmpeg = getattr(module, "ffmpeg_path", None)
    except Exception:
        pass
    return FFmpegRGBReader(path, ffmpeg=ffmpeg)


class SAnodesFlashVSRStreamLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta_batch": ("VHS_BatchManager",),
                "video": ("STRING", {"default": "", "multiline": True}),
                "overlap_frames": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "expected_frames": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "decoder": ([DECODER_FFMPEG, DECODER_OPENCV], {"default": DECODER_FFMPEG}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "FLASHVSR_SANODES_STREAM")
    RETURN_NAMES = ("frames", "audio", "fps", "stream_info")
    FUNCTION = "load_window"
    CATEGORY = "FlashVSR/SAnodes streaming"
    DESCRIPTION = "Load at most frames_per_batch + overlap_frames RGB frames; keep the decoder open."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Each VHS requeue must decode the next window, even for identical input values.
        return float("nan")

    def load_window(self, meta_batch, video, overlap_frames, expected_frames=0,
                    decoder=DECODER_FFMPEG, unique_id=None):
        import torch
        if unique_id is None:
            raise RuntimeError("ComfyUI did not supply the loader's unique_id.")
        if not hasattr(meta_batch, "inputs") or not hasattr(meta_batch, "has_closed_inputs"):
            raise RuntimeError("Connect a VideoHelperSuite Meta Batch Manager.")
        core, overlap = validate_window(meta_batch.frames_per_batch, overlap_frames)
        entry = meta_batch.inputs.get(unique_id)
        if entry is None:
            path = resolve_video_path(video)
            reader = open_reader(path, decoder)
            stream = WindowStream(reader, core, overlap)
            stream.expected_frames = int(expected_frames)
            stream.decoder = decoder
            guard = close_guard(stream.close)
            next(guard)
            try:
                audio = vhs_load_module().lazy_get_audio(path, 0, 0)
            except BaseException:
                guard.close()
                raise
            # VHS closes the last tuple element when resetting/cancelling a batch.
            entry = (stream, audio, guard)
            meta_batch.inputs[unique_id] = entry
            meta_batch.has_closed_inputs = False
            if reader.total_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, reader.total_frames)
            LOGGER.info("[SAnodes] Open: %s | %.6f fps | %s frames | window=%s+%s",
                        path, reader.fps, reader.total_frames or "unknown", core, overlap)
        stream, audio, guard = entry
        if (stream.core_frames, stream.overlap_frames) != (core, overlap):
            raise RuntimeError("Batch/overlap settings changed during a run. Queue a fresh run.")
        if stream.expected_frames != int(expected_frames):
            raise RuntimeError("expected_frames changed during a run. Queue a fresh run.")
        if getattr(stream, "decoder", decoder) != decoder:
            raise RuntimeError("decoder changed during a run. Queue a fresh run.")
        try:
            raw_frames, info = stream.next_window()
            if info["last"] and expected_frames and info["decoded_frames"] != expected_frames:
                raise RuntimeError(
                    f"Decoded {info['decoded_frames']} frames; expected {expected_frames}. "
                    "Check the source path/file or set expected_frames=0 for a different video."
                )
            # One bounded conversion. uint8 source buffers are discarded immediately.
            array = np.stack(raw_frames).astype(np.float32)
            array *= np.float32(1.0 / 255.0)
            del raw_frames
            tensor = torch.from_numpy(array)
        except BaseException:
            guard.close()
            meta_batch.inputs.pop(unique_id, None)
            meta_batch.has_closed_inputs = True
            raise
        if info["last"]:
            guard.close()
            meta_batch.inputs.pop(unique_id, None)
            meta_batch.has_closed_inputs = True
            meta_batch.total_frames = info["decoded_frames"]
        LOGGER.info("[SAnodes] Input %s..%s (%s frames)%s",
                    info["start"], info["start"] + info["length"] - 1,
                    info["length"], " | final" if info["last"] else "")
        return tensor, audio, stream.reader.fps, info


class SAnodesFlashVSROverlapBlend:
    def __init__(self):
        self.blender = OverlapBlend()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "stream_info": ("FLASHVSR_SANODES_STREAM",),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "blend_window"
    CATEGORY = "FlashVSR/SAnodes streaming"
    DESCRIPTION = "Blend equal original timestamps and retain only the output overlap for the next window."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def blend_window(self, images, stream_info, meta_batch, unique_id=None):
        import torch
        if unique_id is None:
            raise RuntimeError("ComfyUI did not supply the blend node's unique_id.")
        key = f"SAnodesOverlapBlend:{unique_id}"
        if stream_info["first"]:
            old = meta_batch.inputs.pop(key, None)
            if old is not None:
                old[-1].close()
            self.blender.reset()
            guard = close_guard(self.blender.reset)
            next(guard)
            meta_batch.inputs[key] = (self.blender, guard)
        try:
            array = images.detach().to(device="cpu", dtype=torch.float32).numpy()
            result = self.blender.consume(array, stream_info)
        except BaseException:
            entry = meta_batch.inputs.pop(key, None)
            if entry is not None:
                entry[-1].close()
            raise
        written = self.blender.written_frames
        LOGGER.info("[SAnodes] Emit %s frames | written=%s%s",
                    result.shape[0], written, " | COMPLETE" if stream_info["last"] else "")
        if stream_info["last"]:
            entry = meta_batch.inputs.pop(key, None)
            if entry is not None:
                entry[-1].close()
        return (torch.from_numpy(result),)


NODE_CLASS_MAPPINGS = {
    "SAnodesFlashVSRStreamLoad": SAnodesFlashVSRStreamLoad,
    "SAnodesFlashVSROverlapBlend": SAnodesFlashVSROverlapBlend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAnodesFlashVSRStreamLoad": "FlashVSR Stream Load + Context (SAnodes)",
    "SAnodesFlashVSROverlapBlend": "FlashVSR Same-Frame Overlap Blend (SAnodes)",
}
