"""Welche Farbmatrix wendet OpenCV beim Dekodieren eines Masters an?

Vergleicht EINEN Frame (Standard: Frame 3000, ein gesättigter aus der Mitte)
per cv2 gegen zwei ffmpeg-Referenzen: BT.709 und BT.601, beide mit exakter
Rundung. Zeigt ausserdem die Stream-Tags des Masters.

    python_embeded\\python.exe check_master_decode_v2.py "Master.mp4" "ffmpeg.exe" [frame]
"""

import re
import subprocess
import sys

import cv2
import numpy as np


def reference(ffmpeg, video, frame, matrix, h, w):
    vf = (f"select=eq(n\\,{frame}),scale=in_color_matrix={matrix}:in_range=tv:out_range=pc:"
          "flags=accurate_rnd+full_chroma_int,format=rgb24")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", video, "-vf", vf, "-vsync", "0",
           "-frames:v", "1", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) != h * w * 3:
        raise RuntimeError("ffmpeg-Referenz fehlgeschlagen: " + proc.stderr.decode(errors="replace")[:400])
    return np.frombuffer(proc.stdout, np.uint8).reshape(h, w, 3).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    video, ffmpeg = sys.argv[1], sys.argv[2]
    frame = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    dump = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", video],
                          capture_output=True, text=True).stderr
    line = re.search(r"Stream #.*Video:.*", dump)
    print("Stream:", line.group(0).strip() if line else "(nicht gefunden)")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("OpenCV kann die Datei nicht öffnen.")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = max(0, min(frame, max(total - 1, 0)))
    bgr = None
    for _ in range(frame + 1):
        ok, bgr = cap.read()
        if not ok:
            print("OpenCV liefert Frame", frame, "nicht.")
            return 1
    cap.release()
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = a.shape[:2]
    print(f"OpenCV {cv2.__version__}: Frame {frame} von {total}, {w}x{h}, "
          f"mittlere Sättigung {np.abs(a - a.mean(axis=2, keepdims=True)).mean():.1f}")

    results = {}
    for name in ("bt709", "bt601"):
        b = reference(ffmpeg, video, frame, name, h, w)
        d = a - b
        results[name] = np.abs(d).mean()
        print(f"cv2 − ffmpeg({name}): |abs| {results[name]:.3f}  max {np.abs(d).max():.0f}  "
              f"Bias R {d[..., 0].mean():+.2f} G {d[..., 1].mean():+.2f} B {d[..., 2].mean():+.2f}")

    if results["bt709"] <= 0.15:
        print("→ OpenCV rechnet mit BT.709 (exakt). Loader muss nicht umgebaut werden.")
    elif results["bt601"] <= 0.15:
        print("→ OpenCV rechnet mit BT.601 — FALSCH für dieses Master. ffmpeg-Decoder nutzen.")
    elif results["bt709"] < results["bt601"]:
        print("→ OpenCV liegt bei BT.709, aber mit Rundungs-/Dither-Unterschieden. "
              "ffmpeg-Decoder ist exakter, der Unterschied ist klein.")
    else:
        print("→ OpenCV liegt näher an BT.601 als an BT.709. ffmpeg-Decoder nutzen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
