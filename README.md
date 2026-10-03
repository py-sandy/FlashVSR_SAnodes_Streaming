# ComfyUI SAnodes — FlashVSR streaming for large video files

Upscale videos of **any length** with **FlashVSR** in ComfyUI — without ever loading the whole video into memory.

Two small custom nodes turn the existing FlashVSR workflow into a **bounded streaming pipeline**: the source is decoded in overlapping frame windows, each window is upscaled, the overlap is blended frame-for-frame, and the output is written continuously. Click **Queue** once; the rest is automatic.

> Built on top of **[ComfyUI-FlashVSR_Stable](https://github.com/naxci1/ComfyUI-FlashVSR_Stable)** and **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)**. Both stay installed and untouched — SAnodes only adds the streaming layer around them.  
> License: **GPLv3**

---

## 🎬 Why this exists

FlashVSR is fast and beautiful, but the stock video loader reads the **entire** file into RAM as float32 frames before the first tile is upscaled. A four-minute 720p clip (5 664 frames) is already ~60 GB of float32 tensors; a long film ends in an out-of-memory exception long before VRAM is even the question.

SAnodes keeps a **fixed, small number of frames alive at any moment** — the decoder holds one window, the blender holds one overlap, the encoder writes what is final. Memory use is flat, no matter how long the video is.

---

## ✨ What you get

- **Bounded memory** — at most `frames_per_batch + overlap_frames + 1` decoded frames and `overlap_frames` upscaled frames in RAM, ever.
- **Same-frame overlap blending** — the last `overlap_frames` of a window are upscaled *again* at the start of the next window and cross-faded with a raised-cosine curve. No duplicated frames, no time shift, no visible seams.
- **Exact frame accounting** — the pipeline counts every frame in and out and **stops loudly** rather than silently dropping or repeating one. The log ends with `written=N | COMPLETE`.
- **Colour-correct decoding (v1.3)** — frames are decoded through an `ffmpeg` pipe with the colour matrix and range taken from the stream tags (BT.709 / BT.601 / BT.2020, tv / pc) and exact rounding. HEVC 4:4:4 10-bit masters are read faithfully. OpenCV remains available as a fallback decoder.
- **One click** — VideoHelperSuite's Meta Batch Manager re-queues the next window automatically until the file is done.
- **Audio carried over** — the source audio is passed through to Video Combine.
- **Windows HTTP fix** — switches aiohttp to buffered file transfer on Windows, avoiding the `WinError 87` sendfile failure on large previews/downloads.

---

## 🧠 How the overlap works

```
window 1 :  frames   0 … 95   →  write   0 … 63   keep  64 … 95
window 2 :  frames  64 … 159  →  blend  64 … 95 (kept ⟷ new)
                               →  write  64 … 127  keep 128 … 159
window 3 :  frames 128 … 223  →  …
```

Every window is `frames_per_batch + overlap_frames` frames long and advances by `frames_per_batch`. The blend only ever mixes **the same original timestamps** — the end of the previous window with the cold start of the next — so the output has exactly as many frames as the input.

With 5664 frames and `256 + 32`, that is 23 windows and ~6 600 frames through FlashVSR (17 % overhead). With `64 + 32` it would be 89 windows and ~8 500 frames (51 % overhead) — larger batches are faster, as long as they fit.

---

## 🧰 Requirements

- **ComfyUI** (a current build; tested with the Windows portable package on Python 3.13)
- **ComfyUI-FlashVSR_Stable** — the FlashVSR nodes and models
- **ComfyUI-VideoHelperSuite** — Meta Batch Manager and Video Combine
- **ffmpeg** — VideoHelperSuite usually brings one (imageio-ffmpeg); otherwise put `ffmpeg` on your `PATH`
- **Triton** for Windows users running the sparse SageAttention path (see the FlashVSR_Stable docs; with ComfyUI portable you also need the Python `include` and `libs` folders)

No additional `pip install` is normally needed — the nodes use the Python dependencies already present for FlashVSR_Stable and VideoHelperSuite.

---

## 🚀 Installation

**Option A — ComfyUI Manager (recommended)**

Open **Manager → Custom Nodes Manager → Install via Git URL** and paste:

```
https://github.com/py-sandy/FlashVSR_SAnodes_Streaming
```

Restart ComfyUI when the Manager asks for it.

**Option B — git**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/py-sandy/FlashVSR_SAnodes_Streaming.git
```

Restart ComfyUI. The node folder is `ComfyUI/custom_nodes/FlashVSR_SAnodes_Streaming/` with `__init__.py` directly inside it.

**Option C — release zip**

Download the zip from the [Releases](../../releases) page and extract it so that `__init__.py` sits directly in `ComfyUI/custom_nodes/FlashVSR_SAnodes_Streaming/`. Do not create a second nested folder.

**Updating:** use the Manager's update button or `git pull` in the node folder, restart ComfyUI, and re-import the bundled workflow if the node definitions changed (the release notes say so).

---

## 🖥️ Quick start

1. In ComfyUI, load `FlashVSR_SAnodes_Streaming_Overlap.json` (bundled in the node folder).
2. In node **1 · Stream Load + Context**, paste the **full path** to your source video.
3. Optionally set `expected_frames` to the exact frame count of the file (a final sanity check), or leave `0`.
4. Click **Queue** once. Leave *Auto Queue* **off** — VideoHelperSuite requeues the remaining windows itself.
5. Watch the console: each window logs `Input a..b (n frames)` and `Emit n frames | written=N`. The run is finished when you see `COMPLETE` and Video Combine has closed the file.

---

## ⚙️ The two nodes

### FlashVSR Stream Load + Context (SAnodes)

| Input | Meaning |
|---|---|
| `meta_batch` | VideoHelperSuite **Meta Batch Manager**. Its `frames_per_batch` is the number of **new** frames per window. |
| `video` | Full path to the source file. |
| `overlap_frames` | Context frames re-upscaled at the start of each window and blended. |
| `expected_frames` | Expected total frame count; `0` disables the check. |
| `decoder` | `ffmpeg (colour matrix from stream tags, exact rounding)` — default; `opencv` — legacy path. |

Outputs: `frames` (one window), `audio`, `fps` (from the source), `stream_info` (for the blend node).

Rules: `frames_per_batch` ≥ 32 and `overlap_frames` ≥ 8, both multiples of 8, overlap ≤ batch. Settings are fixed for the duration of a run — change them, then queue a fresh run.

### FlashVSR Same-Frame Overlap Blend (SAnodes)

Takes the upscaled window from FlashVSR plus `stream_info` and `meta_batch`, blends the overlap with the frames kept from the previous window, and emits exactly the frames that are final. Connect its output to Video Combine.

---

## 🎚️ Settings that worked

Reference configuration for a 1280 × 720 source → 2560 × 1440 output on an RTX 5090 (32 GB VRAM) with 128 GB system RAM:

| Parameter | Value |
|---|---|
| Meta Batch Manager `frames_per_batch` | 256 (64 is the conservative default) |
| Stream Load `overlap_frames` | 32 |
| Frames per FlashVSR call | batch + overlap (288 / 96) |
| FlashVSR `frame_chunk_size` | 0 (= process the already bounded input batch) |
| Model / mode | FlashVSR-v1.1 / full |
| Precision / device | bf16 / cuda:0 |
| VAE | Wan2.1 |
| `scale` / `resize_factor` | 2 / 1.0 |
| `tiled_vae` / `tiled_dit` | true / true |
| `tile_size` / `tile_overlap` | 256 / 24 |
| `sparse_ratio` / `kv_ratio` / `local_range` | 2 / 3 / 11 |
| Seed | fixed |
| Output | ProRes (intermediate master) or H.264/H.265 MP4 with a low CRF |
| Frame rate | taken from the source, wired into Video Combine |

On a real **CUDA** out-of-memory error, reduce `frames_per_batch` (256 → 128 → 64 → 32). Note that the FlashVSR pre-flight estimate is computed for the whole image at once; with tiling enabled, much larger batches run fine than the estimate suggests.

---

## 🧪 Diagnostics: is my decoder colour-correct?

`check_master_decode_v2.py` (in the node folder) decodes one frame of your source with OpenCV and compares it against two ffmpeg references (BT.709 and BT.601, exact rounding). Run it with ComfyUI's own Python so the test sees the same OpenCV build the nodes would use:

```bat
python_embeded\python.exe custom_nodes\FlashVSR_SAnodes_Streaming\check_master_decode_v2.py "C:\path\to\master.mp4" "C:\path\to\ffmpeg.exe" 3000
```

If OpenCV lands on BT.601 for a BT.709 master, keep the default `ffmpeg` decoder — it is the reason it exists.

---

## 🧯 Troubleshooting

**The run stops after the first window / nothing is requeued**  
Auto Queue must be **off**; the Meta Batch Manager does the requeueing. Make sure the Stream Load node is connected to the *same* Meta Batch Manager as Video Combine.

**`A video batch was skipped or repeated. Queue a fresh run.`**  
The blend node lost sync with the loader (for example after changing settings mid-run). Queue a fresh run; nothing partial is written silently.

**`Decoded N frames; expected M`**  
`expected_frames` does not match the file. Set it to the real count or `0`.

**`ffmpeg not found`**  
Install ffmpeg or put it on `PATH`; VideoHelperSuite's bundled ffmpeg is picked up automatically when present. `ffprobe` is optional — without it the frame count is unknown until the end, which is harmless.

**Triton compile error on Windows (`Python.h not found`)**  
ComfyUI's embedded Python ships without C headers. Add the `include` and `libs` folders for your Python version to `python_embeded` (see the triton-windows project), then restart.

**CUDA out of memory**  
Lower `frames_per_batch`; keep `tiled_vae` and `tiled_dit` on.

---

## 🧭 Limitations

- No resume: after an abort, queue the run again from the start.
- Constant frame rate is assumed; variable-frame-rate timestamps are not carried over.
- The overlap blend reduces seam risk but is not a continuous KV cache across the whole video — results are not guaranteed identical to a hypothetical single-pass inference of the full length.
- The final MP4 must be completely written before it is used.

---

## 🤝 Contributing

Issues and PRs are welcome. Please keep the frame accounting strict — the nodes should rather stop than guess — and test with a long file, not just a short clip.

---

## 🙏 Acknowledgements

- **[FlashVSR](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)** by Junhao Zhuang et al., and **ComfyUI-FlashVSR_Stable** by naxci1 for the ComfyUI integration.
- **ComfyUI-VideoHelperSuite** by Kosinkadink — the Meta Batch Manager is what makes the one-click streaming possible.
- The ComfyUI community.

---

## 📜 License

GPLv3. FlashVSR, ComfyUI-FlashVSR_Stable and VideoHelperSuite are licensed separately; see their repositories.

Created and maintained with 💙 by **py-sandy** (SAmedia)
