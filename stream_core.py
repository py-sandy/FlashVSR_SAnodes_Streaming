"""Video-length-independent buffering and same-frame overlap blending.

This module deliberately has no ComfyUI, Torch, OpenCV or CUDA dependency.
Frames held by WindowStream are uint8 RGB; only the active inference window
is converted to Float32 by the ComfyUI adapter.
"""

from collections import deque
from itertools import islice
from uuid import uuid4

import numpy as np


def validate_window(core_frames, overlap_frames):
    core_frames, overlap_frames = int(core_frames), int(overlap_frames)
    if core_frames < 32 or core_frames % 8:
        raise ValueError("frames_per_batch must be at least 32 and a multiple of 8.")
    if overlap_frames < 8 or overlap_frames % 8:
        raise ValueError("overlap_frames must be at least 8 and a multiple of 8.")
    if overlap_frames > core_frames:
        raise ValueError("overlap_frames must not exceed frames_per_batch.")
    return core_frames, overlap_frames


class WindowStream:
    """Read a window of B+O frames, then advance B frames without seeking.

    One extra uint8 frame detects EOF exactly, including files whose metadata
    frame count is inaccurate. B is the number of frames normally emitted;
    O frames are processed again by the next inference window. The last
    window emits its complete remaining content, including any overlap.
    """

    def __init__(self, reader, core_frames, overlap_frames):
        self.core_frames, self.overlap_frames = validate_window(
            core_frames, overlap_frames
        )
        self.reader = reader
        self.stream_id = uuid4().hex
        self.start = 0
        self.decoded_frames = 0
        self.max_buffered_frames = 0
        self.buffer = deque()
        self.eof = False
        self.finished = False
        self.closed = False

    def next_window(self):
        if self.finished or self.closed:
            raise RuntimeError("This video stream has finished. Queue a new run.")
        target = self.core_frames + self.overlap_frames
        try:
            while len(self.buffer) < target + 1 and not self.eof:
                frame = self.reader.read_frame()
                if frame is None:
                    self.eof = True
                    break
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError("The decoder must return uint8 RGB frames.")
                self.buffer.append(frame)
                self.decoded_frames += 1
                self.max_buffered_frames = max(self.max_buffered_frames, len(self.buffer))
            if not self.buffer:
                raise RuntimeError("No video frames could be decoded.")
            length = min(target, len(self.buffer))
            frames = list(islice(self.buffer, length))
            last = self.eof and len(self.buffer) <= target
            info = {
                "stream_id": self.stream_id,
                "start": self.start,
                "length": length,
                "core_frames": self.core_frames,
                "overlap_frames": self.overlap_frames,
                "first": self.start == 0,
                "last": last,
                "decoded_frames": self.decoded_frames,
            }
            if last:
                self.finished = True
                self.close()
            else:
                for _ in range(self.core_frames):
                    self.buffer.popleft()
                self.start += self.core_frames
            return frames, info
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.buffer.clear()
            self.reader.close()


class OverlapBlend:
    """Retain only O output frames and combine equal source timestamps.

    A raised-cosine transition suppresses the new window's cold start and the
    previous window's end. It never mixes different timestamps. Operations
    use one frame of scratch memory rather than two full overlap tensors.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stream_id = None
        self.pending = None
        self.next_start = 0
        self.written_frames = 0
        self.finished = False

    def consume(self, frames, info):
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError("Expected an IMAGE batch with shape [N, H, W, 3].")
        if frames.shape[0] != info["length"]:
            raise RuntimeError(
                f"FlashVSR returned {frames.shape[0]} frames for an input of "
                f"{info['length']}. Stop: no frames will be silently removed."
            )
        core, overlap = validate_window(info["core_frames"], info["overlap_frames"])
        if info["first"]:
            self.reset()
            self.stream_id = info["stream_id"]
        if self.stream_id != info["stream_id"] or self.finished:
            raise RuntimeError("The overlap stream is out of sync. Queue a fresh run.")
        if info["start"] != self.next_start:
            raise RuntimeError("A video batch was skipped or repeated. Queue a fresh run.")
        if not info["last"] and frames.shape[0] != core + overlap:
            raise RuntimeError("A non-final inference window is incomplete.")
        count = frames.shape[0] if info["last"] else core
        output = np.empty((count, *frames.shape[1:]), dtype=np.float32)
        if self.pending is None:
            output[:] = frames[:count]
        else:
            if self.pending.shape != (overlap, *frames.shape[1:]) or count < overlap:
                raise RuntimeError("Overlap length or output dimensions changed mid-run.")
            weights = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap))).astype(np.float32)
            scratch = np.empty(frames.shape[1:], dtype=np.float32)
            for index, weight in enumerate(weights):
                np.multiply(self.pending[index], 1.0 - weight, out=output[index])
                np.multiply(frames[index], weight, out=scratch)
                np.add(output[index], scratch, out=output[index])
            output[overlap:] = frames[overlap:count]
        # A copy is essential: a slice would retain the full inference batch.
        self.pending = (
            None if info["last"]
            else np.array(frames[core:core + overlap], dtype=np.float32, copy=True)
        )
        self.written_frames += count
        self.next_start = info["start"] + core
        self.finished = bool(info["last"])
        if self.finished:
            actual_total = info["start"] + info["length"]
            if self.written_frames != actual_total or actual_total != info["decoded_frames"]:
                raise RuntimeError("Frame accounting failed. The output must not be used.")
        return output


def close_guard(close_function):
    """VHS BatchManager.close_inputs() sends 1 to the last tuple element."""
    try:
        while (yield None) is None:
            pass
    finally:
        close_function()

