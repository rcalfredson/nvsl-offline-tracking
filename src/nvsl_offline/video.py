"""Sequential decoding. A full pass is required before background/tracking output."""

from dataclasses import dataclass, asdict
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    header_frames: int


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def probe(path):
    cap = open_video(path)
    try:
        result = VideoInfo(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           float(cap.get(cv2.CAP_PROP_FPS)),
                           int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if min(result.width, result.height) <= 0 or not np.isfinite(result.fps) or result.fps <= 0:
            raise ValueError("Invalid video dimensions or nominal FPS")
        return result
    finally:
        cap.release()


def frames(path, *, info=None):
    info = info or probe(path)
    cap = open_video(path)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (info.height, info.width, 3):
                raise ValueError(f"Frame {index}: changing/unsupported decoded shape {frame.shape}")
            yield index, frame
            index += 1
    finally:
        cap.release()


def sample_indices(nframes, count):
    if count < 3:
        raise ValueError("At least three background samples are required")
    return np.unique(np.linspace(0, nframes-1, min(count, nframes), dtype=int))


def report_progress(message):
    print(message, flush=True)


def decode_samples(path, *, expected_frames, count, sample_path: Path, progress=report_progress):
    info = probe(path)
    if info.header_frames != expected_frames:
        raise ValueError(f"Video header has {info.header_frames} frames; trajectories have {expected_frames}")
    indices = sample_indices(expected_frames, count)
    samples = np.lib.format.open_memmap(sample_path, mode="w+", dtype=np.uint8,
                                      shape=(len(indices), info.height, info.width))
    next_sample = decoded = 0
    start = last_report = time.monotonic()
    for index, frame in frames(path, info=info):
        decoded = index+1
        if index >= expected_frames:
            raise ValueError("Decoded more frames than the trajectories; refusing to trim")
        if next_sample < len(indices) and index == indices[next_sample]:
            samples[next_sample] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_sample += 1
        now = time.monotonic()
        if now-last_report >= 15:
            progress(f"Decoded {decoded}/{expected_frames} saved frames ({decoded/(now-start):.0f} fps)")
            last_report = now
    samples.flush()
    if decoded != expected_frames or next_sample != len(indices):
        raise ValueError(f"Decoded {decoded} frames, expected {expected_frames}; refusing partial output")
    return dict(video=asdict(info), decoded_frames=decoded, sample_indices=indices.tolist(),
                decode_seconds=time.monotonic()-start)
