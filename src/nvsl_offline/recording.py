"""Trusted legacy recording I/O and explicit, independently defined calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np

FIELDS = ("x", "y", "w", "h", "theta")


class LegacyFlyDetector:
    """State-only placeholder for Python 2 acquisition metadata."""


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FlyDetector":
            return LegacyFlyDetector
        return super().find_class(module, name)


def load_pickle(path: Path):
    # Pickle can execute code: only open recordings from trusted acquisition sources.
    with path.open("rb") as stream:
        return LegacyUnpickler(stream, encoding="latin1").load()


def identity(path: Path, *, content_hash=False):
    stat = path.stat()
    result = dict(path=str(path.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if content_hash:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def json_write(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def round_acquisition(value):
    """Python 2 round/intR behavior, including negative half ties."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def sidecar(video: Path, suffix: str) -> Path:
    choices = [video.with_suffix(suffix), Path(str(video) + suffix)]
    present = [p for p in choices if p.is_file()]
    if len(present) != 1:
        raise ValueError(f"Expected exactly one {suffix} sidecar for {video}; found {present}")
    return present[0]


@dataclass
class Recording:
    video: Path
    trx_path: Path
    data_path: Path
    raw: dict
    data: dict
    arrays: dict[str, np.ndarray]
    ts: np.ndarray

    @classmethod
    def load(cls, video, *, trx=None, data=None):
        video = Path(video).resolve(strict=True)
        trx_path = Path(trx).resolve(strict=True) if trx else sidecar(video, ".trx")
        data_path = Path(data).resolve(strict=True) if data else sidecar(video, ".data")
        raw, metadata = load_pickle(trx_path), load_pickle(data_path)
        if not isinstance(raw, dict) or not isinstance(metadata, dict):
            raise ValueError("Trajectory and experiment metadata must be dictionaries")
        try:
            arrays = {k: np.asarray(raw[k], dtype=np.float64) for k in FIELDS}
            ts = np.asarray(raw["ts"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid legacy trajectory fields: {exc}") from exc
        shapes = {a.shape for a in arrays.values()}
        if len(shapes) != 1 or arrays["x"].ndim != 2 or 0 in arrays["x"].shape:
            raise ValueError("Geometry must have matching nonempty (flies, frames) shapes")
        if ts.shape != (arrays["x"].shape[1],) or not np.isfinite(ts).all():
            raise ValueError("ts must contain one finite timestamp per saved frame")
        x_missing = np.isnan(arrays["x"])
        for key, a in arrays.items():
            if np.isinf(a).any() or not np.array_equal(np.isnan(a), x_missing):
                raise ValueError(f"{key}: inconsistent geometry missing mask or infinite values")
        if any(np.any(a[np.isfinite(a)] <= 0) for a in (arrays["w"], arrays["h"])):
            raise ValueError("Measured body diameters must be positive")
        if not isinstance(metadata.get("protocol"), dict):
            raise ValueError("Missing protocol dictionary")
        rec = cls(video, trx_path, data_path, raw, metadata, arrays, ts)
        rec.event_summary()  # Validate before decoding or creating output.
        return rec

    @property
    def shape(self):
        return self.arrays["x"].shape

    @property
    def protocol(self):
        return self.data["protocol"]

    def sources(self, *, hash_video=False):
        return {"video": identity(self.video, content_hash=hash_video),
                "trx": identity(self.trx_path, content_hash=True),
                "data": identity(self.data_path, content_hash=True)}

    def event_summary(self):
        groups = self.protocol.get("frameNums")
        if isinstance(groups, dict):
            groups = [groups]
        if not isinstance(groups, list) or not groups:
            raise ValueError("protocol.frameNums must be a dictionary or nonempty list")
        count = boundaries = 0
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Each frameNums group must be a dictionary")
            for name, indices in group.items():
                values = np.asarray(indices)
                if values.ndim != 1:
                    raise ValueError(f"Event {name}: expected a sequence of frame indices")
                if values.size and (not np.issubdtype(values.dtype, np.integer)
                                    or np.any(values < 0) or np.any(values > self.shape[1])):
                    raise ValueError(f"Event {name}: invalid saved-frame indices")
                if values.size and np.any(np.diff(values) < 0):
                    raise ValueError(f"Event {name}: indices are not ordered")
                count += len(values)
                boundaries += int(np.count_nonzero(values == self.shape[1]))
        # frameNum() is the NEXT saved index; N is a valid endpoint at shutdown.
        return dict(count=count, endpoint_events=boundaries, valid_index_interval=[0, self.shape[1]])

    def summary(self):
        dt = np.diff(self.ts)
        flies = []
        for slot, x in enumerate(self.arrays["x"]):
            missing = np.isnan(x)
            edges = np.diff(np.r_[False, missing, False].astype(int))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            lengths = stops - starts
            longest = sorted(zip(starts.tolist(), stops.tolist()), key=lambda p: p[1]-p[0], reverse=True)[:5]
            flies.append(dict(slot=slot, missing_frames=int(missing.sum()),
                              missing_fraction=float(missing.mean()),
                              initial_missing=int(stops[0]) if len(starts) and starts[0] == 0 else 0,
                              gap_count=len(starts), longest_gap=int(lengths.max()) if len(lengths) else 0,
                              longest_gaps=[dict(start=a, stop=b) for a, b in longest]))
        return dict(flies=flies, events=self.event_summary(),
                    timestamps=dict(first=float(self.ts[0]), last=float(self.ts[-1]),
                                    zero_steps=int(np.count_nonzero(dt == 0)),
                                    negative_steps=int(np.count_nonzero(dt < 0)),
                                    minimum_step=float(dt.min()) if len(dt) else None,
                                    maximum_step=float(dt.max()) if len(dt) else None))


@dataclass
class Calibration:
    layout: str
    width: int
    height: int
    x: float
    y: float
    scale: float
    floor_width: int
    offset: int
    separator: int
    margin: int

    @classmethod
    def from_recording(cls, rec: Recording, width, height, *, layout, margin=24):
        if rec.shape[0] != 4 or rec.protocol.get("ct") != "large":
            raise ValueError("This baseline currently supports four-slot large-chamber recordings only")
        if layout not in ("large", "large2"):
            raise ValueError("Select large (35 mm) or large2 (39 mm) explicitly")
        if margin < 0:
            raise ValueError("Sidewall margin must be nonnegative")
        tm = rec.protocol.get("tm", {})
        try:
            x, y, factor = (float(tm[k]) for k in ("x", "y", "fctr"))
            resize = float(tm.get("rszFctr", 1.5 if height == 1080 else 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A valid saved protocol.tm transform is required") from exc
        if not np.isfinite([x, y, factor, resize]).all() or min(factor, resize) <= 0:
            raise ValueError("Invalid template transform")
        floor, offset, separator = (295, 337, 321) if layout == "large2" else (244, 284, 268)
        cal = cls(layout, width, height, x, y, factor*resize, floor, offset, separator, margin)
        sx, sy = cal.separators
        if not (0 < sx < width and 0 < sy < height):
            raise ValueError("Saved chamber separators are outside the video")
        for i in range(4):
            x0, y0, x1, y1 = cal.floor(i)
            if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
                raise ValueError(f"Slot {i}: saved floor does not fit video; verify layout/transform")
        return cal

    @property
    def separators(self):
        return (round_acquisition(self.x + self.separator*self.scale),
                round_acquisition(self.y + self.separator*self.scale))

    def floor(self, slot):
        row, col = divmod(slot, 2)
        origin = 5 if self.layout == "large2" else 4
        x0 = self.x + (origin + col*self.offset)*self.scale
        y0 = self.y + (origin + row*self.offset)*self.scale
        return x0, y0, x0+self.floor_width*self.scale, y0+self.floor_width*self.scale

    def roi(self, slot):
        x0, y0, x1, y1 = self.floor(slot)
        sx, sy = self.separators
        row, col = divmod(slot, 2)
        # bisect_left assigns a center exactly on the separator to the lower slot.
        left, right = (0, sx+1) if col == 0 else (sx+1, self.width)
        top, bottom = (0, sy+1) if row == 0 else (sy+1, self.height)
        return (max(left, math.floor(x0)-self.margin), max(top, math.floor(y0)-self.margin),
                min(right, math.ceil(x1)+self.margin+1), min(bottom, math.ceil(y1)+self.margin+1))
