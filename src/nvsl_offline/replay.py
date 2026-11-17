"""Synchronized raw-trajectory comparison, independent of analysis interpolation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import cv2
import numpy as np

from .pipeline import assert_sources
from .recording import Calibration, FIELDS, Recording, identity, json_write, load_pickle
from .video import open_video, probe

ORIGINAL_COLOR = (0, 220, 255)
VARIANT_COLOR = (80, 255, 80)
INFERRED_COLOR = (255, 140, 255)
WINDOW = "Original / offline comparison"
HELP = ["Space play/pause | ,/. step | -/+ speed | f cycle fly | a all flies",
        "n/p original gaps | r/b new tracking | e ellipses | c centers | s snapshot",
        "Type frame or MM:SS[.fraction], then g/Enter | Esc clear input | q quit"]


def parse_position(value, fps):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Use an integer saved frame or MM:SS / HH:MM:SS[.fraction]")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("Invalid frame/time position") from exc
    if (not np.isfinite(numbers).all() or any(n < 0 for n in numbers)
            or any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1])):
        raise ValueError("Invalid clock time")
    seconds = 0.
    for number in numbers:
        seconds = seconds*60+number
    return int(math.floor(seconds*fps+.5))


def run_starts(mask):
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def next_start(starts, current, direction, start, stop):
    available = starts[(starts >= start) & (starts < stop)]
    available = available[available > current] if direction > 0 else available[available < current]
    return int(available[0 if direction > 0 else -1]) if len(available) else None


def clock(seconds):
    return f"{int(seconds//3600):02d}:{int(seconds//60%60):02d}:{seconds%60:05.2f}"


@dataclass
class Comparison:
    original: Recording
    variant: Recording
    fps: float
    width: int
    height: int
    calibration: Calibration | None
    variant_label: str
    status: np.ndarray | None
    status_names: dict
    reasons: np.ndarray | None
    reason_names: dict

    @classmethod
    def load(cls, companion, *, video=None, original=None, data=None, layout=None, label=None):
        companion = Path(companion).resolve(strict=True)
        payload = load_pickle(companion)
        if not isinstance(payload, dict):
            raise ValueError("Comparison trajectory must be a dictionary")
        metadata = payload.get("tracking_metadata", {})
        sources = metadata.get("source_recording", {})
        if video is None:
            video = sources.get("video", {}).get("path")
        if video is None:
            raise ValueError("--video is required for trajectories without source metadata")
        original = original or sources.get("trx", {}).get("path")
        data = data or sources.get("data", {}).get("path")
        base = Recording.load(video, trx=original, data=data)
        if base.trx_path.resolve() == companion:
            raise ValueError("Select a companion different from the original trajectory")
        if sources:
            assert_sources(sources)
            # Explicit overrides must still select the recorded acquisition source.
            for name, path in [("video", base.video), ("trx", base.trx_path), ("data", base.data_path)]:
                expected = sources.get(name)
                if expected is None or identity(path, content_hash="sha256" in expected) != expected:
                    raise ValueError(f"Selected {name} does not match companion source metadata")
        other = Recording.load(base.video, trx=companion, data=base.data_path)
        if other.shape != base.shape or not np.array_equal(other.ts, base.ts):
            raise ValueError("Original and companion must have identical fly/frame shapes and timestamps")
        info = probe(base.video)
        if info.header_frames != base.shape[1]:
            raise ValueError("Video header frame count does not match trajectories")
        saved_cal = metadata.get("configuration", {}).get("calibration", {})
        layout = layout or saved_cal.get("layout")
        calibration = None
        if layout:
            calibration = Calibration.from_recording(base, info.width, info.height,
                                                     layout=layout, margin=saved_cal.get("margin", 24))
        quality = payload.get("tracking_quality", {})
        status = cls.quality_array(quality, "status", base.shape)
        reasons = cls.quality_array(quality, "reason", base.shape)
        names = {int(v): k for k, v in quality.get("status_codes", {}).items()}
        reason_names = {int(v): k for k, v in quality.get("reason_codes", {}).items()}
        if status is not None:
            if not names or not np.isin(status, list(names)).all():
                raise ValueError("Companion contains unknown tracking status codes")
            missing_code = quality.get("status_codes", {}).get("missing")
            if missing_code is None or not np.array_equal(status == missing_code, np.isnan(other.arrays["x"])):
                raise ValueError("Companion status and measurement missing masks disagree")
        detector = metadata.get("configuration", {}).get("detector", {})
        mode = detector.get("mode", "selected variant")
        if mode == "absolute":
            mode = f"absolute threshold {detector.get('threshold', '?'):g}" if isinstance(detector.get("threshold"), (int, float)) else mode
        if detector.get("shape_fit", "box") != "box":
            mode += " + " + detector["shape_fit"].replace("_", " ")
        if detector.get("illumination", "none") != "none":
            mode += " + " + detector["illumination"].replace("_", " ")
        if detector.get("segmentation", "components") != "components":
            mode += " + " + detector["segmentation"].replace("_", " ")
        return cls(base, other, info.fps, info.width, info.height, calibration,
                   label or mode, status, names, reasons, reason_names)

    @staticmethod
    def quality_array(quality, key, shape):
        if quality.get(key) is None:
            return None
        value = np.asarray(quality[key])
        if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"Companion quality {key} must have matching integer fly/frame shape")
        return value

    @property
    def nframes(self):
        return self.original.shape[1]

    def provenance(self, slot, frame):
        if self.status is not None:
            return self.status_names[int(self.status[slot, frame])]
        return "finite; provenance unknown" if np.isfinite(self.variant.arrays["x"][slot, frame]) else "missing"

    def starts(self, fly, kind):
        slots = [fly] if fly is not None else list(range(self.original.shape[0]))
        masks = np.isnan(self.original.arrays["x"][slots])
        if kind == "new":
            masks &= np.isfinite(self.variant.arrays["x"][slots])
            if self.status is not None:
                # Prediction/interpolation must not be presented as image-based recovery.
                image_codes = [code for code, name in self.status_names.items() if name in ("observed", "recovered")]
                masks &= np.isin(self.status[slots], image_codes)
        # Keep individual slot run boundaries, even when other flies are missing.
        return np.unique(np.concatenate([run_starts(mask) for mask in masks]))

    def crop(self, fly):
        if fly is None:
            return 0, 0, self.width, self.height
        if fly not in range(self.original.shape[0]):
            raise ValueError("Physical fly slot is outside the trajectory")
        if self.calibration is None:
            raise ValueError("A saved calibration or explicit --layout is needed for a fixed chamber crop")
        return self.calibration.roi(fly)

    def render(self, frame, index, *, fly=None, scale=None, ellipses=True, centers=True):
        if frame.shape != (self.height, self.width, 3) or not 0 <= index < self.nframes:
            raise ValueError("Decoded frame/index does not match comparison recording")
        crop = self.crop(fly)
        scale = scale if scale is not None else (2.0 if fly is not None else 1.0)
        if not np.isfinite(scale) or not .25 <= scale <= 8:
            raise ValueError("Display scale must be between 0.25 and 8")
        slots = [fly] if fly is not None else list(range(self.original.shape[0]))
        x0, y0, x1, y1 = crop
        width, height = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
        if width < 160 or width > 8192 or height > 8192:
            raise ValueError("Choose a scale producing panels between 160 and 8192 pixels wide")
        panels = []
        for variant, rec, color in [(False, self.original, ORIGINAL_COLOR), (True, self.variant, VARIANT_COLOR)]:
            image = frame[y0:y1, x0:x1].copy()
            # Resize both copies identically BEFORE drawing: keeps thin ellipse and
            # center markers legible without changing the stored measurement.
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
            lines = []
            for slot in slots:
                vals = [rec.arrays[k][slot, index] for k in FIELDS]
                valid = np.isfinite(vals).all()
                state = self.provenance(slot, index) if variant else ("measured" if valid else "missing")
                reason = ""
                if variant and self.reasons is not None and not valid:
                    reason = self.reason_names.get(int(self.reasons[slot, index]), "unknown reason")
                lines.append(f"slot {slot}: {state.upper()}" + (f" ({reason})" if reason else ""))
                if valid:
                    x, y, w, h, theta = vals
                    current_color = INFERRED_COLOR if state == "interpolated" else color
                    position = ((x-x0)*scale, (y-y0)*scale)
                    if ellipses:
                        cv2.ellipse(image, (position, (w*scale, h*scale), theta), current_color, 1, cv2.LINE_AA)
                    if centers:
                        cv2.drawMarker(image, tuple(round(v) for v in position), current_color,
                                       cv2.MARKER_CROSS, 7, 1)
                    if fly is not None:
                        lines.append(f"x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f} theta={theta:.1f}")
            panel = np.zeros((height+134, width, 3), np.uint8)
            panel[48:height+48] = image
            title = "ORIGINAL .trx" if not variant else f"OFFLINE: {self.variant_label}"
            fit_text(panel, title, (8, 20), color)
            fit_text(panel, "raw measurements; no analysis interpolation" if not variant else
                     "candidate provenance is not an accuracy judgment", (8, 39), (200, 200, 200), .39)
            for j, line in enumerate(lines):
                fit_text(panel, line, (8, height+66+j*18), (255, 255, 255), .42)
            panels.append(panel)
        pair = np.concatenate(panels, axis=1)
        canvas = np.zeros((pair.shape[0]+112, pair.shape[1], 3), np.uint8)
        canvas[36:36+pair.shape[0]] = pair
        fit_text(canvas, f"saved frame {index}/{self.nframes-1} | video {clock(index/self.fps)} | source ts {self.original.ts[index]:.3f}s",
                 (8, 23), (255, 255, 255))
        for j, line in enumerate(HELP):
            fit_text(canvas, line, (8, canvas.shape[0]-52+j*18), (190, 190, 190), .43)
        return canvas


def fit_text(image, text, origin, color, size=.52):
    width = image.shape[1]-origin[0]-6
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] > width and len(text) > 3:
        text = text[:-4]+"..."
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


class FrameReader:
    """One decoder supplies both panels, including after backwards/random seeks."""

    def __init__(self, path):
        self.cap = open_video(path)
        self.index, self.frame = -1, None

    def read(self, index):
        if index == self.index:
            return self.frame
        if index != self.index+1 and not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Decoder cannot seek to saved frame {index}")
        ok, frame = self.cap.read()
        if not ok or abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1)) > .5:
            raise ValueError(f"Could not decode requested saved frame {index}")
        self.index, self.frame = index, frame
        return frame

    def close(self):
        self.cap.release()


@dataclass
class Controls:
    start: int
    stop: int
    index: int
    fly: int | None
    speed: float = 1.
    paused: bool = True
    ellipses: bool = True
    centers: bool = True
    text: str = ""
    notice: str = ""

    def seek(self, index):
        if self.start <= index < self.stop:
            self.index, self.paused = index, True
            self.notice = ""
        else:
            self.notice = f"Frame must be in [{self.start}, {self.stop})"

    def key(self, key, comparison):
        if key == "q":
            return "quit"
        self.notice = ""
        if key == " ":
            if self.index == self.stop-1 and self.paused:
                self.index = self.start
            self.paused = not self.paused
        elif key in (",", "."):
            if key == "." and self.text:
                self.text += key
            else:
                self.seek(self.index+(1 if key == "." else -1))
        elif key in ("g", "\r", "\n"):
            try:
                self.seek(parse_position(self.text, comparison.fps))
            except ValueError as exc:
                self.notice = str(exc)
            self.text = ""
        elif key in ("\b", "\x7f"):
            self.text = self.text[:-1]
        elif key == "\x1b":
            self.text = ""
        elif key in "0123456789:":
            self.text += key
        elif key in ("+", "=", "-"):
            self.speed = min(32., max(.0625, self.speed*(.5 if key == "-" else 2.)))
        elif key == "e":
            self.ellipses = not self.ellipses
        elif key == "c":
            self.centers = not self.centers
        elif key in ("f", "a"):
            if key == "a":
                self.fly = None
            elif comparison.calibration is None:
                self.notice = "Chamber crop requires saved calibration or --layout"
            else:
                self.fly = 0 if self.fly is None else ((self.fly+1) if self.fly+1 < comparison.original.shape[0] else None)
        elif key in ("n", "p", "r", "b"):
            starts = comparison.starts(self.fly, "original" if key in ("n", "p") else "new")
            target = next_start(starts, self.index, 1 if key in ("n", "r") else -1, self.start, self.stop)
            if target is None:
                self.notice = "No further matching intervals in playback range"
            else:
                self.seek(target)
        elif key == "s":
            return "snapshot"
        return None


def save_snapshot(path, image, comparison, state):
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Snapshot must have a .png extension")
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise ValueError("Snapshot or its metadata already exists")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Could not encode snapshot")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    json_write(metadata_path, dict(video=str(comparison.original.video), original=str(comparison.original.trx_path),
                                  variant=str(comparison.variant.trx_path), frame=state.index,
                                  physical_slot=state.fly, ellipses=state.ellipses, centers=state.centers,
                                  source_ts=float(comparison.original.ts[state.index])))


def export_clip(comparison, path, *, start, stop, fly=None, scale=None, speed=1.):
    path = Path(path).resolve()
    if path.suffix.lower() != ".mp4" or path.exists():
        raise ValueError("Export requires a new .mp4 filename")
    if not 0 <= start < stop <= comparison.nframes or not np.isfinite(speed) or not .0625 <= speed <= 32:
        raise ValueError("Invalid export interval or speed")
    path.parent.mkdir(parents=True, exist_ok=True)
    reader, writer, temporary = FrameReader(comparison.original.video), None, None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mp4", delete=False) as f:
            temporary = Path(f.name)
        for index in range(start, stop):
            image = comparison.render(reader.read(index), index, fly=fly, scale=scale)
            if image.shape[0] % 2 or image.shape[1] % 2:
                image = cv2.copyMakeBorder(image, 0, image.shape[0] % 2, 0, image.shape[1] % 2, cv2.BORDER_CONSTANT)
            if writer is None:
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                                         comparison.fps*speed, image.shape[1::-1])
                if not writer.isOpened():
                    raise ValueError("Cannot open MP4 encoder")
            writer.write(image)
        writer.release()
        writer = None
        encoded = probe(temporary)
        if encoded.header_frames != stop-start:
            raise ValueError("Encoded comparison clip is incomplete; no export published")
        os.link(temporary, path)
    finally:
        reader.close()
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink()
    return dict(output=str(path), saved_frame_start=start, saved_frame_stop_exclusive=stop,
                exported_frames=stop-start, playback_fps=comparison.fps*speed)


def play(comparison, *, start, stop, fly=None, scale=None, speed=1., autoplay=False,
         snapshot_dir="runs/comparison-snapshots"):
    if "NONE" in cv2.getBuildInformation().split("GUI:")[-1].splitlines()[0]:
        raise ValueError("Interactive replay needs GUI-enabled OpenCV; use --output for headless export")
    if os.name == "posix" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise ValueError("No display available; use --output or --snapshot for headless inspection")
    state = Controls(start, stop, start, fly, speed, paused=not autoplay)
    reader = FrameReader(comparison.original.video)
    slider = {"setting": False, "requested": None}
    def on_seek(value):
        if not slider["setting"]:
            slider["requested"] = value
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        slider["setting"] = True
        cv2.createTrackbar("saved frame", WINDOW, start, comparison.nframes-1, on_seek)
        slider["setting"] = False
        deadline = time.monotonic()
        print("\n".join(HELP), flush=True)
        previous_view, image = None, None
        while True:
            if slider["requested"] is not None:
                if slider["requested"] != state.index:
                    state.seek(slider["requested"])
                slider["requested"] = None
                slider["setting"] = True
                cv2.setTrackbarPos("saved frame", WINDOW, state.index)
                slider["setting"] = False
            view = (state.index, state.fly, state.ellipses, state.centers)
            if view != previous_view:
                image = comparison.render(reader.read(state.index), state.index, fly=state.fly,
                                          scale=scale, ellipses=state.ellipses, centers=state.centers)
                if previous_view is None:
                    # Size once from the rendered aspect ratio; retain subsequent user resizing.
                    cv2.resizeWindow(WINDOW, 2500, round(2500*image.shape[0]/image.shape[1]))
                previous_view = view
                slider["setting"] = True
                cv2.setTrackbarPos("saved frame", WINDOW, state.index)
                slider["setting"] = False
            display = image.copy()
            # Status strip is between the image panels and the static controls.
            message = f"{'PAUSED' if state.paused else 'PLAYING'} {state.speed:g}x"
            message += f" | seek: {state.text}" if state.text else ""
            message += f" | {state.notice}" if state.notice else ""
            fit_text(display, message, (8, display.shape[0]-70), (255, 255, 255), .43)
            cv2.imshow(WINDOW, display)
            code = cv2.waitKey(10)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if code >= 0:
                action = state.key(chr(code & 255), comparison)
                deadline = time.monotonic()
                if action == "quit":
                    break
                if action == "snapshot":
                    name = f"frame{state.index:07d}_slot{state.fly if state.fly is not None else 'all'}_{uuid.uuid4().hex[:8]}.png"
                    save_snapshot(Path(snapshot_dir)/name, image, comparison, state)
                    state.notice = f"Saved {name}"
            if not state.paused and time.monotonic()-deadline >= 1/(comparison.fps*state.speed):
                if state.index+1 == stop:
                    state.paused = True
                else:
                    state.index += 1
                deadline = time.monotonic()  # Slow preview if necessary; never skip saved frames.
    finally:
        reader.close()
        cv2.destroyAllWindows()


def compare(companion, *, video=None, original=None, data=None, layout=None, label=None,
            start="0", end=None, fly=None, scale=None, speed=1., autoplay=False,
            output=None, snapshot=None, snapshot_dir="runs/comparison-snapshots"):
    comparison = Comparison.load(companion, video=video, original=original, data=data,
                                 layout=layout, label=label)
    first = parse_position(start, comparison.fps)
    stop = comparison.nframes if end is None else parse_position(end, comparison.fps)
    if not 0 <= first < stop <= comparison.nframes:
        raise ValueError(f"Require 0 <= start < end <= {comparison.nframes}")
    if not np.isfinite(speed) or not .0625 <= speed <= 32:
        raise ValueError("Playback speed must be between 0.0625 and 32")
    comparison.crop(fly)
    if snapshot and output:
        raise ValueError("Choose either a snapshot or a video export")
    if output:
        return export_clip(comparison, output, start=first, stop=stop, fly=fly, scale=scale, speed=speed)
    if snapshot:
        reader = FrameReader(comparison.original.video)
        try:
            image = comparison.render(reader.read(first), first, fly=fly, scale=scale)
            save_snapshot(snapshot, image, comparison, Controls(first, stop, first, fly))
        finally:
            reader.close()
        return dict(snapshot=str(snapshot), saved_frame=first)
    play(comparison, start=first, stop=stop, fly=fly, scale=scale, speed=speed,
         autoplay=autoplay, snapshot_dir=snapshot_dir)
    return dict(last_action="player closed")
