"""Label image-plane fly centers on a selected list of saved video frames."""

import argparse
import csv
from functools import lru_cache
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from nvsl_offline.replay import (
    Comparison, FrameReader, ORIGINAL_COLOR, VARIANT_COLOR, fit_text,
)


WINDOW = "Fly benchmark annotation"
INPUT_FIELDS = ("companion", "fly", "frame")
LABEL_FIELDS = ("video", "status", "x", "y")
STATUSES = ("visible", "no_fly", "unclear")


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or len(set(fields)) != len(fields) or any(
        None in row or None in row.values() for row in rows
    ):
        raise ValueError(f"{path}: expected a nonempty CSV with consistent columns")
    return fields, rows


def read_selection(path):
    fields, rows = read_rows(path)
    if not set(INPUT_FIELDS) <= set(fields) or set(LABEL_FIELDS) & set(fields):
        raise ValueError("Selection needs companion, fly, frame and may include fields such as split and notes")
    for row in rows:
        if not row["companion"].strip():
            raise ValueError("Each selection row needs a companion path")
        companion = (path.parent / row["companion"].strip()).resolve(strict=True)
        fly, frame = int(row["fly"]), int(row["frame"])
        if not 0 <= fly <= 3 or frame < 0:
            raise ValueError("Use zero-based fly slots 0–3 and nonnegative saved frames")
        row.update(companion=str(companion), fly=str(fly), frame=str(frame))
    return [*fields, *LABEL_FIELDS], rows


def load_labels(output, fields, rows):
    if not output.exists():
        return
    saved_fields, saved_rows = read_rows(output)
    if saved_fields != fields or len(saved_rows) != len(rows):
        raise ValueError("Output CSV does not match the selection")
    for current, saved in zip(rows, saved_rows):
        if any(current[key] != saved[key] for key in current):
            raise ValueError("Output CSV does not match the selected rows; use the original selection")
        status, x, y = (saved[k] for k in ("status", "x", "y"))
        if status == "visible":
            if not all(math.isfinite(float(value)) for value in (x, y)):
                raise ValueError("Visible labels require finite x and y coordinates")
        elif status not in ("", "no_fly", "unclear") or x or y:
            raise ValueError("Invalid annotation status or coordinates")
        current.update(status=status, x=x, y=y)


def save_labels(output, fields, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def click_position(x, y, roi, scale):
    left, top, right, bottom = roi
    if not (0 <= x < (right-left)*scale and 0 <= y < (bottom-top)*scale):
        return None
    return left + x//scale, top + y//scale


def annotate(selection, output, *, scale):
    selection, output = Path(selection).resolve(), Path(output).resolve()
    if output == selection or output.suffix.lower() != ".csv":
        raise ValueError("Choose a separate .csv file for annotations")
    fields, rows = read_selection(selection)
    # Validate each source once before starting the annotation session.
    comparisons = lru_cache(maxsize=2)(Comparison.load)
    keys = set()
    for row in sorted(rows, key=lambda item: item["companion"]):
        comp = comparisons(row["companion"])
        fly, frame = int(row["fly"]), int(row["frame"])
        comp.crop(fly)
        if frame >= comp.nframes:
            raise ValueError(f"Frame {frame} is outside {comp.original.video.name}")
        row["video"] = str(comp.original.video)
        key = (row["video"], fly, frame)
        if key in keys:
            raise ValueError(f"Duplicate recording, fly, and frame: {key}")
        keys.add(key)
    load_labels(output, fields, rows)
    if "NONE" in cv2.getBuildInformation().split("GUI:")[-1].splitlines()[0]:
        raise ValueError("Annotation requires GUI-enabled OpenCV")
    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("Annotation requires a desktop display")

    index = next((i for i, row in enumerate(rows) if not row.get("status")), 0)
    offset = 0
    overlay = False
    reader = None
    opened_video = None
    pending_click = None

    def on_mouse(event, x, y, flags, data):
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (x, y)

    def label(row, status, position=None):
        x, y = position if position is not None else ("", "")
        row.update(status=status, x=x, y=y)
        save_labels(output, fields, rows)

    # An autosized window keeps callback positions in displayed image pixels.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            row = rows[index]
            comp = comparisons(row["companion"])
            fly, target = int(row["fly"]), int(row["frame"])
            frame = target + offset
            roi = comp.crop(fly)
            if opened_video != comp.original.video:
                if reader is not None:
                    reader.close()
                reader = FrameReader(comp.original.video)
                opened_video = comp.original.video
            left, top, right, bottom = roi
            crop = reader.read(frame)[top:bottom, left:right]
            image = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if overlay:
                for rec, color in [(comp.original, ORIGINAL_COLOR), (comp.variant, VARIANT_COLOR)]:
                    x, y = (rec.arrays[k][fly, frame] for k in ("x", "y"))
                    if np.isfinite([x, y]).all():
                        center = (round((x-left)*scale), round((y-top)*scale))
                        cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 11, 1)
            if not offset and row.get("status") == "visible":
                center = (round((float(row["x"])-left)*scale), round((float(row["y"])-top)*scale))
                cv2.drawMarker(image, center, (255, 200, 60), cv2.MARKER_CROSS, 15, 1)
            height = image.shape[0]
            image = cv2.copyMakeBorder(image, 0, 83, 0, max(0, 530-image.shape[1]),
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            status = row.get("status") or "unlabeled"
            view = f"target {target}" if not offset else f"context {frame} (target {target})"
            lines = [
                f"{index+1}/{len(rows)}  {comp.original.video.name}  fly {fly}",
                f"{view} | {status}" + (" | tracker centers on" if overlay else ""),
                "Click: visible   a: no fly   u: unclear   x: clear   q: quit",
                "n/p: next/prev target   ,/.: context   t: target   o: overlays",
            ]
            for i, line in enumerate(lines):
                fit_text(image, line, (8, height+17+20*i), (255, 255, 255), .45)
            cv2.imshow(WINDOW, image)
            code = cv2.waitKey(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            click, pending_click = pending_click, None
            if click is not None and not offset:
                pos = click_position(*click, roi, scale)
                if pos is not None:
                    label(row, "visible", pos)
            key = chr(code & 255) if code >= 0 else ""
            if key == "q":
                break
            if key in ("n", "p"):
                index = min(len(rows)-1, max(0, index + (1 if key == "n" else -1)))
                offset, overlay = 0, False
            elif key in (",", "."):
                offset = min(comp.nframes-1, max(0, frame + (1 if key == "." else -1))) - target
            elif key == "t":
                offset = 0
            elif key == "o":
                overlay = not overlay
            elif not offset and key in ("a", "u", "x"):
                label(row, {"a": "no_fly", "u": "unclear", "x": ""}[key])
    finally:
        if reader is not None:
            reader.close()
        cv2.destroyAllWindows()
    print(f"{sum(bool(row.get('status')) for row in rows)}/{len(rows)} labeled. Output: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", help="CSV with companion, fly, and frame columns")
    parser.add_argument("--output", required=True, help="Annotation CSV, reused to resume")
    parser.add_argument("--scale", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    try:
        annotate(args.selection, args.output, scale=args.scale)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
