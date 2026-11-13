"""Compare box/moment geometry on explicit development windows, without labels.

Run from the repository root with PYTHONPATH=src. Statistics describe variation,
not accuracy; adjacent frames can contain real changes in posture and orientation.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.geometry import canonical_ellipse
from nvsl_offline.pipeline import assert_sources, create_run, csv_write, load_run, producer, write_image
from nvsl_offline.recording import json_write
from nvsl_offline.replay import FrameReader, fit_text


def describe_pairs(rows, first, last, slot):
    selected = [r for r in rows if r["interval_start"] == first and r["interval_stop"] == last
                and r["slot"] == slot and r["accepted"]]
    if not selected:
        return dict(interval=[first, last], slot=slot, accepted_frames=0)
    indices = np.array([r["frame"] for r in selected])
    adjacent = np.diff(indices) == 1
    result = dict(interval=[first, last], slot=slot, accepted_frames=len(selected), adjacent_pairs=int(adjacent.sum()))
    for name in ("box", "moments"):
        dims = np.array([[r[f"{name}_w"], r[f"{name}_h"]] for r in selected])
        angles = np.array([r[f"{name}_theta"] for r in selected])
        angle_steps = np.abs((np.diff(angles)+90)%180-90)[adjacent]
        dim_steps = np.abs(np.diff(dims, axis=0))[adjacent]
        result[name] = dict(median_minor_major_px=np.median(dims, axis=0).tolist(),
                           median_absolute_axis_step_px=np.median(dim_steps, axis=0).tolist() if len(dim_steps) else None,
                           median_p95_absolute_axial_angle_step_deg=np.quantile(angle_steps, [.5, .95]).tolist() if len(angle_steps) else None)
    result["median_area_ratio_moments_over_box"] = float(np.median([
        (r["moments_w"]*r["moments_h"])/(r["box_w"]*r["box_h"]) for r in selected]))
    return result


def snapshot(frame, old, new, cal, index, output):
    slot = 0
    x0, y0, x1, y1 = cal.roi(slot)
    panels = []
    for title, detection, color in [("Normalized + box", old[slot], (0, 220, 255)),
                                     ("Normalized + weighted moments", new[slot], (80, 255, 80))]:
        image = frame.copy()
        if detection.ellipse is not None:
            cv2.ellipse(image, detection.ellipse, color, 1, cv2.LINE_AA)
        crop = cv2.resize(image[y0:y1, x0:x1], (2*(x1-x0), 2*(y1-y0)), interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((crop.shape[0]+70, crop.shape[1], 3), np.uint8)
        panel[40:40+crop.shape[0]] = crop
        fit_text(panel, title, (8, 22), color)
        fit_text(panel, f"frame {index}, slot 0; identical center and acceptance", (8, panel.shape[0]-10), (255, 255, 255), .43)
        panels.append(panel)
    write_image(output/f"{index:07d}_shape_comparison.png", np.concatenate(panels, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", action="append", help="START:STOP saved frames, stop exclusive (repeatable)")
    args = parser.parse_args()
    rec, manifest, bg, cal = load_run(args.run)
    intervals = [tuple(map(int, text.split(":"))) for text in (args.interval or
                  ["4800:5000", "5480:5540", "6940:7000", "10000:10200"])]
    if any(len(v) != 2 or not 0 <= v[0] < v[1] <= rec.shape[1] for v in intervals):
        parser.error("Intervals must satisfy 0 <= start < stop <= recording length")
    out = create_run(args.output)
    config = DetectionConfig(mode="normalized", shape_fit="box")
    rows = []
    reader = FrameReader(rec.video)
    try:
        for first, last in intervals:
            for index in range(first, last):
                frame = reader.read(index)
                old, _, mask1 = detect(frame, bg, cal, config)
                new, _, mask2 = detect(frame, bg, cal, replace(config, shape_fit="weighted_moments"))
                np.testing.assert_array_equal(mask1, mask2)
                for a, b in zip(old, new):
                    assert a.reason == b.reason and (a.ellipse is None) == (b.ellipse is None)
                    row = dict(interval_start=first, interval_stop=last, frame=index, slot=a.slot,
                               accepted=a.ellipse is not None, shape_method=b.shape_method)
                    for method in ("box", "moments"):
                        row.update({f"{method}_{field}": "" for field in ("w", "h", "theta")})
                    if a.ellipse is not None:
                        assert a.ellipse[0] == b.ellipse[0]
                        for method, ellipse in [("box", a.ellipse), ("moments", b.ellipse)]:
                            _, (w, h), theta = canonical_ellipse(ellipse)
                            row.update({f"{method}_w": w, f"{method}_h": h, f"{method}_theta": theta})
                    rows.append(row)
                if index in (first, first+(last-first)//2, 5500):
                    snapshot(frame, old, new, cal, index, out)
    finally:
        reader.close()
    assert_sources(manifest["sources"])
    csv_write(out/"measurements.csv", rows, list(rows[0]))
    summary = dict(producer=producer(), source_recording=manifest["sources"], intervals=intervals,
                   image_masks_acceptance_and_centers_identical=True,
                   interpretation="Development-window variation, not anatomical accuracy or a held-out benchmark.",
                   comparisons=[describe_pairs(rows, a, b, slot) for a, b in intervals for slot in range(rec.shape[0])])
    json_write(out/"summary.json", summary)
    print(json.dumps(summary["comparisons"][::rec.shape[0]], indent=2))


if __name__ == "__main__":
    main()
