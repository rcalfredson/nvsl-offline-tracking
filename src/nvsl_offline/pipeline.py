"""Audited runs, review artifacts and optional full-length companion trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import cv2
import numpy as np

from . import __version__
from .background import Background, BackgroundConfig
from .detection import DetectionConfig, detect
from .recording import Calibration, FIELDS, Recording, identity, json_write
from .video import decode_samples, frames, probe


def producer():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for p in sorted(package.glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=package, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(name="nvsl-offline-tracking", version=__version__, git_commit=commit,
                working_tree_dirty=dirty, package_sha256=digest.hexdigest(),
                numpy_version=np.__version__, opencv_version=cv2.__version__)


def assert_sources(sources):
    for src in sources.values():
        now = identity(Path(src["path"]), content_hash="sha256" in src)
        if now != src:
            raise ValueError(f"Source changed since audit: {src['path']}")


def create_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def audit(video, output, *, layout, margin=24, samples=129, config=BackgroundConfig(),
          hash_video=False, trx=None, data=None):
    config.validate()
    rec = Recording.load(video, trx=trx, data=data)
    info = probe(rec.video)
    cal = Calibration.from_recording(rec, info.width, info.height, layout=layout, margin=margin)
    if rec.shape[1] < 3 or samples < 3:
        raise ValueError("Background estimation needs at least three saved frames and samples")
    sources = rec.sources(hash_video=hash_video)
    out = create_run(output)
    try:
        decoded = decode_samples(rec.video, expected_frames=rec.shape[1], count=samples,
                                 sample_path=out/"samples.npy")
        sample_stack = np.load(out/"samples.npy", mmap_mode="r", allow_pickle=False)
        start = time.monotonic()
        background = Background.fit(sample_stack, config)
        background.save(out/"background.npz")
        for name, array in [("background_median", background.median),
                            ("background_quantile", background.estimate)]:
            write_image(out/f"{name}.png", np.clip(array, 0, 255).astype(np.uint8))
        noise_image = cv2.applyColorMap(np.clip(background.noise/25*255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        write_image(out/"noise_0_to_25.png", noise_image)
        assert_sources(sources)
        manifest = dict(schema_version=1, complete=True, producer=producer(), sources=sources,
                        calibration=asdict(cal), background_config=asdict(config), channel="opencv_bgr_to_gray",
                        background_sha256=identity(out/"background.npz", content_hash=True)["sha256"],
                        **decoded, background_seconds=time.monotonic()-start, recording=rec.summary(),
                        limitations=["Layout and sidewall margins require visual confirmation.",
                                     "Temporal MAD includes illumination drift; it is not pure sensor noise.",
                                     "Background estimates are unvalidated and may contain stationary bodies."])
        json_write(out/"manifest.json", manifest)
        return manifest
    except Exception as exc:
        json_write(out/"FAILED.json", dict(error=str(exc), complete=False))
        raise


def load_run(path):
    out = Path(path)
    manifest = json.loads((out/"manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("A completed, supported audit manifest is required")
    if identity(out/"background.npz", content_hash=True)["sha256"] != manifest.get("background_sha256"):
        raise ValueError("Background checksum missing or changed; create a new audit run")
    assert_sources(manifest["sources"])
    src = manifest["sources"]
    rec = Recording.load(src["video"]["path"], trx=src["trx"]["path"], data=src["data"]["path"])
    background = Background.load(out/"background.npz")
    expected = (manifest["video"]["height"], manifest["video"]["width"])
    if (any(a.shape != expected or not np.isfinite(a).all() for a in
            (background.median, background.estimate, background.noise)) or np.any(background.noise <= 0)):
        raise ValueError("Invalid background arrays")
    return rec, manifest, background, Calibration(**manifest["calibration"])


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def overlay(frame, rec, cal, index, detections):
    image = frame.copy()
    sx, sy = cal.separators
    cv2.line(image, (sx, 0), (sx, cal.height-1), (255, 200, 0), 1)
    cv2.line(image, (0, sy), (cal.width-1, sy), (255, 200, 0), 1)
    for slot, detection in enumerate(detections):
        roi = cal.roi(slot)
        cv2.rectangle(image, roi[:2], (roi[2]-1, roi[3]-1), (200, 200, 0), 1)
        floor = np.rint(cal.floor(slot)).astype(int)
        cv2.rectangle(image, tuple(floor[:2]), tuple(floor[2:]), (255, 100, 0), 1)
        for candidate in detection.candidates:
            cv2.drawContours(image, [candidate.contour], -1, (0, 165, 255), 1)
        if np.isfinite(rec.arrays["x"][slot, index]):
            x, y, w, h, theta = [rec.arrays[k][slot, index] for k in FIELDS]
            cv2.ellipse(image, ((x, y), (w, h), theta), (0, 0, 255), 1)
        if detection.ellipse is not None:
            cv2.ellipse(image, detection.ellipse, (0, 255, 0), 1)
        cv2.putText(image, f"{slot}: {detection.reason}", (roi[0]+3, roi[1]+17),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return image


def diagnostic_sheet(frame, rec, cal, index, background, results, residual, mask, config):
    noise_score = residual/background.noise
    residual_image = cv2.applyColorMap(np.clip((residual+50)/100*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    score_image = cv2.applyColorMap(np.clip(noise_score/10*255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    noise_image = cv2.applyColorMap(np.clip(background.noise/25*255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    tiles = [(overlay(frame, rec, cal, index, results), "red: original; green: accepted; orange: candidates"),
             (cv2.cvtColor(background.estimate.astype(np.uint8), cv2.COLOR_GRAY2BGR), "recording-wide background"),
             (residual_image, "signed residual: -50 to +50 gray levels"),
             (score_image, "residual / temporal noise: 0 to 10"),
             (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), f"{config.mode} mask (before chamber ROI)"),
             (noise_image, "temporal MAD noise scale: 0 to 25 gray levels")]
    canvas = np.zeros((2*(cal.height+42), 3*cal.width, 3), np.uint8)
    for j, (tile, label) in enumerate(tiles):
        y, x = (j//3)*(cal.height+42), (j%3)*cal.width
        canvas[y:y+cal.height, x:x+cal.width] = tile
        cv2.putText(canvas, label, (x+4, y+cal.height+16), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(canvas, f"saved frame {index}; slot numbering is physical", (x+4, y+cal.height+33),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return canvas


def csv_write(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect(run, output, *, indices, configs):
    rec, manifest, background, cal = load_run(run)
    indices = sorted(set(indices))
    if not indices or indices[0] < 0 or indices[-1] >= rec.shape[1]:
        raise ValueError("Review frames must be nonempty saved-frame indices inside the recording")
    for config in configs:
        config.validate()
    if len({c.mode for c in configs}) != len(configs):
        raise ValueError("Only one configuration per detector mode per review run")
    out = create_run(output)
    rows, candidates, annotations = [], [], []
    wanted = set(indices)
    for index, frame in frames(rec.video):
        if index not in wanted:
            if index > indices[-1]:
                break
            continue
        wanted.remove(index)
        write_image(out/f"{index:07d}_frame.png", frame)
        for slot in range(4):
            annotations.append(dict(video=str(rec.video), frame=index, slot=slot, visibility="",
                                    x="", y="", major_diameter="", minor_diameter="", axis_theta="", notes=""))
        for config in configs:
            results, residual, mask = detect(frame, background, cal, config)
            write_image(out/f"{index:07d}_{config.mode}.jpg",
                        diagnostic_sheet(frame, rec, cal, index, background, results, residual, mask, config))
            for result in results:
                row = dict(frame=index, slot=result.slot, mode=config.mode, reason=result.reason,
                           shape_method=result.shape_method,
                           original_missing=bool(np.isnan(rec.arrays["x"][result.slot, index])),
                           candidate_count=len(result.candidates), runner_up_ratio=result.runner_up_ratio,
                           **{k: "" for k in FIELDS}, **result.rejected_counts)
                if result.ellipse is not None:
                    (row["x"], row["y"]), (row["w"], row["h"]), row["theta"] = result.ellipse
                rows.append(row)
                for rank, candidate in enumerate(result.candidates):
                    (x, y), (w, h), theta = candidate.ellipse
                    candidates.append(dict(frame=index, slot=result.slot, mode=config.mode, rank=rank,
                                           area=candidate.area, truncated=candidate.truncated,
                                           x=x, y=y, w=w, h=h, theta=theta))
    if wanted:
        raise ValueError(f"Video ended before review frames {sorted(wanted)}")
    assert_sources(manifest["sources"])
    csv_write(out/"detections.csv", rows, list(rows[0]))
    csv_write(out/"candidates.csv", candidates,
              ["frame", "slot", "mode", "rank", "area", "truncated", *FIELDS])
    csv_write(out/"annotations.csv", annotations, list(annotations[0]))
    result = dict(schema_version=1, producer=producer(), audit=str(Path(run).resolve()),
                  source_recording=manifest["sources"], frames=indices,
                  configurations=[asdict(c) for c in configs],
                  decisions={c.mode: dict(Counter(r["reason"] for r in rows if r["mode"] == c.mode)) for c in configs},
                  interpretation="Unvalidated image-supported candidate detections; coverage is not accuracy.")
    json_write(out/"review.json", result)
    return result


def track(run, output, *, config=DetectionConfig(), exclude_slots=()):
    """Full saved-frame output, only after a complete decode audit. No gap filling."""
    config.validate()
    rec, manifest, background, cal = load_run(run)
    output = Path(output).resolve()
    # Companion destination must never resolve to any acquisition source.
    if output in (rec.video, rec.trx_path.resolve(), rec.data_path.resolve()) or output.exists():
        raise ValueError("Companion destination must be new and different from all acquisition sources")
    if output.suffix != ".trx":
        raise ValueError("Companion output must have .trx extension")
    if any(slot not in range(4) for slot in exclude_slots):
        raise ValueError("Excluded slots must be physical IDs 0..3")
    arrays = {key: np.full(rec.shape, np.nan, np.float32) for key in FIELDS}
    status = np.full(rec.shape, 3, np.uint8)
    # Codes are categorical provenance, not manual correctness labels.
    reasons = np.zeros(rec.shape, np.uint8)
    counts = np.zeros(rec.shape, np.uint16)
    ratios = np.full(rec.shape, np.nan, np.float32)
    shape_method = np.zeros(rec.shape, np.uint8)
    shape_codes = {"missing": 0, "box": 1, "weighted_moments": 2, "box_fallback": 3}
    reason_codes = {"accepted": 0, "no_candidate": 1, "ambiguous": 2, "truncated": 3, "excluded": 4}
    start = last_report = time.monotonic()
    decoded = 0
    for index, frame in frames(rec.video):
        if index >= rec.shape[1]:
            raise ValueError("Video grew beyond audited length")
        decoded += 1
        results, _, _ = detect(frame, background, cal, config)
        for result in results:
            slot = result.slot
            counts[slot, index] = min(len(result.candidates), np.iinfo(np.uint16).max)
            if result.runner_up_ratio is not None:
                ratios[slot, index] = result.runner_up_ratio
            reasons[slot, index] = reason_codes["excluded" if slot in exclude_slots else result.reason]
            if result.ellipse is None or slot in exclude_slots:
                continue
            shape_method[slot, index] = shape_codes[result.shape_method]
            (x, y), (w, h), theta = result.ellipse
            for key, value in zip(FIELDS, (x, y, w, h, theta)):
                arrays[key][slot, index] = value
            status[slot, index] = 1 if np.isnan(rec.arrays["x"][slot, index]) else 0
        now = time.monotonic()
        if now-last_report >= 15:
            print(f"Detected {decoded}/{rec.shape[1]} saved frames", flush=True)
            last_report = now
    if decoded != rec.shape[1]:
        raise ValueError(f"Video ended at {decoded}, expected {rec.shape[1]}; no companion written")
    assert_sources(manifest["sources"])
    changed = np.zeros(rec.shape, bool)
    common = np.isfinite(arrays["x"]) & np.isfinite(rec.arrays["x"])
    for key in FIELDS:
        changed |= common & (arrays[key] != rec.arrays[key])
    payload = dict(**arrays, ts=rec.raw["ts"], tracking_metadata=dict(
        schema_version=1, producer=producer(), configuration=dict(detector=asdict(config),
        background=manifest["background_config"], calibration=manifest["calibration"],
        exclude_slots=list(exclude_slots)), source_recording=manifest["sources"],
        background_sha256=manifest["background_sha256"],
        audit=str(Path(run).resolve()), processing_seconds=time.monotonic()-start,
        measurement_convention="saved-image pixels; OpenCV Box2D full diameters and degrees; body estimator in configuration",
        validation="experimental; no measured accuracy claim"), tracking_quality=dict(
        schema_version=1, status=status,
        status_codes={"observed": 0, "recovered": 1, "interpolated": 2, "missing": 3},
        reason=reasons, reason_codes=reason_codes, candidate_count=counts,
        runner_up_area_ratio=ratios, changed_from_original=changed,
        shape_method=shape_method, shape_method_codes=shape_codes,
        uncertainty_calibrated=False, position_uncertainty_px=None,
        shape_uncertainty_px=None, orientation_uncertainty_deg=None))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically, without replacing a concurrently created
    # destination. A failed serialization leaves no apparent companion trajectory.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=".offline-", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()
    return dict(output=str(output), frames=decoded,
                status_counts={name: int(np.count_nonzero(status == code))
                               for name, code in payload["tracking_quality"]["status_codes"].items()},
                processing_seconds=payload["tracking_metadata"]["processing_seconds"])
