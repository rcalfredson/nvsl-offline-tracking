"""Compare optional lighting correction on explicit saved-frame intervals.

Plan: {"recordings": [{"companion": path, "intervals": [[start, stop], ...],
"slots": [0, ...], "snapshots": [frame, ...]}]}. Stops are exclusive.
Writes review measurements, not truncated .trx files or accuracy labels.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.jumps import jump_pairs
from nvsl_offline.pipeline import assert_sources, create_run, csv_write, load_run, producer, write_image
from nvsl_offline.recording import FIELDS, identity, json_write, load_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = create_run(args.output)
    report = dict(producer=producer(), plan=plan, recordings=[],
                  interpretation="Selected-interval measurements and diagnostic raw pair counts; no manual accuracy labels. Pair counts exclude interval boundaries.")
    start = time.monotonic()
    for number, item in enumerate(plan['recordings']):
        payload = load_pickle(Path(item['companion']))
        metadata = payload['tracking_metadata']
        rec, manifest, bg, cal = load_run(metadata['audit'])
        baseline = DetectionConfig(**metadata['configuration']['detector'])
        if baseline.illumination != 'none':
            raise ValueError('Plan requires an uncorrected baseline companion')
        corrected = replace(baseline, illumination='global_offset')
        directory = out / f"{number+1:02d}-{rec.video.stem}"
        directory.mkdir()
        cap = cv2.VideoCapture(str(rec.video))
        rows, summaries = [], []
        baseline_max_error = 0.
        try:
            for first, stop in item['intervals']:
                if not 0 <= first < stop <= rec.shape[1]:
                    raise ValueError('Invalid interval')
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != first:
                    raise ValueError('Could not seek to saved-frame index')
                arrays = [np.full((4, stop-first, 5), np.nan) for _ in range(2)]
                offsets, spreads = [], []
                for index in range(first, stop):
                    ok, image = cap.read()
                    if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index+1:
                        raise ValueError('Incomplete or misaligned interval decode')
                    results = []
                    for variant, config in enumerate([baseline, corrected]):
                        found, _, _ = detect(image, bg, cal, config)
                        results.append(found)
                        for result in found:
                            slot = result.slot
                            if result.ellipse is not None:
                                (x, y), (w, h), theta = result.ellipse
                                arrays[variant][slot, index-first] = x, y, w, h, theta
                            measurement = arrays[variant][slot, index-first]
                            if variant == 0:
                                saved = np.array([payload[k][slot, index] for k in FIELDS])
                                np.testing.assert_allclose(measurement, saved, atol=1e-3, rtol=1e-5, equal_nan=True)
                                if np.isfinite(saved).all():
                                    baseline_max_error = max(baseline_max_error, float(np.max(np.abs(measurement-saved))))
                            rows.append(dict(frame=index, slot=slot, variant=config.illumination,
                                reason=result.reason, candidate_count=len(result.candidates),
                                illumination_offset=result.illumination_offset,
                                illumination_tile_mad=result.illumination_tile_mad,
                                **dict(zip(FIELDS, measurement))))
                    offsets.append(results[1][0].illumination_offset)
                    spreads.append(results[1][0].illumination_tile_mad)
                    if index in item.get('snapshots', []):
                        for slot in item['slots']:
                            x0, y0, x1, y1 = cal.roi(slot)
                            panels = []
                            original_values = [rec.arrays[k][slot, index] for k in FIELDS]
                            original_ellipse = ((original_values[0], original_values[1]),
                                                (original_values[2], original_values[3]), original_values[4]) if np.isfinite(original_values).all() else None
                            for ellipse, title, color in [
                                (original_ellipse, 'original', (0, 220, 255)),
                                (results[0][slot].ellipse, 'baseline: '+results[0][slot].reason, (80, 255, 80)),
                                (results[1][slot].ellipse, 'corrected: '+results[1][slot].reason, (255, 100, 255))]:
                                panel = image.copy()
                                if ellipse is not None:
                                    cv2.ellipse(panel, ellipse, color, 1)
                                panel = cv2.copyMakeBorder(panel[y0:y1, x0:x1], 45, 0, 0, 0, cv2.BORDER_CONSTANT)
                                cv2.putText(panel, title, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
                                cv2.putText(panel, f"f{index} s{slot} offset {offsets[-1]:+.1f}", (3, 34), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                                panels.append(panel)
                            write_image(directory/f'frame{index:07d}-slot{slot}.png', np.hstack(panels))
                for slot in range(4):
                    summary = dict(start=first, stop=stop, slot=slot,
                                   offset_median=float(np.median(offsets)),
                                   offset_min=float(min(offsets)), offset_max=float(max(offsets)),
                                   tile_mad_median=float(np.median(spreads)))
                    for variant, label in enumerate(['baseline','corrected']):
                        a = arrays[variant][slot]
                        _, pairs = jump_pairs(a[:,0], a[:,1])
                        summary[label] = dict(missing_frames=int(np.isnan(a[:,0]).sum()),
                            raw_pairs=[dict(p, first_jump=p['first_jump']+first, second_jump=p['second_jump']+first) for p in pairs])
                    summaries.append(summary)
                np.savez_compressed(directory/f'frames-{first}-{stop}.npz',
                                    frames=np.arange(first,stop), baseline=arrays[0], corrected=arrays[1],
                                    illumination_offset=offsets, illumination_tile_mad=spreads)
                print(f"{rec.video.name}: {first}:{stop} offset {np.median(offsets):+.2f}", flush=True)
        finally:
            cap.release()
        assert_sources(manifest['sources'])
        csv_write(directory/'measurements.csv', rows, list(rows[0]))
        report['recordings'].append(dict(video=str(rec.video), baseline=identity(Path(item['companion']),content_hash=True),
            baseline_config=asdict(baseline), corrected_config=asdict(corrected),
            baseline_reproduction_max_error=baseline_max_error, intervals=summaries))
    report['seconds'] = time.monotonic()-start
    json_write(out/'summary.json', report)
    print(f"Complete: {out}", flush=True)


if __name__ == '__main__':
    main()
