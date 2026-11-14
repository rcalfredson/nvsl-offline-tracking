"""Compare completed companions with acquisition using diagnostic jump counts.

Counts are screening proxies, not false-positive labels. Existing trajectories
are read without modification; interpolation is confined to the diagnostic copy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from nvsl_offline.jumps import aggregate_comparisons, score_positions
from nvsl_offline.pipeline import create_run, csv_write, producer
from nvsl_offline.recording import identity, json_write
from nvsl_offline.replay import Comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("companions", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lost-limit", type=float, default=.6)
    args = parser.parse_args()
    if not 0 <= args.lost_limit <= 1:
        parser.error("Lost-frame fraction must be in [0,1]")
    out = create_run(args.output)
    rows, all_pairs, sources = [], [], []
    for path in args.companions:
        c = Comparison.load(path)
        sources.append(dict(companion=identity(Path(path), content_hash=True),
                            configuration=c.variant.raw.get("tracking_metadata", {}).get("configuration"),
                            acquisition=c.variant.raw.get("tracking_metadata", {}).get("source_recording")))
        for slot in range(c.original.shape[0]):
            first_original = np.flatnonzero(np.isfinite(c.original.arrays["x"][slot]))
            first_original = int(first_original[0]) if len(first_original) else c.nframes
            for source, rec in [("original", c.original), ("offline", c.variant)]:
                summary, pairs = score_positions(rec.arrays["x"][slot], rec.arrays["y"][slot],
                    lost_limit=args.lost_limit, open_loop=rec.protocol.get("pt") == "openLoop")
                rows.append(dict(video=str(c.original.video), companion=str(c.variant.trx_path), slot=slot,
                    source=source, nominal_fps=c.fps, **summary,
                    raw_pairs_after_original_start=sum(p["mode"] == "raw" and p["first_jump"] >= first_original for p in pairs)))
                for pair in pairs:
                    a, b = pair["first_jump"], pair["second_jump"]
                    endpoints = [a, a+1, b, b+1]
                    offline_missing = np.isnan(c.variant.arrays["x"][slot, endpoints])
                    all_pairs.append(dict(video=str(c.original.video), companion=str(c.variant.trx_path),
                        slot=slot, source=source, **pair,
                        first_jump_video_seconds=a/c.fps, second_jump_video_seconds=b/c.fps,
                        offline_missing_endpoints=int(offline_missing.sum())))
    aggregate = aggregate_comparisons(rows)
    csv_write(out/"comparison.csv", rows, list(rows[0]))
    csv_write(out/"pairs.csv", all_pairs, ["video", "companion", "slot", "source", "first_jump", "second_jump",
        "first_length_px", "second_length_px", "reverse_endpoint_mismatch_px", "dwell_frames", "mode",
        "synthetic_endpoints", "missing_frames_between_jumps", "first_jump_video_seconds", "second_jump_video_seconds",
        "offline_missing_endpoints"])
    report = dict(producer=producer(), sources=sources, lost_limit=args.lost_limit,
                  interpretation="Unchanged tracker. Candidate jump counts and availability, not annotated false-positive rates.",
                  comparisons=rows, aggregate=aggregate)
    json_write(out/"summary.json", report)
    for row in rows:
        print(f"{Path(row['video']).name} slot {row['slot']} {row['source']}: "
              f"raw={row['raw_suspicious_pairs']} analysis={row['analysis_suspicious_pairs']} "
              f"missing={row['missing_fraction']:.3%}")
    print("Full-set aggregate:")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
