"""Rank large-chamber acquisition trajectories from the local experiment list.

Reads sidecars only; never executes commands in the experiment list or imports
reference code. The analysis definition uses fixed pixel thresholds 30 and 10.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
from pathlib import Path
import re
import time

import numpy as np

from nvsl_offline.jumps import score_positions
from nvsl_offline.pipeline import create_run, csv_write, producer
from nvsl_offline.recording import identity, json_write, load_pickle, sidecar


def paths_from_list(path):
    text = Path(path).read_text()
    start = text.index("Large chamber tasks")
    stop = text.index("Summary for 0273 overlap", start)
    section = text[start:stop]
    patterns = [p.strip() for group in re.findall(r'-v\s+"([^\"]+)"', section) for p in group.split(",")]
    paths = sorted({str(Path(name).resolve()) for pattern in patterns
                    for name in glob.glob(pattern+".trx") if ".offline." not in name})
    return paths, dict(start_line=text[:start].count("\n")+1,
                       stop_line_exclusive=text[:stop].count("\n")+1,
                       patterns=len(patterns), unique_trajectories=len(paths))


def examine(path, lost_limit):
    path = Path(path)
    video = path.with_suffix("") if path.stem.endswith(".avi") else path.with_suffix(".avi")
    if not video.is_file():
        raise ValueError(f"Missing video for {path}")
    metadata_path = sidecar(video, ".data")
    before = {"trx": identity(path), "data": identity(metadata_path)}
    metadata, tracks = load_pickle(metadata_path), load_pickle(path)
    protocol = metadata.get("protocol", {})
    if protocol.get("ct") != "large" or len(tracks["x"]) != 4:
        raise ValueError("Not a four-slot large-chamber recording")
    rows, pairs = [], []
    for slot, x in enumerate(tracks["x"]):
        summary, events = score_positions(x, tracks["y"][slot], lost_limit=lost_limit,
                                         open_loop=protocol.get("pt") == "openLoop")
        row = dict(video=str(video), trx=str(path), data=str(metadata_path), slot=slot, **summary)
        rows.append(row)
        pairs.extend(dict(video=str(video), slot=slot, **event) for event in events)
    after = {"trx": identity(path), "data": identity(metadata_path)}
    if before != after:
        raise ValueError("Source sidecar changed while scanning")
    return rows, pairs, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lost-limit", type=float, default=.6, help="Current large2 analysis loss threshold")
    parser.add_argument("--limit", type=int, help="Optional first-N limit for an explicitly partial screen")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 <= args.lost_limit <= 1 or (args.limit is not None and args.limit < 1):
        parser.error("Invalid workers, loss threshold, or limit")
    paths, selection = paths_from_list(args.list)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No large-chamber trajectory files matched")
    out = create_run(args.output)
    rows, pairs, errors, sources = [], [], [], []
    start = reported = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(examine, path, args.lost_limit): path for path in paths}
        for i, future in enumerate(as_completed(jobs), 1):
            try:
                found, events, source = future.result()
                rows.extend(found)
                pairs.extend(events)
                sources.append(source)
            except Exception as exc:
                errors.append(dict(trx=jobs[future], error=str(exc)))
            if time.monotonic()-reported >= 15:
                print(f"Screened {i}/{len(paths)} trajectories; {len(errors)} errors", flush=True)
                reported = time.monotonic()
    rows.sort(key=lambda r: (-(r["analysis_pairs_per_100k_frames"] or -1),
                             -r["raw_suspicious_pairs"], r["video"], r["slot"]))
    if rows:
        csv_write(out/"rankings.csv", rows, list(rows[0]))
    pairs.sort(key=lambda r: (r["video"], r["slot"], r["mode"], r["first_jump"]))
    csv_write(out/"pairs.csv", pairs, ["video", "slot", "first_jump", "second_jump", "first_length_px",
              "second_length_px", "reverse_endpoint_mismatch_px", "dwell_frames", "mode",
              "synthetic_endpoints", "missing_frames_between_jumps"])
    json_write(out/"manifest.json", dict(producer=producer(), list_source=identity(Path(args.list), content_hash=True),
        selection=selection, attempted=len(paths), successful=len(sources), errors=errors, sources=sources,
        seconds=time.monotonic()-start, lost_limit=args.lost_limit,
        rule="Consecutive >30 px jumps with summed reverse endpoint mismatch <10 px; bad if >=3 pairs and pairs/long_jumps >=.03.",
        ranking="Analysis-interpolated pairs per 100000 post-start saved frames; raw observed-edge counts also retained.",
        interpretation="Candidate selection only. A suspicious jump is not a verified false detection."))
    print(json.dumps(dict(successful=len(sources), errors=len(errors), fly_slots=len(rows), top=rows[:12]), indent=2))


if __name__ == "__main__":
    main()
