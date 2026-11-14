"""Diagnostic reproduction of nvsl-analysis jump rules; never edits trajectories.

Reference: Trajectory._interpolate, _calcDistances and _suspiciousJumps.
Interpolation here is exclusively a diagnostic view for reproducing analysis;
no interpolated samples are written to a tracking companion.
"""

from __future__ import annotations

import numpy as np


def jump_pairs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Expected matching one-dimensional position sequences")
    lengths = np.hypot(np.diff(x), np.diff(y))
    jumps = np.flatnonzero(lengths > 30)
    if len(jumps) < 2:
        return jumps, []
    first, second = jumps[:-1], jumps[1:]
    mismatch = (np.hypot(x[first+1]-x[second], y[first+1]-y[second])
                + np.hypot(x[first]-x[second+1], y[first]-y[second+1]))
    pairs = [dict(first_jump=int(a), second_jump=int(b),
                  first_length_px=float(lengths[a]), second_length_px=float(lengths[b]),
                  reverse_endpoint_mismatch_px=float(score), dwell_frames=int(b-a))
             for a, b, score in zip(first, second, mismatch) if score < 10]
    return jumps, pairs


def score_positions(x, y, *, lost_limit=.6, open_loop=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("Expected matching nonempty position sequences")
    if (np.isinf(x).any() or np.isinf(y).any()
            or not np.array_equal(np.isnan(x), np.isnan(y))):
        raise ValueError("Inconsistent x/y missingness or infinite coordinates")
    missing = np.isnan(x)
    valid = ~missing
    first = int(np.flatnonzero(valid)[0]) if valid.any() else len(x)
    n = len(x)
    raw_jumps, raw_pairs = jump_pairs(x, y)
    lost = float(missing[first:].mean()) if first < n else 1.
    state = "eligible"
    if int(missing.sum()) > .99*n:
        state = "empty"
    elif lost > lost_limit and not open_loop:
        state = "excess_loss"
    analysis_pairs, analysis_jumps = [], np.empty(0, int)
    if state == "eligible":
        # np.interp matches interior linear interpolation and terminal holding.
        # Retain leading NaNs, as the analysis does after removing its first run.
        indices = np.arange(n)
        xx, yy = x.copy(), y.copy()
        later_missing = missing & (indices >= first)
        for source, target in [(x, xx), (y, yy)]:
            target[later_missing] = np.interp(indices[later_missing], indices[valid], source[valid])
        analysis_jumps, analysis_pairs = jump_pairs(xx, yy)
    for mode, pairs in [("raw", raw_pairs), ("analysis", analysis_pairs)]:
        for pair in pairs:
            a, b = pair["first_jump"], pair["second_jump"]
            endpoints = [a, a+1, b, b+1]
            pair["mode"] = mode
            pair["synthetic_endpoints"] = bool(missing[endpoints].any())
            pair["missing_frames_between_jumps"] = int(missing[a:b+2].sum())
    fraction = len(analysis_pairs)/len(analysis_jumps) if len(analysis_jumps) else 0.
    summary = dict(frames=n, first_observed_frame=first, observed_frames=int(valid.sum()),
                   observed_adjacent_pairs=int(np.count_nonzero(valid[:-1] & valid[1:])),
                   missing_fraction=float(missing.mean()), loss_after_start=lost,
                   analysis_state=state, analysis_long_jumps=len(analysis_jumps),
                   analysis_suspicious_pairs=len(analysis_pairs) if state == "eligible" else None,
                   analysis_suspicious_fraction=fraction if state == "eligible" else None,
                   analysis_pairs_per_100k_frames=100000*len(analysis_pairs)/(n-first) if state == "eligible" else None,
                   analysis_would_mark_bad=bool(len(analysis_pairs) >= 3 and fraction >= .03) if state == "eligible" else None,
                   raw_long_jumps=len(raw_jumps), raw_suspicious_pairs=len(raw_pairs),
                   raw_pairs_per_100k_observed_edges=100000*len(raw_pairs)/max(1, int(np.count_nonzero(valid[:-1] & valid[1:]))))
    return summary, analysis_pairs+raw_pairs


def aggregate_comparisons(rows):
    """Pool paired slot results using frame counts, retaining regressions.

    One companion per recording is required to avoid counting the acquisition
    exposure repeatedly when different variants of one recording are supplied.
    Analysis-gated slots are counted as unassessed, not zero suspicious pairs.
    """
    paired = {}
    for row in rows:
        if row["source"] not in ("original", "offline"):
            raise ValueError("Expected original/offline comparison rows")
        key = (row["video"], row["slot"])
        group = paired.setdefault(key, {})
        if row["source"] in group:
            raise ValueError("Aggregate requires one companion per recording/slot")
        group[row["source"]] = row
    if not paired:
        raise ValueError("No comparison rows to aggregate")
    for group in paired.values():
        if set(group) != {"original", "offline"}:
            raise ValueError("Every slot must have both original and offline results")
        if group["original"]["frames"] != group["offline"]["frames"]:
            raise ValueError("Paired slots must cover the same saved frames")
        for row in group.values():
            if not 0 <= row["observed_frames"] <= row["frames"] or row["frames"] <= 0:
                raise ValueError("Invalid measurement exposure")
    result = dict(schema_version=1, recordings=len({key[0] for key in paired}), fly_slots=len(paired),
                  weighting="Missingness = summed missing fly-frames / summed saved fly-frames; no averaging of percentages.",
                  interpretation="Diagnostic jump counts and availability, not an annotated false-positive rate.")
    for source in ("original", "offline"):
        selected = [group[source] for group in paired.values()]
        frames = sum(row["frames"] for row in selected)
        observed = sum(row["observed_frames"] for row in selected)
        raw_pairs = sum(row["raw_suspicious_pairs"] for row in selected)
        assessed = [row for row in selected if row["analysis_suspicious_pairs"] is not None]
        result[source] = dict(saved_fly_frames=frames, observed_fly_frames=observed,
            missing_fly_frames=frames-observed, missing_fraction=(frames-observed)/frames,
            raw_suspicious_pairs=raw_pairs, raw_pairs_per_100k_saved_fly_frames=100000*raw_pairs/frames,
            slots_with_raw_suspicious_pairs=sum(row["raw_suspicious_pairs"] > 0 for row in selected),
            analysis_assessed_slots=len(assessed), analysis_unassessed_slots=len(selected)-len(assessed),
            analysis_suspicious_pairs=sum(row["analysis_suspicious_pairs"] for row in assessed) if assessed else None)
    changes = [group["offline"]["raw_suspicious_pairs"]-group["original"]["raw_suspicious_pairs"]
               for group in paired.values()]
    result["slot_outcomes_by_raw_pair_count"] = dict(decreased=sum(d < 0 for d in changes),
        increased=sum(d > 0 for d in changes), unchanged=sum(d == 0 for d in changes))
    return result
