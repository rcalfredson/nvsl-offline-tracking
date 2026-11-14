**Wrong-object tracking tests, 13 Nov 2026.** The detector, background settings,
acceptance rules and body estimator were held fixed. New work adds screening
and evaluation utilities, not tracker tuning. Results show that some acquisition
false detections are conservatively rejected, while a clear offline-specific
wrong-object failure remains.

**What “suspicious jumps” means.** `analyze.py` imports `Trajectory`; the rule is
implemented in [`Trajectory._suspiciousJumps()`](../reference/nvsl-analysis/src/analysis/trajectory.py#L395).
It finds displacements greater than 30 saved-image pixels and examines consecutive
members of that long-jump list. If the first goes approximately A→B and the next
approximately B→A, it counts a suspicious pair when the sum of reverse-endpoint
distances is less than 10 pixels. Return need not occur in the next video frame.
Pairs can overlap, so their count is not the number of independent false-lock
episodes. The jump-based bad flag requires at least three suspicious pairs and
a suspicious-pairs / long-jumps fraction of at least 0.03.

The production analysis checks this after its missing-position interpolation,
not directly on the raw measurements. The new independent scorer reproduces
leading-NaN handling, interior interpolation, terminal holding, the >99%-missing
empty check and the current large2 post-start loss gate (0.6, bypassed for
open-loop). It also retains counts restricted to raw observed edges. Diagnostic
interpolation operates on copies and never changes a source or companion file.
Tests compare both pairing and post-interpolation counts against the actual
reference function bodies, without importing or modifying the reference package.

The rule is a search aid: real rapid movement, return excursions, or segmentation
changes can trigger it. Persistent wrong locks, gradual drift, and errors below
the 30-pixel cutoff can escape it. A smaller count is not by itself evidence of
better localization, especially if the tracker leaves more frames missing.

**Recording screen.** Safely expanded the quoted video patterns from the large-
chamber section of `full_video_list.txt` (lines 364–421); did not execute any
commands in that file. All 700 unique matched four-slot recording sidecars were
read successfully: 2800 physical fly slots, of which 2699 passed the diagnostic
eligibility gates, 73 were >99% missing, and 28 exceeded the post-start loss gate.
Fifty-three slots had an analysis-style suspicious pair and 28 had a raw pair.
Those counts are not a false-positive prevalence estimate. Original suspicious
rates were ranked per 100000 post-start saved frames, with raw counts and observed-
edge exposure also recorded. See [jump_screen_summary.json](jump_screen_summary.json).

**Three new complete runs and an earlier baseline check.** Selected the first three ranked fly/recording
candidates before inspecting their offline results. These correspond to two
July 25 recordings and an August 1 second-round recording. All passed full
sequential decode/frame/trajectory alignment checks. Generated normalized,
weighted-moment companions with the unchanged defaults: 129 background samples,
80th-percentile background, temporal MAD noise estimate, threshold score >4,
signal floor >6, area 40–1000 px², runner-up area ratio cutoff 0.5, 24-pixel
sidewall margin and the saved large2 transform. Acquisition files and recorded
events were preserved. New companions cover every saved frame, with NaNs for
unresolved measurements.

| Recording / physical slot | Saved frames | Original → offline suspicious pairs | Original → offline missing measurements |
|---|---:|---:|---:|
| July 25 `c51__12-02-51`, slot 0 | 68198 | 11 → 0 | 9.527% → 0.859% |
| July 25 `c52__12-02-53`, slot 2 | 68196 | 3 → 0 | 13.938% → 2.871% |
| August 1, second round `c52__14-57-52`, slot 1 | 101581 | 3 → 1 | 6.381% → 0.200% |
| May 30 `c51__19-22-58`, slot 1 — earlier baseline, rim-lock regression | 108293 | 0 → 5 | 2.135% → 1.174% |

For these rows the raw and analysis-interpolated pair counts coincide. Across all
12 slots in the three new runs, pairs fall from 19 to 1; none of the other slots
acquires a new pair. That aggregate excludes the retrospective May 30 baseline
check, whose slot-1 regression adds five offline pairs. The May companion was
already generated during baseline development; it is not a fourth new run.
These are diagnostic counts for a deliberately enriched
sample, not annotated precision results. Full per-slot metrics, configurations
and source identities are in [false_positive_comparison.json](false_positive_comparison.json).

**Full-set aggregate, including the May regression.** Pooling every physical slot
in that comparison file covers four recordings, 16 fly slots and 1385072 saved
fly-frame opportunities per method. These totals include all slots, not just the
highlighted slots in the preceding table. Missingness is weighted by frame count:
sum missing measurements divided by sum saved fly-frames, rather than an average
of per-slot percentages. Every slot is eligible for the analysis-style count in
this comparison set.

| Full comparison set | Original | Offline |
|---|---:|---:|
| Raw suspicious jump pairs | 19 | 6 |
| Analysis-style suspicious jump pairs | 19 | 6 |
| Physical slots with at least one raw suspicious pair | 4 / 16 | 2 / 16 |
| Missing fly-frame measurements | 180855 | 14979 |
| Frame-weighted missing fraction | 13.057% | 1.081% |

Raw pair counts decrease for four slots, increase for one (May 30 slot 1), and
remain unchanged for 11. The net pair-count reduction is 68.4%, including the
regression. There are 165876 additional finite measurements overall, but their
correctness is not established by availability. These are neither a measured
68.4% reduction in false positives nor a representative population estimate:
recordings were selected using the jump screen and supplemented with a known
baseline failure. Rejection as missing can remove suspicious pairs without
recovering the fly. The machine-readable report now includes this full-set
`aggregate` as well as its per-slot `comparisons`.

**What the images establish.** July 25 c51 slot 0 repeatedly switches between
the visible fly near the upper chamber edge and a feature on the upper well rim,
about 43–46 pixels away, around frames 40648–40700. Another wrong-object example
is visible at frame 43576. At frame 40649 the offline candidates have areas 151
and 150.5 pixels²; the detector declines to choose between the fly and artifact.
The July 25 c52 slot-2 examples at 30114 and 30189 likewise show the original
ellipse on a feature away from the visible fly. Offline detections at these
reviewed wrong-object frames are rejected as ambiguous. Some neighboring frames
are rejected too, including ones where the original measurement was correct.

“Offline ambiguous” means a missing **fly measurement**, not a missing video
frame: x/y/w/h/theta are NaN and the saved timestamp/index remain present. It is
different from “no candidate”: plausible competing components exist, but the
second-largest is at least half the largest's area. These specific examples
demonstrate rejection of false choices, not successful recovery of their true
positions. A lower aggregate missing rate elsewhere does not change that fact.

The August 1 candidate is a useful caution/control. In the reviewed
37643–37974 interval the original ellipses appear consistent with a fly moving
through the chamber; the 330-frame separation between two long jumps allows a
return excursion to be flagged. The one remaining offline pair at 1485/1486
occurs at the same indices as an original pair. Its presence is not sufficient
to label either estimate a false detection. These visual assessments are limited
assistant spot checks, not a completed manual benchmark.

**A clear remaining offline failure.** Also screened the previously generated
May 30 c51 normalized companion. Original slot 1 has zero suspicious pairs;
offline has five, at first-jump indices 49, 63, 195, 211 and 223. Two precede
the original tracker’s first observation at frame 152; three occur afterward.
All are in the first 30 seconds of the recording, before training. At frames
211–213 the original ellipse follows the visible fly near the upper well while
the offline ellipse follows a long, thin feature on the right chamber rim.
At frame 212 their centers are approximately (511,50) and (674,234), respectively.
This is an inspected offline-specific wrong-object lock, despite improved overall
availability (slot-1 missingness 2.135%→1.174%). The five-pair heuristic count
does not measure how many frames that wrong lock lasts.

The May position sequence is identical to the earlier normalized box-geometry
companion, as verified in the geometry pilot; changing to weighted moments did
not introduce this position error. A future gap-only substitution policy would
protect original finite measurements such as frames 211–213, but could still
insert false measurements into original gaps such as frames 49–51. It is not
a substitute for validating recovered positions.

This susceptibility is expected from the unchanged code: the largest surviving
component can win if its competitors are absent, filtered out, or less than half
its area. There is no appearance model or temporal association enforcing that
the chosen object is the fly. A moment-based body estimate changes geometry after
selection and cannot repair selection of the wrong object. The current contour
area limits do not reject every implausibly long, thin component. These are
failure mechanisms suggested and, in the rim example, supported by actual images;
no corrective filter was introduced in this test.

**Review artifacts and reproducible commands.** Local run artifacts are not
checked in. The main examples are:

- [July 25 c51 snapshot, frame 40649](../runs/jump-screen-large/jul25_c51_frame40649.png)
  and [switching clip](../runs/jump-screen-large/jul25_c51_switching.mp4): original
  switches targets; offline abstains.
- [July 25 c52 switching clip](../runs/jump-screen-large/jul25_c52_switching.mp4):
  another wrong-object acquisition example.
- [May 30 frame 212](../runs/jump-screen-large/may30_slot1_frame212.png) and
  [rim-lock clip](../runs/jump-screen-large/may30_slot1_rim_lock.mp4): a clear
  remaining offline failure.
- [August 1 retained-pair clip](../runs/jump-screen-large/aug01_retained_pair.mp4):
  review the remaining diagnostic flag without assuming it is wrong tracking.

```bash
PYTHONPATH=src python scripts/scan_suspicious_jumps.py full_video_list.txt \
  --output runs/jump-screen-new --workers 4

PYTHONPATH=src python scripts/compare_jump_trajectories.py \
  runs/fp-jul25-c51/jul25-c51.normalized.offline.trx \
  runs/fp-jul25-c52/jul25-c52.normalized.offline.trx \
  runs/fp-aug01-c52/aug01-c52.normalized.offline.trx \
  --output runs/jump-comparison-new

PYTHONPATH=src python -m nvsl_offline compare \
  runs/fp-jul25-c51/jul25-c51.normalized.offline.trx --fly 0 --start 40635
```

Scanner outputs are `rankings.csv`, `pairs.csv` and a manifest with explicit
selection, source file identities and any read errors. Comparator outputs include
both sources' per-slot exposure/missingness, raw and analysis-style counts,
individual pair indices and offline missingness at the pair endpoints. The
complete new outputs are in `runs/fp-jul25-c51/`, `runs/fp-jul25-c52/` and
`runs/fp-aug01-c52/`; each has its own audit and independent companion.

**Next validation.** Retain well-rim rejection, offline rim-lock failure and
plausible true-motion controls in a small review set, together with ordinary
finite frames and non-returning wrong locks. Label localization and target
identity, not just whether an ellipse exists. Measure how often false choices
are replaced by correct measurements versus missing measurements, and whether
new false choices appear. Keep the algorithm stable during that review rather
than tuning to the few examples just discovered. Forty-six tests pass, including
reference parity for pairing and interpolation-aware screening, frame-weighted
aggregation and retention of individual regressions in the pooled report.
