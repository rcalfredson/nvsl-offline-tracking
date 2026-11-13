**First implementation, 2026-11-13.** The repository now has an independent
experimental classical baseline. It reads trusted acquisition pickles through
a state-only compatibility placeholder and uses saved geometry; no reference
code or template images are imported at runtime. See the [README](../README.md)
for commands, configuration, provenance and limitations.

The initial implementation includes a full sequential decode audit, tiled
recording-wide background/noise estimation, exact-frame diagnostics, two detector
modes, blank manual-annotation worksheets and full-length companion output.
It never interpolates or predicts positions. All output destinations must be new;
source identities are rechecked, and companion publication is atomic and refuses
replacement.

**Real recording checks.** Both four-hour May recordings completed sequential
decode checks, not just header checks:

| Recording | Header, decoded and trajectory frames | Current audit directory |
|---|---:|---|
| `c51__2025-05-30__19-22-58` | 108293 | `runs/c51-baseline` |
| `c52__2025-05-30__19-23-03` | 108292 | `runs/c52-baseline` |

All five geometry fields have matching shapes and missing masks. Source timestamps
are retained without resampling, and recorded event indices are checked. The
canonical audits used 129 uniformly spaced grayscale samples, the 80th-percentile
background, smoothed temporal MAD with a 3-level floor, the saved LgC2 transform,
and a 24-pixel sidewall search margin. Audit manifests contain source sidecar
SHA-256 hashes, video path/size/mtime, configuration and background checksums.
The final audits took about 80 and 83 seconds while running concurrently on this
host, including sequential decoding; this is an exploratory runtime observation,
not a controlled hardware benchmark. Earlier 12-pixel-margin audits/reviews are
development artifacts, not the canonical runs.

**What the first images show.** Review used seven c51 saved frames (4827, 4900,
5500, 6200, 6967, 6968, 10000) and three c52 saved frames (1000, 5000, 10000),
with all physical slots included. These are development examples, not a random
or held-out accuracy sample. Raw images, overlays, residuals, masks and candidate
tables are in `runs/c51-review-baseline` and `runs/c52-review-baseline`.

- The first 12-pixel margin clipped the fly candidate at the dark upper-left
  edge. Rather than accept a clipped shape, the inspected margin was increased
  to 24 pixels and truncation rejection retained. It remains configurable and
  needs review for other layouts/recordings.
- At c51 slot 0, frames 4900, 5500 and 6200 still have no accepted contour at
  absolute threshold 25, but produce accepted candidates in normalized mode.
  Frame 6967, also originally missing, produces a candidate in both modes.
- Across these seven c51 frames and four slots, absolute-25 accepted 25/28
  opportunities and normalized accepted 28/28. Across three c52 frames and four
  slots, absolute-25 accepted 11/12 and normalized accepted 10/12, rejecting two
  ambiguous cases. These are **candidate acceptance counts**, not correct
  detection rates or recovery accuracy.
- Normalized body estimates are larger in several inspected ordinary frames.
  Improved center availability does not establish better body dimensions or
  orientation. These changes need annotated body-outline review.

The [dark-edge preview](baseline_dark_edge_preview.png) shows frame 5500, physical
slot 0, with source pixels enlarged threefold. Green marks are accepted
rectangle-derived ellipses, not manual labels. The absolute-12 control also
detects this body. In fact it accepts all 28 opportunities in the c51 review.
This is expected where temporal noise reaches its floor: normalized threshold
4 times noise floor 3 gives an effective raw threshold of 12. Therefore these
examples support testing lower-threshold detection with a recording-wide
background, but **do not yet show an advantage of noise normalization over an
appropriately lowered absolute threshold**. The separate review is in
`runs/c51-review-threshold12`.

**Validation and next evidence.** Nineteen automated tests passed, including
synthetic faint-body recovery, stationary-body detection against a clean
background, competing candidates, crop truncation, ellipse-axis equivalence,
legacy metadata loading, timestamp preservation, original-file preservation,
source/background mutation rejection, frame-count mismatch and failed-output
publication. The package also built as a wheel and its CLI ran from an isolated
temporary installation outside the repository.

**Full c51 companion runs.** Both modes completed all 108293 saved frames and
produced separate files under `runs/c51-baseline/`. Each has four physical slots,
all six legacy fields, unchanged source timestamps, consistent geometry missing
masks, and status/rejection metadata. Source identities were rechecked after
processing. No interpolation status was emitted. Detailed descriptive results
are in [baseline_run_summary.json](baseline_run_summary.json).

| Method | Missing fly-frames / 433172 | Missing fraction | New candidates in originally missing fly-frames | Candidates in slot-0 gap [4828,6968) |
|---|---:|---:|---:|---:|
| Original acquisition | 42601 | 9.83% | — | 0 / 2140 |
| Recording-wide background, absolute 25 | 23160 | 5.35% | 19477 | 26 / 2140 |
| Same background, normalized | 3687 | 0.85% | 41931 | 2140 / 2140 |

These totals include startup and all slots, with no occupancy exclusions. They
measure availability, not positional correctness. Normalized processing also
rejects 3017 formerly finite fly-frames, compared with 36 for absolute-25.
For slot 1, normalized processing creates a longest missing run of 611 frames,
versus 283 in acquisition and 187 with absolute-25. Thus aggregate coverage
improvement can conceal worse gaps for an individual fly.

There is also a substantial shape difference: on common finite frames, median
ellipse-area ratios relative to acquisition are 1.16–1.29 for absolute-25 and
1.79–2.36 for normalized detection across the four slots. Corresponding per-slot
95th-percentile center disagreements are 1.02–1.34 pixels and 1.76–4.07 pixels.
Acquisition is not ground truth, so these are disagreements rather than error
estimates. They make body-outline and ordinary-frame annotation necessary before
using these trajectories for wall-contact or other scientific analyses.

Detector passes took about 305 and 354 seconds respectively while running
concurrently on this host, excluding the separate initial audit/background pass.
They used CPU processing. The threshold-12 control has only been reviewed on the
small development sample; its full-recording precision/coverage comparison remains
future evaluation, not an established normalization benefit.

The next scientific step is to label diverse ordinary and difficult frames using
the prediction-free `_frame.png` images and blank `annotations.csv` sheets,
including finite-but-wrong acquisition detections. Compare center precision,
coverage, false detections, and body-axis/diameter accuracy across original,
absolute-25, absolute-12 and normalized results, with fresh recordings held out
before tuning. Longer ambiguous sequences need temporal review. No annotation
accuracy gate has been passed, no learned model is justified by these results,
and no acquisition events or reference files have been modified.
