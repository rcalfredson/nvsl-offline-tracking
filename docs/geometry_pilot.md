**Optional body-geometry pilot, 2026-11-13.** The current detection baseline
measures each frame independently; it does not optimize trajectory continuity.
Foreground thresholding determines which pixels enter a candidate. Including
weak peripheral pixels can improve center availability while also enlarging or
destabilizing an enclosing rectangle. Recovering additional frames is therefore
separate from estimating a good body ellipse.

The low-cost trial is **squared-darkness-weighted moments**, after detection has
already accepted a component. Pixels in that foreground component have weight
`max(background - image, 0)²`. The weighted second-moment matrix is calculated
about the unchanged rectangle center. Eigenvectors give the principal body axis;
full diameters are `4 * sqrt(eigenvalue)`, matching the moments of a uniform filled
ellipse. This is an intensity-weighted shape estimate, not a contour least-squares
fit or a guarantee that the fly's anatomical outline has been identified.

Only w/h/theta change. Foreground masks, candidate contours/ranking, acceptance
rules, x/y, original timestamps, physical slot ordering and missingness are kept
fixed. If moments are numerically unresolved, the existing rectangle is retained.
The default remains `box`; use `--shape-fit weighted_moments` explicitly. Per-frame
`tracking_quality.shape_method` codes distinguish missing, box, weighted moments,
and box fallback. No temporal information or smoothing is used in this estimator.

Both estimators are converted to equivalent minor/major axes and axial angles
modulo 180 degrees for comparison. Frame-to-frame angle steps use the shortest
axial difference; a width/height swap plus 90-degree angle change is equivalent
geometry, not a genuine rotation. Nearly circular bodies still have poorly
determined orientation, and neither method measures head/tail direction.

**Development-window evidence.** Four c51 windows, covering 520 saved frames and
all four physical slots, were compared with the same normalized foreground rule:
[4800,5000), [5480,5540), [6940,7000), and [10000,10200). These reuse previously
reviewed portions of the development recording; they are not an independent or
representative benchmark. `scripts/shape_pilot.py` reproduces the experiment.
Its CSV records individual measurements, while statistics use only adjacent
frames that have an accepted measurement in both methods.

For the example fly, physical slot 0:

| Saved-frame interval | Median ellipse-area ratio, moments / box | P95 absolute axial-angle step: box → moments |
|---|---:|---:|
| [4800,5000) | 0.839 | 36.9° → 21.7° |
| [5480,5540) | 0.845 | 38.2° → 20.8° |
| [6940,7000) | 0.755 | 59.7° → 47.3° |
| [10000,10200) | 0.624 | 61.2° → 42.2° |

In [5480,5540), median absolute minor/major-diameter steps fell from 0.82/1.11
pixels to 0.68/0.85 pixels. This was not uniform across windows: slot 0's major-axis
step increased from 1.07 to 1.27 pixels in [6940,7000). Across all 16 fly/window
combinations, P95 angle steps decreased in 14 and increased slightly in two
(slot 1: 58.5°→59.8° and 52.4°→52.7°). Lower variation can reflect suppressed
noise, but can also suppress real posture/shape changes; it is not an accuracy
measurement. Local artifacts are in `runs/shape-pilot-reviewed/`.

**Complete c51 run.** The opt-in method was also run over all 108293 saved
frames. Comparison against the saved normalized baseline verified exact equality
of x/y (including NaNs), timestamps, missing masks for every geometry field,
detection status/reasons, candidate counts, and runner-up ratios. Missingness is
unchanged at 3687 / 433172 fly-frames (0.851%). All 429485 accepted measurements
used weighted moments; no numerical fallback occurred. Median ellipse areas
were 57–61% of the previous normalized rectangle-derived areas across the four
slots, a substantial geometry change requiring inspection. Detailed invariance
checks and file identities are in [geometry_full_run_summary.json](geometry_full_run_summary.json).
This verifies unchanged center availability, not improved body accuracy.

The new companion is `runs/c51-baseline/c51.normalized.moments.offline.trx`.
Use the existing comparison player to inspect it against acquisition, or the
paired views in `runs/shape-pilot-reviewed/` to compare it against normalized
box geometry. A local [example snapshot](../runs/comparison/weighted_moments_slot0_frame5500.png)
and [gap-onset clip](../runs/comparison/weighted_moments_slot0_gap_onset.mp4) were
generated; these run artifacts are not checked in. Example command:

```bash
PYTHONPATH=src python -m nvsl_offline compare \
  runs/c51-baseline/c51.normalized.moments.offline.trx --fly 0 --start 4800
```

Thirty-eight automated tests pass, including known uniform-ellipse moments,
canonical axis equivalence, weak-periphery weighting and unchanged centers,
acceptance, statuses and timestamps through full synthetic companion generation.

**Interpretation and next step.** The option often gives tighter, steadier
geometry in these development examples without requiring detection tradeoffs.
It can also suppress faint true body parts or distort dimensions when the fixed
detection center differs from the intensity center. Manual body-outline review
is required before promoting it to a default or using its geometry for boundary
contact measurements. Testing additional dates, flies, illumination conditions,
and ordinary frames is now more valuable than repeatedly tuning this same fly.

Using offline measurements only in original gaps is a plausible separate output
policy. It would preserve original finite measurements, including any original
errors, and could introduce discontinuities where two estimators meet. No hybrid
trajectory has been substituted for either source. Originals and experimental
companions remain separate.
