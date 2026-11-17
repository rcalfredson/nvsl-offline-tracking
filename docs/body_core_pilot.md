Body-core segmentation pilot, 15 Nov 2026.

Implemented an optional intrinsic segmentation stage to separate a substantial
fly body from thin attached foreground, without well locations or trajectory
prediction. Defaults remain unchanged: `--segmentation components` uses the
previous detector. The experimental setting is `--segmentation body_core`.

**Method.** Keep normalized thresholding and global additive illumination
correction fixed. For each chamber's threshold mask:

1. Compute a Euclidean distance transform, padding the crop with background.
2. Form core seeds from pixels more than 2 saved-image pixels from background.
   Label their connected components separately.
3. Expand each seed for two synchronous 8-neighbor steps, inside the original
   foreground only. Pixels neighboring competing labels remain unassigned;
   labels never overlap. Growth stops after two steps rather than continuing
   until the original attached arc has been reconstructed.
4. Apply the existing 40–1000 px² contour-area limits to the refined regions,
   rank by refined area, and retain the existing 0.5 ambiguity cutoff. Parent
   components touching the ROI border remain marked truncated even if erosion
   removes their boundary pixels.
5. Calculate the positive-residual-squared weighted centroid of the selected
   body mask, then its second moments about that centroid. Use the existing
   eigenvalue convention: full diameters `4 * sqrt(eigenvalue)`, minor width,
   major height, and OpenCV angle modulo 180 degrees.

The centroid is deliberately recalculated, unlike the previous weighted-moment
method that retained the enclosing rectangle's center. Refinement happens before
candidate acceptance, on every frame, including frames the baseline rejects.
There is no jump-triggered correction, interpolation, motion limit, or use of
the original fly coordinates in segmentation. Chamber ROIs are still required;
well geometry is not used. No-core or rejected measurements remain NaN.

The core radius and growth count are configurable (`--core-radius 2`,
`--core-growth 2`), in saved-image pixels and 8-neighbor steps. Body-core mode
requires weighted-moment geometry. It records `core_weighted_moments` in the
existing shape-method quality field and supports an explicit `no_core` missing
reason. Inspection candidate CSVs include core and reconstructed-body pixel counts.

**Initial diagnostics and experiment.** In the three known well-rim examples,
the thin arc disappears from the distance-transform core while a compact region
remains on the fly. The August 2 blurred-fly example retains one core at radius
2, fragments at 2.5, and has none at 3. Radius 2 and growth 2 were therefore
selected as a development prototype, not optimized on a held-out benchmark.

Evaluated 6,840 saved frames from seven recordings, processing all four slots:
startup and distributed ordinary windows, the known contaminated frames and
their neighbors, two recovered-gap windows, the previously reviewed genuine
jumps, the May baseline, and July 17 c61's previously ambiguous startup fly.
The [plan](body_core_pilot_plan.json) lists all intervals and snapshots.

The baseline is the prior **illumination-corrected offline detector**, not the
real-time tracker. On the three recordings with complete corrected companions,
baseline reprocessing reproduced the stored geometry and missing masks within
the configured float tolerance. Both variants used the same decoded images,
backgrounds, noise estimates, thresholds and chamber calibration. Original files
and existing companions were preserved.

A centroid-only ablation recalculates weighted moments on the original accepted
component without cleaning its mask. Its acceptance/missingness is unchanged;
it is a diagnostic alternative, not another production tracking mode.

| Known contaminated example | Previous major diameter | Centroid only | Body-core major diameter |
|---|---:|---:|---:|
| July 17 c52, slot 2, frame 1113 | 61.77 px | 23.92 px | 18.35 px |
| July 17 c52, slot 3, frame 293 | 57.53 px | 25.05 px | 14.37 px |
| July 27 c61, slot 2, frame 362 | 61.74 px | 21.99 px | 18.84 px |

Visual spot checks show the body-core centers on the visible flies and exclusion
of the attached arcs. Recentring alone accounts for a substantial improvement;
core reconstruction additionally tightens the body region in these examples.
Smaller axes alone are not accuracy evidence, and no annotated position or
ellipse error benchmark has been completed.

| Selected windows | Fly-frame opportunities | Previous missing | Body-core missing |
|---|---:|---:|---:|
| Startup | 22,000 | 4,154 | 391 |
| Ordinary distributed frames | 4,400 | 8 | 14 |
| Recovered-gap controls | 440 | 0 | 1 |
| Genuine-jump controls | 520 | 1 | 1 |
| **Total** | **27,360** | **4,163 (15.216%)** | **407 (1.488%)** |

These are selected-interval results, not full-recording missing rates. There
are 3,882 newly finite measurements and 126 newly missing measurements. Many
recoveries replace ambiguity after thin artifacts are removed. July 17 c61 slot
3 frame 668 and July 21 c42 slot 2 frame 400 show the accepted ellipse on the
visible fly where the previous detector abstains. Newly finite does not imply
correct without inspection.

Both 40-frame August 4 slot-2 jump windows remain fully measured. August 2
slot 0 retains 49/50 measurements in its jump window. Raw pairs across all
selected intervals change from 1 to 2; the additional pair is the already
reviewed genuine August 4 jump at indices 60549/60550. The centroid-only ablation
also registers this pair. No raw pairs appear in the sampled non-jump intervals;
counts do not include transitions across interval boundaries.

**Remaining limitations.** Some useful measurements are rejected. July 17 c52
accounts for 112 of the 126 newly missing measurements and has a small net
availability loss over its sampled windows. At slot 1 frame 54002, the visible
sidewall fly's reconstructed contour has area 39 px², just below the unchanged
40 px² cutoff. At slot 3 frame 61592, the two reconstructed regions have contour
areas 29 and 30 px² and are rejected. Slot 2 frame 714 yields two competing
regions (43 and 82.5 px²), causing ambiguity. That baseline measurement is itself
visibly broad over a well, so not every newly missing measurement is a loss of
a correct position. The old area threshold was chosen for unrefined components;
applying it after reconstruction needs evaluation. Multiple seeds can also
fragment one body or represent separate artifacts; this version does not merge
them using an additional appearance model.

Thickness is not proof of a fly: substantial artifacts may survive, while thin
sidewall or blurred bodies may fail. Parameters are expressed in image pixels,
so performance at different camera scales is not established. Orientation of
nearly circular bodies remains uncertain. Keep the mode experimental and review
both the new finite positions and dropped measurements before changing defaults
or generating cohort-wide outputs.

**Artifacts and commands.** [Machine-readable results](body_core_pilot_summary.json)
include configurations, source identities, all per-slot/window counts, the
centroid-only ablation and rejection spot checks. Local images and arrays are
under ignored `runs/body-core-diagnostics/` and `runs/body-core-pilot/`.

- [Distance map and cores, July 17 slot 2](../runs/body-core-diagnostics/well-jul17-s2.png)
- [Previous / centroid-only / body-core comparison, frame 1113](../runs/body-core-pilot/jul17-c52/frame0001113-slot2.png)
- [Recovered previously ambiguous fly, frame 668](../runs/body-core-pilot/jul17-c61/frame0000668-slot3.png)
- [Retained blurred fly, August 2 frame 74631](../runs/body-core-pilot/jump-aug02/frame0074631-slot0.png)
- [Newly rejected sidewall fly, frame 54002](../runs/body-core-pilot/rejection-review/frame54002-slot1.png)

```bash
PYTHONPATH=src python scripts/evaluate_body_core.py \
  docs/body_core_pilot_plan.json --output runs/body-core-pilot-new

PYTHONPATH=src python -m nvsl_offline inspect \
  runs/full-recording-check/02-c52__2025-07-17__12-27-27/audit \
  --mode normalized --illumination global_offset --segmentation body_core \
  --core-radius 2 --core-growth 2 --frames 293,1113,54002,61592 \
  --output runs/body-core-inspection
```

The same segmentation options are available on `track` for writing a new full
companion. This pilot generated interval arrays and review artifacts only;
no full-length body-core companion or revised behavioral comparison was generated.
Synthetic tests cover bounded growth, disjoint seed ownership, inherited
truncation, centroid/coordinate conventions, thin-attachment removal, explicit
missingness and full companion contracts. The complete suite passes 68 tests.
