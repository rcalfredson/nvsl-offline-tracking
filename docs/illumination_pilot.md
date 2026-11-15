Experimental illumination correction, 15 Nov 2026.

Follow-up: [three-recording full-length check](full_recording_check.md), including
behavioral comparisons and residual well-rim segmentation problems.

The user's review identified early wall/rim locks, including sustained incorrect
positions without suspicious pairs. Some later suspicious pairs are genuine fly
jumps, notably August 4 c31 slot 2 and August 2 c52 slot 0. Brightness changes
between startup and later frames motivate testing a lighting correction inside
detection, rather than filtering the trajectory by displacement.

**Method.** The optional `--illumination global_offset` setting changes foreground
subtraction; the default remains `none`. Each frame uses the same stored
recording-wide background and noise map. Split each of the four chamber ROIs into
4x4 tiles. In every tile, sample a stride-four grid of pixels and take the median
of `current_gray - stored_background`. The median of the 64 tile estimates is the
additive offset. Tiles have equal weight so the estimate reflects spatially
distributed image evidence rather than the darkest component. A small fly can
affect a tile without dominating the recording-wide estimate.

```text
adjusted_background = stored_background + offset
corrected_residual  = adjusted_background - current_gray

negative offset: darken the background to match a darker frame
positive offset: brighten the background to match a brighter frame
```

Apply the existing residual/noise thresholds and candidate rules to that residual.
Noise, area limits, ambiguity cutoff, chamber calibration and body estimator are
unchanged. Corrected residuals also supply the body-moment weights, so geometry
and centers can change as the foreground mask changes. There is no motion limit,
position interpolation, temporal smoothing, startup cutoff or use of the original
fly position in the correction. Each frame is processed independently.

This first version estimates an offset, not a gain or a local brightness field.
It assumes most sampled tiles share an additive lighting change and remain
dominated by background. Spatially varying illumination, saturation, shadows,
reflections or camera movement can violate that assumption. The median absolute
deviation of tile estimates is recorded as spatial disagreement; it is not a
calibrated uncertainty or an acceptance gate. The normalized noise model is not
refitted in this pilot.

**Selected-interval experiment.** Reused six completed audits and their background
files. Evaluated frames 0–1299 and 10000–10099 of each recording, plus August 2
frames 74610–74659 and August 4 frames 44270–44309 and 60530–60569. These are
8,530 unique saved frames and 34,120 fly-frame opportunities across all slots.
Both variants were evaluated on the same decoded images. Recomputed uncorrected
measurements match the stored companions within 0.000016 across all five geometry
fields, including identical missing masks. No source or existing companion was
modified. The experiment produced interval arrays, CSV measurements and review
images, not new full-length trajectories.

Startup offsets have medians of -7 gray levels for May 30, July 17 and July 21,
-6 for July 31 and August 2, and -3 for August 4. At frames 10000–10099 each
recording has a median offset of -1. This supports a startup lighting mismatch;
it does not establish that lighting explains every failure.

| Reviewed rim-lock slot | Raw pairs, baseline → corrected | Missing frames, baseline → corrected |
|---|---:|---:|
| May 30 c51, slot 1 | 5 → 0 | 1066 → 1 |
| July 17 c61, slot 3 | 5 → 0 | 864 → 1285 |
| July 21 c42, slot 0 | 8 → 0 | 1054 → 86 |
| July 31 c62, slot 0 | 5 → 0 | 680 → 186 |
| July 31 c62, slot 2 | 15 → 0 | 541 → 0 |

These rows cover frames 0–1299, with zero-based physical slot IDs. Across all
24 slots in the startup windows, raw pairs fall from 38 to 0, and missing
measurements fall from 15,199/31,200 (48.715%) to 5,466/31,200 (17.519%).
Other slots can lose availability too: August 2 slot 1 goes from 332 to 650
missing frames and August 4 slot 0 from 37 to 246. Counts alone cannot tell
whether each removed or added measurement is correct.

Spot checks show recovery of the visible fly instead of the rim at May 30 frame
212, July 21 frame 1005, and July 31 slot 2 frame 510. July 17 slot 3 frame 668
becomes ambiguous rather than selecting the lower wall: that is abstention,
not position recovery. These are assistant visual spot checks, not a completed
manual localization benchmark. Sustained locks must also be assessed directly;
removing their associated suspicious pairs does not measure their full duration.

**Controls.** All 2,400 fly measurements in the later ordinary windows remain
finite in both variants, with zero raw pairs. Centers do move as segmentation
changes; per-recording median differences are about 0.22–0.30 pixels, and a
single common-frame difference reaches about 5.04 pixels. Differences from the
baseline are not errors against ground truth.

The August 4 slot-2 jump window at 60530–60569 retains measurements in all 40
frames. Its raw pair count changes from 1 to 0 despite that availability being
preserved. Centers differ by a median 0.24 pixels and at most 2.06 pixels in
this window; threshold-sensitive pair counts are not a correctness score. The
earlier retained pair at 44287/44288 remains. August 2 slot 0 retains its 49/50
finite measurements in the 74610–74659 window, including the inspected blurred
fly at frame 74631. That analysis-only pair is not measured by these raw counts.
Neither genuine jump is rejected merely for moving quickly.

**Artifacts and reproduction.** Detailed per-slot/window results and configurations
are in [illumination_pilot_summary.json](illumination_pilot_summary.json); selection
is in [illumination_pilot_plan.json](illumination_pilot_plan.json). All sampled
measurements and comparison images are local, ignored artifacts under
`runs/illumination-pilot/`. Image panels show acquisition, uncorrected baseline
and corrected detection, respectively:

- [May 30: rim to fly, frame 212](../runs/illumination-pilot/01-c51__2025-05-30__19-22-58/frame0000212-slot1.png)
- [July 17: ambiguous after correction, frame 668](../runs/illumination-pilot/02-c61__2025-07-17__12-27-30/frame0000668-slot3.png)
- [July 31: rim to fly, frame 510](../runs/illumination-pilot/04-c62__2025-07-31__20-08-27/frame0000510-slot2.png)
- [August 4: retained fly, frame 60550](../runs/illumination-pilot/06-c31__2025-08-04__14-48-45/frame0060550-slot2.png)
- [August 2: retained blurred fly, frame 74631](../runs/illumination-pilot/05-c52__2025-08-02__22-03-53/frame0074631-slot0.png)

```bash
PYTHONPATH=src python scripts/evaluate_illumination.py \
  docs/illumination_pilot_plan.json --output runs/illumination-pilot-new

PYTHONPATH=src python -m nvsl_offline inspect runs/c51-baseline \
  --mode normalized --illumination global_offset --frames 212,400,10000 \
  --output runs/c51-illumination-review

# Optional full-length output using an existing audit, to a new companion:
PYTHONPATH=src python -m nvsl_offline track runs/c51-baseline \
  --mode normalized --illumination global_offset \
  --output runs/c51-baseline/c51.global-offset.offline.trx
```

Full companions retain the original timestamp and frame contract, configuration
and source provenance. `tracking_quality.illumination_offset` and
`illumination_tile_mad` each have one value per saved frame, in gray levels.
Default uncorrected processing records zero values; the configuration distinguishes
disabled correction from an estimated zero shift. The comparison player labels
the selected lighting method. Existing trajectories and experiment events remain
unchanged.

The next decision requires reviewing correct startup/sidewall measurements that
became missing and measuring residual wall-lock duration, then running complete
corrected companions and broader controls. Keep the method optional until that
review. This is a promising selected-interval pilot, not a full-recording accuracy
claim or a basis for treating every added finite position as correct.
