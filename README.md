# nvsl-offline-tracking

Independent, experimental classical fly detection for four-slot large-chamber
recordings. Uses NumPy and OpenCV; no runtime imports or assets from `reference/`.
Measurements remain in saved-video pixel coordinates. No interpolation, motion
prediction, cross-frame association, or learned model is used.

The implementation provides four commands:

- `audit`: fully decode the video; check original trajectory/event alignment;
  estimate median, high-quantile background and temporal MAD noise maps from
  evenly spaced grayscale frames.
- `inspect`: export original frames, calibration/candidate overlays, signed
  residuals, masks, noise maps, candidate tables and a blank annotation worksheet.
- `track`: write a new experimental companion `.trx` after a successful audit.
  Reject ambiguous or crop-truncated detections; leave unresolved values NaN.
- `compare`: inspect synchronized original/offline ellipses in an interactive
  player, or export a comparison snapshot/clip without a display.

Use Python 3.10+ with NumPy and OpenCV. Install with
`python -m pip install -e '.[test]'`, then use `nvsl-offline`. With an existing
environment, `PYTHONPATH=src python -m nvsl_offline` runs directly without installing
the package. This session used
`/home/tracking/miniconda3/envs/analysis3.13/bin/python` (OpenCV 4.12.0, NumPy 2.2.6).

To review the existing normalized result immediately, run this from the repository
root in a desktop terminal:

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare \
  runs/c51-baseline/c51.normalized.offline.trx --fly 0 --start 4800
```

The player starts paused, just before the example tracking loss. Original
measurements are yellow on the left; the selected variant is green on the right.
Each panel uses the **same decoded video frame and fixed crop**, with identical
pixel magnification. Missing values have no ellipse or held position. Provenance
(`OBSERVED`, `RECOVERED`, `INTERPOLATED`, `MISSING`) and missing-detection reasons
come from the saved companion, without running detection or analysis again.
Interpolation, if present in a future file, is shown in pink and excluded from
the new-tracking navigation. Shape-equivalent raw OpenCV ellipses are drawn using
the original axis/angle values; no head/tail correction or interpolation is applied.

| Control | Action |
|---|---|
| Space | Play/pause; resume at the beginning after reaching the end |
| `,` / `.` | Previous/next saved frame; pauses playback |
| Type `5500`, then `g` or Enter | Seek to that exact zero-based saved frame |
| Type `12:13.33`, then `g` | Seek by nominal video time (MM:SS or HH:MM:SS) |
| Backspace / Escape | Edit / clear typed seek command |
| Frame slider | Seek within the recording; respects `--start`/`--end` limits |
| `n` / `p` | Next/previous original missing-run start |
| `r` / `b` | Next/previous new finite-run start where original was missing; known interpolations excluded |
| `f` / `a` | Cycle physical chamber slots / show all slots |
| `e` / `c` | Toggle ellipse outlines / center crosses on both panels |
| `+` / `-` | Double/halve playback speed |
| `s` | Save comparison PNG and source/frame JSON to `runs/comparison-snapshots/` |
| `q` or close window | Exit |

Gap/recovery shortcuts use the selected physical slot; in all-slot view they
include individual run boundaries from every slot. Playback visits each saved
frame; if rendering cannot keep up, it slows rather than skipping samples.
`--start` is inclusive and `--end` exclusive, as saved frames or video clock times.
The banner separately shows the original acquisition timestamp. Default image
magnification is 2× for one chamber and 1× for the full frame; override with
`--scale`. The resizable window may additionally scale the displayed image to
fit your screen. Use `e` and `c` to inspect the underlying pixels without markers.
Switch variants by selecting a different companion path, e.g.
`runs/c51-baseline/c51.absolute.offline.trx`; `--label` supplies a custom panel title.

Interactive playback requires a GUI-enabled OpenCV build (`opencv-python`, now
the package dependency). Do not install it alongside `opencv-python-headless` in
the same environment. The existing `analysis3.13` environment has a Qt5 GUI build.
Headless environments can still render/export with their existing headless build:

```bash
PYTHONPATH=src python -m nvsl_offline compare \
  runs/c51-baseline/c51.normalized.offline.trx --fly 0 --start 5500 \
  --snapshot runs/comparison/frame5500.png

PYTHONPATH=src python -m nvsl_offline compare \
  runs/c51-baseline/c51.normalized.offline.trx --fly 0 --start 4800 --end 5000 \
  --output runs/comparison/gap-onset.mp4
```

Export visits every frame in the selected interval; `--speed` changes the encoded
frame rate, not the trajectory index or sample count. Exports and screenshots
refuse to replace existing files. Source paths are inferred from the companion's
metadata and checked against the acquisition files; original/variant frame shapes
and timestamps must agree exactly. For legacy variants without source metadata,
provide `--video`, optionally `--original` and `--data`; use `--layout large2`
for chamber crops. Finite values without quality metadata are labeled with
unknown provenance, including in new-finite-run navigation. No tracking correctness
is inferred from file presence.

Local example artifacts (not checked in): a [comparison snapshot](runs/comparison/normalized_slot0_frame5500.png)
and [gap-onset clip](runs/comparison/normalized_slot0_gap_onset.mp4) were generated
from the real recording. The clip has 200 consecutive saved frames (4800–4999)
at 7.5 fps. The interactive controls were tested without creating a desktop
window; frame seeking, rendering and MP4 export were also exercised on actual data.

To label a benchmark, first use `compare` to choose saved frames. Make a
selection CSV with one row per frame and physical fly slot (both zero-based).
Companion paths may be absolute or relative to the selection CSV. Optional
columns such as `split`, `window`, and `notes` are copied to the output.

```csv
companion,fly,frame,split,window
/path/to/c62.body-core.offline.trx,2,45934,tune,ambiguous-gap
/path/to/c62.body-core.offline.trx,2,90375,evaluation,wall-lock
```

```bash
PYTHONPATH=src python scripts/annotate_benchmark.py selection.csv \
  --output runs/benchmark/labels.csv
```

Click the visible body center in the crop, or press `a` for no fly and `u`
for unclear. Coordinates are saved in the original full-frame pixel system.
Each label is saved immediately. Press `n` or `p` to navigate targets, `,`
or `.` to view nearby frames, and `t` to return to the target. Labels can
only be entered on the target frame. Tracker centers are hidden by default.
Press `o` to show them temporarily, `x` to clear a label, or `q` to quit.
This requires a desktop with GUI-enabled OpenCV. Use `--scale 1` if the window
is too large for your display.

Rerun the same command with the same selection and output to resume at the
first unlabeled row. An empty `status` means pending, while `no_fly` and
`unclear` are explicit labels. Use one annotation session per output CSV.

Example for the reviewed 39 mm large-chamber recording:

```bash
PYTHONPATH=src python -m nvsl_offline audit \
  '/media/Synology4/Yang Chen/2025-05-30/afternoon/c51__2025-05-30__19-22-58.avi' \
  --layout large2 --samples 129 --output runs/c51-baseline

PYTHONPATH=src python -m nvsl_offline inspect runs/c51-baseline \
  --frames 4827,4900,5500,6200,6967,6968,10000 \
  --mode both --output runs/c51-review-baseline

PYTHONPATH=src python -m nvsl_offline track runs/c51-baseline \
  --mode absolute --output runs/c51-baseline/c51.absolute.offline.trx

PYTHONPATH=src python -m nvsl_offline track runs/c51-baseline \
  --mode normalized --output runs/c51-baseline/c51.normalized.offline.trx

python -m pytest -q
```

An experimental lighting correction is available on `inspect` and `track` with
`--illumination global_offset` (default: `none`). It estimates a shared additive
brightness change from spatially distributed chamber pixels, adjusts the stored
background, then applies the existing thresholds and candidate-selection rules.
It uses only the current image and stored background; no position prediction or
interpolation is involved. See the [illumination pilot](docs/illumination_pilot.md)
for the method, selected-interval results, and limitations. Full companions record
the setting and one offset/tile-disagreement value per saved frame in
`tracking_quality`. Existing audits/backgrounds can be reused; write a new companion.

Experimental body-core segmentation is available with `--segmentation body_core`
(default: `components`), using `--core-radius 2 --core-growth 2`. It removes thin
attachments using distance-transform seeds and bounded foreground growth, then
recalculates the weighted center and ellipse before candidate selection. It uses
no well-location prior or trajectory interpolation and requires weighted-moment
geometry. See the [body-core pilot](docs/body_core_pilot.md) for results, controls,
and remaining small-body/ambiguity failures. The previous detector remains default.

Output directories and companion filenames must be new. Acquisition video,
`.trx`, and `.data` files are read-only inputs. Load only trusted acquisition
pickles. `audit` discovers `experiment.trx`/`.data` or `experiment.avi.trx`/`.data`,
rejecting ambiguous sidecars unless explicit `--trx`/`--data` paths are given.
The layout must be explicit: `large` is the older 35 mm layout; `large2` is 39 mm.
The saved template transform is reused without re-registration. Visually confirm
the floor and sidewall search bounds before accepting measurements.

Background defaults are 129 samples, pixelwise 80th percentile, temporal
`1.4826 * MAD`, noise smoothing sigma 3 pixels, and noise floor 3 grayscale levels.
Samples are stored as a disk-backed array; quantiles are computed in row tiles.
For 720×720 video, 129 grayscale samples occupy about 64 MiB on disk; the background
estimator's working memory scales with tile height and sample count. Trajectory
and quality storage still scale with saved-frame count. `--quantile .5` tests the
median background. A single background assumes sufficient unobstructed samples
and comparable illumination; stationary contamination and lighting drift remain
limitations. The noise map includes temporal illumination variation, not just
sensor noise. There is no gain/offset or LED-state correction yet.

Absolute detection uses `background - gray > 25`. Normalized detection requires
both an absolute difference above 6 and `(background - gray) / noise > 4`.
Because the noise floor is 3, this can amount to lowering the raw threshold to
12 in stable regions. To isolate the value of spatial normalization, include an
absolute-threshold-12 control, rather than crediting all gains to normalization:

```bash
PYTHONPATH=src python -m nvsl_offline inspect runs/c51-baseline \
  --frames 4827,4900,5500,6200,6967,6968,10000 \
  --mode absolute --threshold 12 --output runs/c51-review-threshold12
```

All modes share chamber ROIs, contour filters (40–1000 px² and at least five
contour points), and OpenCV `minAreaRect` body estimates. The largest surviving
candidate is rejected if the second is at least half its area, or if it touches
the search ROI boundary. `--largest-only` and `--allow-truncated` explicitly
disable these checks for control experiments. These are experimental rejection
rules, not calibrated correctness guarantees. The search extends 24 saved-image
pixels beyond each floor, clipped to its chamber partition and image bounds.
The initial 12-pixel margin clipped the dark-edge fly in the development example;
24 is an inspected development choice, not a universal sidewall calibration.
This absolute control differs from acquisition in background and spatial/rejection
rules; it does not reproduce camera-time detection exactly.

Default body geometry: `inspect` and `track` use `--shape-fit weighted_moments`.
This leaves foreground segmentation, contour filters, candidate selection and
accepted x/y centers unchanged. Within the accepted component, it weights pixels
by squared positive background difference and estimates an ellipse from second
moments about that same center. Weak peripheral pixels have less influence than
in an enclosing rectangle. No temporal smoothing, position interpolation, extra
detections or missing-value filling is introduced. Numerically unresolved shape
estimates fall back to the original rectangle; per-frame `tracking_quality`
records the chosen method. Explicit `--shape-fit box` selects the initial method.
Existing companion files retain the geometry and configuration they were generated
with; changing the default does not rewrite them. The original baseline results
above used box geometry. To reproduce those exact comparisons, specify
`--shape-fit box`. Foreground-thresholding defaults have not changed.

Moment ellipses use w as minor diameter, h as major diameter, and theta modulo
180 degrees in the existing OpenCV convention. Canonicalizing equivalent box
axes before evaluating angle changes avoids mistaking width/height swaps for
real rotation. Neither method resolves head/tail direction, and orientation is
intrinsically uncertain for nearly circular silhouettes. Smaller or steadier
ellipses are not necessarily anatomically more accurate; weighted moments can
underrepresent faint real body parts. It is now the working default following
visual review, while quantitative anatomical validation remains outstanding.

```bash
PYTHONPATH=src python -m nvsl_offline track runs/c51-baseline \
  --mode normalized --shape-fit weighted_moments \
  --output runs/c51-baseline/c51.normalized.moments.offline.trx

PYTHONPATH=src python scripts/shape_pilot.py runs/c51-baseline \
  --output runs/shape-pilot-review
```

The second command compares shapes on explicit development windows, checks
identical masks/decisions/centers, and exports paired shape views plus statistics.
Use repeatable `--interval START:STOP` arguments to choose other windows. Results
and interpretation are in [the geometry pilot](docs/geometry_pilot.md).
The [weighted-moment explanation](docs/weighted_moment_notes.md) develops the
mathematics, conventions and rationale in a log-ready form.

Review overlays show original ellipses in red, accepted candidates in green,
and candidate contours in orange. Raw `_frame.png` images contain no predictions
and should be used for manual labeling. `annotations.csv` has one blank row per
physical slot and requested frame. Record visibility (`visible`, `ambiguous`,
`occluded`, `empty`), image-plane center x/y, full major/minor body diameters, and
body-axis `axis_theta` in degrees modulo 180: 0 vertical/up, 90 horizontal/right.
Exclude extended legs/wings and detached reflections consistently. Do not force
a center/angle on an unlocalizable body. Body outlines and temporal review may
be needed to adjudicate sidewall postures. These small development reviews are
not a held-out benchmark or annotation-completion claim.

Companions preserve `x,y,w,h,theta,ts`, physical slot ordering and source timestamps.
`tracking_metadata` records configuration, source identities, package-code hash,
version/commit, calibration and background checksum. Source sidecars are SHA-256
hashed; video identity uses path/size/mtime unless `audit --hash-video` is selected.
Sources and background checksums are rechecked before processing/publishing.
Event index N is allowed as a shutdown endpoint because acquisition records the
next saved index; values outside [0,N] fail. Repeated/nonmonotonic timestamps are
reported and retained. Inconsistent source geometry masks or invalid diameters
fail rather than being repaired.

`tracking_quality.status` uses uint8 codes: observed=0, recovered=1,
interpolated=2 (reserved, never emitted), missing=3. “Recovered” means an accepted
image-based candidate where original x was NaN; it is not proof of correct
localization. Reasons, candidate counts, runner-up area ratios and changes from
original finite measurements are retained. Position/shape/orientation uncertainty
is explicitly uncalibrated and unknown, not zero. The default processes every
slot; use `track --exclude-slots 3` if occupancy/exclusion is independently known.
Fully missing original slots are not automatically treated as empty.

The existing analysis does not automatically select these companions or honor
their quality metadata. Keep them separate until explicit source selection and
measurement validation are added. No acquisition events are changed.

Investigation and proposed development plan: [offline tracking investigation](docs/offline_tracking_investigation.md).

Implementation evidence and limitations: [first baseline run](docs/baseline_first_run.md).

False-positive testing with the tracker held fixed: [jump screening and image
review](docs/false_positive_testing.md). `scripts/scan_suspicious_jumps.py` ranks
the large-chamber section of the local experiment list without decoding videos;
`scripts/compare_jump_trajectories.py` compares completed companions. Both retain
raw and analysis-interpolated diagnostic counts. Neither changes source data or
fills tracking gaps. A suspicious jump is a review cue, not a false-positive label.
The [three-recording full-length check](docs/full_recording_check.md) evaluates
the optional illumination correction, documents remaining startup segmentation
errors, and compares standard behavioral outputs using isolated tracking sources.
The comparator also reports full-set totals, frame-weighted missingness and
per-slot improvements/regressions; its aggregate includes every supplied paired
slot and requires one companion per recording to avoid double-counted exposure.

Supporting evidence: [recording probes](docs/recording_probe.json) and [inspection contact sheet](docs/inspection_contact_sheet.png).
