Three-recording full-length check, 15 Nov 2026.

Tracked all saved frames in the three approved flat large-chamber antennae-removed control-genotype recordings. The detector configuration was held fixed: normalized foreground detection, weighted-moment geometry and optional global additive illumination correction. No tracker tuning was performed during these runs. The default correction setting remains `none`.

The selection/configuration/source identities are in [the plan](full_recording_check_plan.json); detailed tracking, gap and behavioral results are in [the summary](full_recording_check_summary.json).

**Tracking and compatibility.** The three recordings contain 324,925 saved video frames, four physical slots each, or 1,299,700 fly-frame opportunities. Full sequential audits were reused where already available or generated for new recordings. Separate complete companions preserve the original frame shapes and timestamps. Source identities, background checksums and downstream loading checks passed. No interpolation or position prediction was used by the detector. Acquisition files, events and read-only references were preserved. Tracking and diagnostic comparison took about 16 minutes with three jobs in parallel.

| Recording | Role | Original missing | Corrected offline missing | Raw pairs, original → offline |
|---|---|---:|---:|---:|
| c42__2025-07-21__20-33-41 | known startup rim-lock | 24.013% | 2.857% | 1 → 0 |
| c52__2025-07-17__12-27-27 | high original loss | 34.656% | 0.806% | 0 → 0 |
| c61__2025-07-27__12-52-43 | ordinary control | 7.632% | 0.384% | 0 → 0 |
| **Combined** | **12 physical slots** | **22.100%** | **1.349%** | **1 → 0** |

Missing counts are 287,240 → 17,534, a net gain of 269,706 finite fly measurements. Percentages are weighted by saved frame counts. Raw and analysis-style suspicious-pair counts coincide here. The earlier uncorrected offline July 21 companion had eight raw pairs (nine analysis-style); the corrected full companion has none. Zero pairs does not establish zero false detections.

| Recording | Physical slot | Original missing | Corrected offline missing |
|---|---:|---:|---:|
| c42__2025-07-21__20-33-41 | 0 | 22.632% | 0.106% |
| c42__2025-07-21__20-33-41 | 1 | 22.091% | 0.394% |
| c42__2025-07-21__20-33-41 | 2 | 12.323% | 1.809% |
| c42__2025-07-21__20-33-41 | 3 | 39.008% | 9.118% |
| c52__2025-07-17__12-27-27 | 0 | 15.914% | 0.045% |
| c52__2025-07-17__12-27-27 | 1 | 52.618% | 1.712% |
| c52__2025-07-17__12-27-27 | 2 | 11.679% | 0.022% |
| c52__2025-07-17__12-27-27 | 3 | 58.414% | 1.445% |
| c61__2025-07-27__12-52-43 | 0 | 13.211% | 0.220% |
| c61__2025-07-27__12-52-43 | 1 | 10.318% | 0.122% |
| c61__2025-07-27__12-52-43 | 2 | 6.220% | 0.147% |
| c61__2025-07-27__12-52-43 | 3 | 0.778% | 1.047% |

**Image review and residual problems.** Generated 104 paired review snapshots, covering predetermined distributed frames (400, 10000, 54000, 90000), longest original gap start/middle/end per slot, largest center disagreements, longest runs of newly missing measurements, and previously reviewed startup rim frames. Reviewed selected images, not every exported image or every recovered frame. Examples show visible flies being recovered near dark sidewalls and at the midpoints of the July 17 slot-1 and slot-3 gaps. A July 21 slot-3 midpoint remains unresolved.

The ordinary-control slot 3 is an availability regression: 0.778% → 1.047% missing. At frame 387 the original tracks the visible fly while the corrected detector rejects ambiguity. At this same frame the uncorrected offline variant selects a long bottom-wall feature, so the corrected abstention improves on that wrong choice but does not recover the fly. July 21 slot 2 frame 400 is another visible-fly/ambiguous example.

Inspection also found startup segmentation contamination by reward-well rims, which the suspicious-pair screen misses:

| Recording / slot | Frame | Observation | Without illumination correction at this frame |
|---|---:|---|---|
| July 17 c52 / 2 | 1113 | Accepted component includes well-rim pixels; major diameter about 62 px and center shifted from original | Also accepted, larger major diameter about 74 px |
| July 17 c52 / 3 | 293 | Accepted well/rim-contaminated component; major diameter about 58 px | Also accepted, major diameter about 74 px |
| July 27 c61 / 2 | 362 | Accepted elongated component includes well-rim pixels; major diameter about 62 px | Ambiguous |

The first two show a limitation that illumination correction reduces but does not eliminate. The third is a correction-specific change from abstention to a contaminated measurement. These are image-based spot checks, not annotated error rates. Across all common finite measurements, only 24 centers differ from the original by more than 10 px; the original is not ground truth, and this comparison does not assess errors in recovered frames. All three highlighted contaminated frames occur before the first recorded training (starts at frames 6949–6985), but they can affect pre-training measurements. No hard startup exclusion was introduced.

Local clips for these cases: [July 17 slot 2](../runs/full-recording-check/review-clips/jul17-slot2-well-merge.mp4), [July 17 slot 3](../runs/full-recording-check/review-clips/jul17-slot3-well-merge.mp4), [July 27 slot 2](../runs/full-recording-check/review-clips/jul27-slot2-well-merge.mp4), and [July 27 slot 3 ambiguity](../runs/full-recording-check/review-clips/jul27-slot3-ambiguous.mp4). Each covers 41 consecutive saved frames and plays at half speed.

**Behavioral comparison.** Both standard upstream analysis runs completed for all six experimental/yoked pairs. They used identical code and options, the same original `.data` and any acquisition note files, separate staged input paths and working directories, and disabled wall caches. The analysis code was read from the explicitly selected installed checkout, not edited. The tracker itself has no dependency on that checkout. Upstream analysis applies its normal missing-position interpolation and eligibility rules to both sources; this does not add interpolated measurements to companions.

Compared standard reward preference indices, SLI and between-reward metrics. Mean SLI below is explicitly T2 sync buckets 2–5, requiring at least three paired valid buckets. No best/worst learner subset was selected. The legacy `--num-trainings 2` limits ordinary plots; standard CSV exports still contain all three recorded trainings.

| Recording | Experimental / yoked slots | Original mean SLI | Corrected offline mean SLI | Change |
|---|---|---:|---:|---:|
| 2025-07-17 c52 | 0 / 2 | 0.1928 | 0.2018 | 0.0090 |
| 2025-07-17 c52 | 1 / 3 | unavailable | unavailable | unavailable |
| 2025-07-21 c42 | 0 / 2 | 0.0604 | 0.0303 | -0.0301 |
| 2025-07-21 c42 | 1 / 3 | unavailable | unavailable | unavailable |
| 2025-07-27 c61 | 0 / 2 | 0.6489 | 0.6707 | 0.0218 |
| 2025-07-27 c61 | 1 / 3 | 0.0658 | 0.0978 | 0.0320 |

The four jointly available mean SLIs change by about -0.030 to +0.032. Two remain unavailable under the same minimum-valid-bucket rule in both sources. Reported actual-reward counts and actual-reward timing summaries are identical. Reconstructed reward entries and trajectory-derived metrics can change.

Between-actual-reward distance by sync bucket has 158 jointly finite cells and 22 missing under both methods, with no availability changes in this section. Some distances change substantially: July 17 experimental pair 1 / yoked slot 3, training-1 bucket 1, increases from 580.829 to 1512.744 in the upstream CSV distance units (saved-image pixels). This is a sensitivity result requiring trajectory review, not evidence that either distance is accurate. Recovery of genuine movement and tracking noise can both increase path length.

Across all 2,058 exported numeric metric cells, 1,441 are paired finite, 17 become finite, 12 become missing, and 588 remain missing. These are cells across many metric/bucket definitions, not independent fly counts. Repeated bucket header labels are paired by column position to avoid overwriting. Full per-pair RPI/SLI values are in [the SLI CSV](full_recording_check_sli.csv), with all standard numeric comparisons in [the metric CSV](full_recording_check_metrics.csv).

**Next review.** The full-length technical check and analysis integration succeeded. Keep the correction optional and inspect the four residual-problem clips before deciding on cohort expansion. The missingness and jump results are encouraging, but sustained wrong locks and contaminated segmentation still require visual assessment. No cohort-wide jobs have been started.

The local [review guide](../runs/full-recording-check/review/review.md) provides commands for every longest original gap. Companions and all review images remain under `runs/full-recording-check/`. `status.json` and `analysis_status.json` now report completion. The initial failed original-analysis launch is retained for diagnostics; `analysis-original-v2/` is the successful original run used in the comparison. Upstream code identity is recorded in the per-source analysis manifests; the upstream command header alone reports the enclosing output repository revision.

Reusable evaluation entry points: `scripts/run_analysis_comparison.py`, `scripts/compare_behavior_exports.py`, and `scripts/review_full_recording_check.py`. The behavioral comparator checks acquisition identities, code identity and options before pairing outputs. Tests cover duplicate column labels, missing-value handling and source-specific command normalization.

Commit review subsequently added rejection of duplicate SLI pair keys and swapped analysis sources. The full suite now passes 59 tests. Recomputing the behavioral comparison with these guards produced byte-identical metric and SLI comparison CSVs; no tracking algorithm or recorded result changed. New analysis manifests point to `result.json` for completion status rather than retaining an initial `complete: false` field.
