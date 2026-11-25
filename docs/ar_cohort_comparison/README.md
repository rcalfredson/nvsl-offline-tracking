Antennae-removed large-chamber comparison plan, 17 Nov 2026.

The tracker configuration is frozen for the first cohort comparison. This is
preparation only: no cohort-wide tracking or behavioral analysis has been launched.

The committed tracker is `17d140d653d9b20d7c9f1405d4ee79d2d5f64955`.
[freeze.json](freeze.json) records its package hash, library versions, all detector
settings, background settings, and audit parameters. The detector configuration
matches the manually reviewed three-recording full check. Preparation introduces
uncommitted evaluation files, so producer reports the overall worktree as dirty;
the tracker package itself matches the recorded commit. Freeze the *settings used
for this run*, rather than change CLI defaults: illumination correction and
body-core segmentation must still be selected explicitly.

The fixed settings include 129 recording-wide grayscale samples, the 80th
percentile background, temporal-MAD noise with floor 3 and Gaussian sigma 3,
`large2` chamber geometry with 24-pixel sidewall margin, normalized detection
with signal floor 6 and score threshold 4, global additive illumination correction,
and body cores with radius 2 and two growth steps. Area limits remain 40–1000,
ambiguity ratio 0.5, with ambiguity/truncation rejection and weighted moments.
There is no trajectory interpolation. The stored absolute threshold 25 is not
the active segmentation cutoff in normalized mode.

The user completed the predetermined QA windows for all 12 flies in the three
full recordings and accepted all 12, with caveats for a persistent dark interval
and startup losses. Reported recoveries generally followed the fly, known merged
well-rim ellipses were corrected, and several original head/wing detections were
improved. Remaining missed frames and occasional wing effects remain limitations.
This is qualitative acceptance of reviewed intervals, not an annotation benchmark
or proof that whole recordings are error-free.

| Cohort | Listed recordings | Ready for audit | Held |
|---|---:|---:|---:|
| Flat antennae removed, control genotype | 35 | 34 | 1 |
| Agarose antennae removed, control genotype | 32 | 32 | 0 |

The authoritative selections are the commands following the control-genotype
headings on lines 369 and 374 of `full_video_list.txt` (commands on lines 370 and
375). All 16 flat and 14 agarose patterns matched recordings. Each pattern set
agrees exactly with notebook variables `FIGURE_4D_ANTENNAE_REMOVED_VIDEO_GLOBS`
and `FIGURE_4T_ANTENNAE_REMOVED_VIDEO_GLOBS`, respectively. The groups do not
overlap. The separate PFNd-silenced antennae-removed cohorts are not included.

The notebook was absent from `reference/nvsl-analysis/notebooks/` at preparation
time. The read-only crosscheck used
`/home/tracking/code/YL/nvsl-analysis/notebooks/manuscript_figure_panels.ipynb`
(cells 71 and 95, zero-based). No commands in either source were executed.
Source hashes and literal selections are preserved in the manifests.

[Flat manifest](flat-ar-control.json) and [CSV](flat-ar-control.csv);
[agarose manifest](agarose-ar-control.json) and [CSV](agarose-ar-control.csv).
They record source video identity by path/size/mtime, SHA-256 for acquisition
sidecars and notes when present, header and trajectory frame counts, phase events,
calibration, timestamps, and original missingness/gaps by physical slot.

The held flat recording is `c32__2025-07-29__20-05-41`: 65 saved frames,
incomplete training phases, and no valid saved calibration transform. Retain it
in the inventory and omit it from both analysis variants with the recorded reason.
Other exceptions are warnings, not silent exclusions:

- Flat `c62__2025-07-18__20-56-27`: original slots 1 and 3 entirely missing.
- Agarose `c52__2025-07-27__17-03-54` and `c32__2025-08-01__21-56-23`:
  original slot 3 entirely missing.
- Six other recordings contain repeated timestamps. Preserve the timestamps;
  the manifests retain their counts. No decreasing timestamps were found.

An entirely missing original slot does not establish chamber occupancy. Inspect
these slots before interpreting any new finite output as a recovery.

Next, process the 34 ready flat recordings, with bounded parallelism and per-job
status/logs. Full sequential decoding in the audit must confirm actual saved-frame
counts before tracking; this preflight checked headers and sidecars only. Record
every job's explicit frozen configuration and source identities. Tracking still
produces all four physical slots, preserves acquisition order/timestamps, and
writes separate companions with NaNs for unresolved measurements.

Launch the flat batch from the repository root in the analysis environment:

```bash
PYTHONPATH=src python scripts/run_cohort_tracking.py \
  --cohort docs/ar_cohort_comparison/flat-ar-control.json \
  --output runs/flat-ar-control-body-core --workers 3
```

The output directory must be new. Add `--dry-run` to validate the local frozen
plan/runtime and list jobs without creating outputs or decoding videos. On an
interrupted/failed run, repeat the launch command with `--resume` and the same
output directory. Completed companions and valid completed audits are reused
after identity/configuration checks; partial attempts are retained and retried
in new attempt directories. The runner rejects changed inputs, package/library
versions, or changed runner/diagnostic code on resume. To reuse inputs from
another run, a job may specify absolute paths in `reuse_audit` and
`reuse_companion`. The runner verifies the audit and companion against the
new plan before using them in place. An invalid reuse entry fails the job.
A batch lock prevents two
launchers from writing to the same output directory. Ctrl-C or SIGTERM stops the
worker process groups and preserves completed work.

`status.json` records each job's stage, output paths, warnings, and errors.
Each recording has an append-only `job.log` with decode/tracking progress.
After each companion, the runner computes original/offline missingness and jump
diagnostics; the final status includes their pooled aggregate. The held recording
remains in `held`, outside the processing jobs. All-missing-slot warnings remain
visible and no physical slots are automatically excluded. This launcher does not
run behavioral analysis. The completed status has the `jobs/video/companion`
fields expected by `scripts/run_analysis_comparison.py`; wait for
`complete: true` before using it as analysis input.

[analysis_plan.json](analysis_plan.json) records the upstream code identity and
standard options used for the earlier comparisons. Analyze original and offline
sources in isolated directories with the same acquisition metadata/notes and
cache policy. Start with RPI, SLI and between-actual-reward distance. Compare
matched recording/fly-pair/bucket values where both are assessable, and report
newly assessable or newly unassessable values separately. The selection `-f 0-1`
refers to experimental/yoked pairs, not only two of the four physical slots.

Recorded stimulus/reward events remain acquisition history. Reconstructed
behavioral events and trajectory metrics may change. Large metric changes need
trajectory inspection; they do not themselves establish accuracy. No learner
subset is selected for the primary comparison. The notebook's more specialized
tortuosity/COM/return-leg recipes are recorded as additional candidates, requiring
their own matched option/export setup rather than being implied by the standard
analysis run. Evaluate agarose as a separate second cohort.

To reproduce the input preflight into a **new** directory:

```bash
PYTHONPATH=src python scripts/prepare_cohort_comparison.py \
  --list full_video_list.txt \
  --notebook /home/tracking/code/YL/nvsl-analysis/notebooks/manuscript_figure_panels.ipynb \
  --date 2026-11-17 --output runs/ar-cohort-preflight-new
```
