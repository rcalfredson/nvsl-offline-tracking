#!/usr/bin/env bash
# Run from the offline tracker root. Arguments are input and output directories.
(
  set +e
  root="$PWD"
  py="${ANALYSIS_PYTHON:-/home/tracking/miniconda3/envs/analysis3.13/bin/python}"
  checkout="${NVSL_ANALYSIS_ROOT:-/home/tracking/code/YL/nvsl-analysis}"
  inputs="${1:-$root/runs/ar-three-way-behavior-inputs}"
  results="${2:-$root/runs/ar-three-way-behavior}"
  if [[ ! -r "$checkout/analyze.py" || ! -x "$py" ]]; then
    printf 'Set NVSL_ANALYSIS_ROOT and ANALYSIS_PYTHON to the analysis checkout and executable.\n' >&2
    exit 1
  fi
  checkout=$(cd "$checkout" && pwd -P) || exit 1
  inputs=$(cd "$inputs" && pwd -P) || exit 1
  mkdir "$results" || exit 1
  results=$(cd "$results" && pwd -P) || exit 1
  # The preparation script writes absolute companion paths, so this copy is portable.
  cp "$inputs/tracking.json" "$inputs/agarose-videos.txt" "$inputs/flat-videos.txt" "$results" || exit 1
  git -C "$checkout" rev-parse HEAD > "$results/analysis-commit.txt" || exit 1
  git -C "$checkout" status --short > "$results/analysis-worktree.txt" || exit 1
  failed=0

  for cohort in agarose flat; do
    videos=$(cat "$results/$cohort-videos.txt") || exit 1
    for variant in original nov17-body-core nov20-two-rule; do
      out="$results/$cohort/$variant"
      mkdir -p "$out/imgs" "$out/exports" || exit 1
      command=("$py" "$checkout/analyze.py" -v "$videos"
        -f 0-1 --rCC 15 --num-trainings 2 --rpd
        --best-worst-trn 2 --sli-use-training-mean
        --sli-select-skip-first-sync-buckets 1
        --sli-min-valid-sync-buckets 3 --no-wall-cache --hidePltTests
        --tracking-variant "$variant"
        --tracking-manifest "$results/tracking.json" --no-tracking-fallback
        --cross-experiment-sli-export "$out/sli.csv")
      if (
        cd "$out" || exit 1
        export MPLBACKEND=Agg MPLCONFIGDIR="$out/mpl-config"
        export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
        {
          printf 'cd %q\n' "$out"
          printf 'export MPLBACKEND=%q MPLCONFIGDIR=%q PYTHONDONTWRITEBYTECODE=%q OPENBLAS_NUM_THREADS=%q OMP_NUM_THREADS=%q\n' \
            "$MPLBACKEND" "$MPLCONFIGDIR" "$PYTHONDONTWRITEBYTECODE" "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS"
          printf '%q ' "${command[@]}"
          printf '\n'
        } > command.sh || exit 1
        "${command[@]}" >process.log 2>&1
      ); then
        printf 'Finished: %s / %s\n' "$cohort" "$variant"
      else
        printf 'Failed: %s / %s. See %s/process.log\n' "$cohort" "$variant" "$out" >&2
        failed=1
      fi
    done
  done
  exit "$failed"
)
