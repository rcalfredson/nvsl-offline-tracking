Offline-only suspicious-pair review commands, 14 Nov 2026.

Uses the existing original/offline comparison player, with zero-based saved-frame indices and physical fly slots. Original is yellow; offline is green. These are review candidates, not newly confirmed tracking failures.

Selection: union of raw and analysis-style offline pairs absent from the original at exactly the same two jump indices in the same mode. Nearby shifted events may describe the same underlying episode; absence of an exact match is not proof of a distinct new error. Includes the previously inspected May rim case and the analysis-only August 2 cases.

A jump index i denotes the transition from frame i to i+1. Analysis-only pairs arise from diagnostic missing-position preprocessing; the player displays raw measurements and therefore may show missing ellipses at those endpoints. No interpolation was written to the companion.

Run commands from the repository root. Players open paused. Space plays/pauses; comma/period step frames. Type a frame number then Enter or g to seek. q closes the player. No --end restriction is used, so surrounding frames remain accessible.

**c51__2025-05-30__19-22-58, slot 1**

Start at frame 34. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 49 | 50 | yes | yes | no |
| 63 | 64 | yes | yes | no |
| 195 | 199 | yes | yes | no |
| 211 | 212 | yes | yes | no |
| 223 | 224 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/c51-baseline/c51.normalized.moments.offline.trx --fly 1 --start 34 --speed 0.5
```

**c61__2025-07-17__12-27-30, slot 3**

Start at frame 144. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 159 | 160 | yes | yes | no |
| 512 | 513 | yes | yes | no |
| 653 | 668 | yes | yes | no |
| 668 | 679 | yes | yes | no |
| 786 | 787 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/01-c61__2025-07-17__12-27-30/c61__2025-07-17__12-27-30.normalized.offline.trx --fly 3 --start 144 --speed 0.5
```

**c42__2025-07-21__20-33-41, slot 0**

Start at frame 602. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 617 | 618 | yes | yes | no |
| 618 | 628 | yes | — | no |
| 919 | 920 | — | yes | yes |
| 947 | 948 | yes | yes | no |
| 1001 | 1002 | yes | yes | no |
| 1004 | 1005 | yes | yes | no |
| 1005 | 1006 | yes | yes | no |
| 1006 | 1007 | yes | yes | no |
| 1046 | 1052 | — | yes | yes |
| 1057 | 1058 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/03-c42__2025-07-21__20-33-41/c42__2025-07-21__20-33-41.normalized.offline.trx --fly 0 --start 602 --speed 0.5
```

**c62__2025-07-31__20-08-27, slot 0**

Start at frame 627. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 642 | 643 | yes | yes | no |
| 644 | 645 | yes | yes | no |
| 678 | 749 | yes | — | no |
| 730 | 731 | — | yes | yes |
| 816 | 817 | yes | yes | no |
| 892 | 893 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/08-c62__2025-07-31__20-08-27/c62__2025-07-31__20-08-27.normalized.offline.trx --fly 0 --start 627 --speed 0.5
```

**c62__2025-07-31__20-08-27, slot 2**

Start at frame 224. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 239 | 240 | yes | yes | no |
| 330 | 331 | yes | yes | no |
| 373 | 374 | yes | yes | no |
| 448 | 457 | yes | yes | no |
| 459 | 469 | yes | yes | no |
| 504 | 505 | yes | yes | no |
| 507 | 508 | — | yes | yes |
| 509 | 510 | yes | yes | no |
| 510 | 511 | yes | yes | no |
| 511 | 512 | yes | yes | no |
| 513 | 514 | yes | yes | no |
| 514 | 519 | yes | yes | no |
| 519 | 520 | yes | yes | no |
| 520 | 553 | yes | yes | no |
| 555 | 556 | yes | yes | no |
| 596 | 597 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/08-c62__2025-07-31__20-08-27/c62__2025-07-31__20-08-27.normalized.offline.trx --fly 2 --start 224 --speed 0.5
```

**c52__2025-08-02__22-03-53, slot 0**

Start at frame 74615. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 74630 | 74631 | — | yes | yes |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/07-c52__2025-08-02__22-03-53/c52__2025-08-02__22-03-53.normalized.offline.trx --fly 0 --start 74615 --speed 0.5
```

**c52__2025-08-02__22-03-53, slot 1**

Start at frame 353. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 368 | 369 | — | yes | yes |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/07-c52__2025-08-02__22-03-53/c52__2025-08-02__22-03-53.normalized.offline.trx --fly 1 --start 353 --speed 0.5
```

**c31__2025-08-04__14-48-45, slot 2**

Start at frame 60534. Offline-only pair indices:

| First jump | Second jump | Raw | Analysis-style | Synthetic analysis endpoint |
|---:|---:|---|---|---|
| 60549 | 60550 | yes | yes | no |

```bash
PYTHONPATH=src /home/tracking/miniconda3/envs/analysis3.13/bin/python -m nvsl_offline compare runs/jump-batch-2/05-c31__2025-08-04__14-48-45/c31__2025-08-04__14-48-45.normalized.offline.trx --fly 2 --start 60534 --speed 0.5
```

