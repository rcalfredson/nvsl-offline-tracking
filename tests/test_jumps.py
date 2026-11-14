"""Jump-screen definition and parity with the supplied analysis source."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nvsl_offline.jumps import aggregate_comparisons, jump_pairs, score_positions


def test_aggregate_weights_exposure_and_retains_a_regression():
    rows = []
    for video, frames, old_observed, new_observed, old_pairs, new_pairs in [
        ("short.avi", 10, 10, 5, 2, 0), ("long.avi", 90, 45, 90, 0, 1)]:
        for source, observed, pairs in [("original", old_observed, old_pairs), ("offline", new_observed, new_pairs)]:
            rows.append(dict(video=video, slot=0, source=source, frames=frames, observed_frames=observed,
                             raw_suspicious_pairs=pairs, analysis_suspicious_pairs=pairs))
    summary = aggregate_comparisons(rows)
    assert summary["recordings"] == 2 and summary["fly_slots"] == 2
    assert summary["original"]["missing_fraction"] == .45
    assert summary["offline"]["missing_fraction"] == .05
    assert summary["original"]["raw_suspicious_pairs"] == 2
    assert summary["offline"]["raw_suspicious_pairs"] == 1
    assert summary["slot_outcomes_by_raw_pair_count"] == dict(decreased=1, increased=1, unchanged=0)


def test_aggregate_excluded_analysis_is_unassessed_and_duplicate_exposure_fails():
    rows = [dict(video="a.avi", slot=0, source=source, frames=10, observed_frames=1,
                 raw_suspicious_pairs=0, analysis_suspicious_pairs=None) for source in ("original", "offline")]
    summary = aggregate_comparisons(rows)
    assert summary["original"]["analysis_unassessed_slots"] == 1
    assert summary["original"]["analysis_suspicious_pairs"] is None
    with pytest.raises(ValueError, match="one companion"):
        aggregate_comparisons(rows+rows)
    with pytest.raises(ValueError, match="both original and offline"):
        aggregate_comparisons(rows[:1])


def test_reverse_pair_rule_strict_thresholds_and_delayed_return():
    _, pairs = jump_pairs([0, 40, 40, 40, 0], [0]*5)
    assert [(p["first_jump"], p["second_jump"]) for p in pairs] == [(0, 3)]
    assert pairs[0]["dwell_frames"] == 3
    assert not jump_pairs([0, 30, 0], [0]*3)[1]
    assert not jump_pairs([0, 50, 10], [0]*3)[1]  # Endpoint mismatch exactly 10 fails.


def test_analysis_interpolation_is_diagnostic_only_and_can_hide_raw_jumps():
    x = np.array([np.nan, 0, 40, np.nan, 0, 0.])
    original = x.copy()
    summary, pairs = score_positions(x, np.where(np.isnan(x), np.nan, 0))
    np.testing.assert_array_equal(x, original)
    assert summary["first_observed_frame"] == 1
    assert summary["raw_long_jumps"] == 1
    assert summary["analysis_long_jumps"] == 1
    assert not pairs


def test_bad_threshold_empty_and_excess_loss_gates():
    summary, _ = score_positions([0, 50, 0, 50, 0], [0]*5)
    assert summary["analysis_suspicious_pairs"] == 3
    assert summary["analysis_would_mark_bad"]
    assert score_positions([np.nan]*100, [np.nan]*100)[0]["analysis_state"] == "empty"
    x = np.r_[0., np.full(8, np.nan), 50.]
    y = np.where(np.isnan(x), np.nan, 0)
    assert score_positions(x, y)[0]["analysis_state"] == "excess_loss"
    assert score_positions(x, y, open_loop=True)[0]["analysis_state"] == "eligible"


def test_vectorized_pair_rule_matches_reference_function():
    path = Path("reference/nvsl-analysis/src/analysis/trajectory.py")
    if not path.exists():
        pytest.skip("Read-only reference not present")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trajectory")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_suspiciousJumps")
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    class Reference:
        JMP_LEN_TH, DIST_TH, SUSP_FRAC_TH, SUSP_NUM_TH, _DEBUG = 30, 10, .03, 3, False
        def _p(self, message): pass
        def dist(self, a, b): return np.hypot(self.x[a]-self.x[b], self.y[a]-self.y[b])
    rng = np.random.default_rng(412)
    for _ in range(30):
        obj = Reference()
        obj.x = rng.choice([0., 2., 50., 52., 110., np.nan], 400)
        obj.y = rng.choice([0., 2., 50., 52.], 400)
        obj.d = np.r_[np.hypot(np.diff(obj.x), np.diff(obj.y)), 0]
        obj.d[np.isnan(obj.d)] = 0
        scope["_suspiciousJumps"](obj)
        _, pairs = jump_pairs(obj.x, obj.y)
        assert obj.susp == [edge for p in pairs for edge in (p["first_jump"], p["second_jump"])]


def test_post_interpolation_counts_match_reference_processing():
    path = Path("reference/nvsl-analysis/src/analysis/trajectory.py")
    if not path.exists():
        pytest.skip("Read-only reference not present")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trajectory")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_interpolate", "_suspiciousJumps")]
    def true_regions(mask):
        changes = np.diff(np.r_[False, mask, False].astype(int))
        return [slice(a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
    scope = {"np": np, "util": SimpleNamespace(trueRegions=true_regions)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    class Reference:
        JMP_LEN_TH, DIST_TH, SUSP_FRAC_TH, SUSP_NUM_TH, _DEBUG = 30, 10, .03, 3, False
        theta = None
        _max_proportion_lost_fms = .6
        va = SimpleNamespace(openLoop=False, on=np.array([]))
        def _p(self, message): pass
        def dist(self, a, b): return np.hypot(self.x[a]-self.x[b], self.y[a]-self.y[b])
    rng = np.random.default_rng(16)
    for _ in range(20):
        x = rng.choice([0., 2., 50., 52., 110.], 100)
        y = rng.choice([0., 2., 50.], 100)
        missing = rng.random(100) < .25
        missing[:3] = True
        missing[3] = False
        missing[-3:] = True
        x[missing] = y[missing] = np.nan
        obj = Reference()
        obj.x, obj.y = x.copy(), y.copy()
        scope["_interpolate"](obj)
        obj.d = np.r_[np.linalg.norm([np.diff(obj.x), np.diff(obj.y)], axis=0), 0]
        obj.d[np.isnan(obj.d)] = 0
        scope["_suspiciousJumps"](obj)
        summary, _ = score_positions(x, y)
        assert summary["analysis_suspicious_pairs"] == len(obj.susp)//2
        assert summary["analysis_long_jumps"] == int((obj.d > 30).sum())
        assert summary["analysis_would_mark_bad"] == obj._bad
