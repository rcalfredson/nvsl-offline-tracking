from dataclasses import replace

import cv2
import numpy as np
import pytest

from test_baseline import recording
from nvsl_offline.background import Background
from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.geometry import canonical_ellipse, weighted_moment_ellipse
from nvsl_offline.pipeline import audit, track
from nvsl_offline.recording import Calibration, Recording, load_pickle


@pytest.mark.parametrize("angle", [0., 25., 89., 135., 179.])
def test_canonical_axes_preserve_geometry_and_axial_angle(angle):
    ellipse = ((50., 50.), (28., 12.), angle)
    center, (w, h), theta = canonical_ellipse(ellipse)
    assert w <= h and 0 <= theta < 180
    left, right = [np.zeros((100, 100), np.uint8) for _ in range(2)]
    cv2.ellipse(left, ellipse, 255, -1)
    cv2.ellipse(right, (center, (w, h), theta), 255, -1)
    assert np.count_nonzero(left & right)/np.count_nonzero(left | right) > .98


def test_uniform_ellipse_moments_recover_known_diameters_and_angle():
    # A uniform filled ellipse has covariance a²/4 and b²/4.
    mask = np.zeros((160, 160), np.uint8)
    cv2.ellipse(mask, (80, 80), (9, 24), 35, 0, 360, 255, -1)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    result = weighted_moment_ellipse(contours[0], mask.astype(float), mask, (80., 80.))
    assert result[0] == (80., 80.)
    np.testing.assert_allclose(result[1], (18, 48), atol=1.5)
    assert abs((result[2]-35+90)%180-90) < 2


def test_weaker_periphery_is_downweighted_without_changing_detection(tmp_path):
    rec = Recording.load(recording(tmp_path))
    cal = Calibration.from_recording(rec, 240, 240, layout="large2", margin=3)
    bg = Background.fit(np.full((9, 240, 240), 150, np.uint8))
    frame = np.full((240, 240, 3), 150, np.uint8)
    cv2.ellipse(frame, (50, 50), (10, 20), 25, 0, 360, (136, 136, 136), -1)
    cv2.ellipse(frame, (50, 50), (6, 14), 25, 0, 360, (70, 70, 70), -1)
    config = DetectionConfig(mode="normalized", shape_fit="box")
    baseline, _, mask1 = detect(frame, bg, cal, config)
    refined, _, mask2 = detect(frame, bg, cal, replace(config, shape_fit="weighted_moments"))
    np.testing.assert_array_equal(mask1, mask2)
    for old, new in zip(baseline, refined):
        assert old.reason == new.reason
        assert len(old.candidates) == len(new.candidates)
        assert (old.ellipse is None) == (new.ellipse is None)
        if old.ellipse is not None:
            assert old.ellipse[0] == new.ellipse[0]
    assert refined[0].shape_method == "weighted_moments"
    assert np.prod(refined[0].ellipse[1]) < np.prod(baseline[0].ellipse[1])*.8


def test_numerically_unresolved_moments_do_not_invent_geometry():
    contour = np.array([[[2, 2]], [[2, 5]], [[5, 5]], [[5, 2]]], np.int32)
    mask = np.full((10, 10), 255, np.uint8)
    assert weighted_moment_ellipse(contour, np.zeros((10, 10)), mask, (3., 3.)) is None


def test_full_companion_centers_status_and_timestamp_are_exactly_preserved(tmp_path):
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    old, new = tmp_path/"box.trx", tmp_path/"moments.trx"
    config = DetectionConfig(mode="normalized", shape_fit="box")
    track(run, old, config=config)
    track(run, new, config=replace(config, shape_fit="weighted_moments"))
    a, b = load_pickle(old), load_pickle(new)
    for key in ("x", "y", "ts"):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("status", "reason", "candidate_count", "runner_up_area_ratio"):
        np.testing.assert_array_equal(a["tracking_quality"][key], b["tracking_quality"][key])
    for key in ("w", "h", "theta"):
        np.testing.assert_array_equal(np.isnan(a[key]), np.isnan(b[key]))
    accepted = np.isfinite(b["x"])
    assert np.all(b["tracking_quality"]["shape_method"][accepted] == 2)


def test_weighted_geometry_is_default_and_box_remains_explicit(tmp_path):
    from nvsl_offline.cli import parser
    assert DetectionConfig().shape_fit == "weighted_moments"
    for argv in (["track", "run", "--output", "new.trx"],
                 ["inspect", "run", "--output", "review", "--frames", "2"]):
        assert parser().parse_args(argv).shape_fit == "weighted_moments"
        assert parser().parse_args([*argv, "--shape-fit", "box"]).shape_fit == "box"
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    default, explicit = tmp_path/"default.trx", tmp_path/"explicit.trx"
    track(run, default)
    track(run, explicit, config=DetectionConfig(shape_fit="weighted_moments"))
    a, b = load_pickle(default), load_pickle(explicit)
    for key in ("x", "y", "w", "h", "theta", "ts"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["tracking_metadata"]["configuration"]["detector"]["shape_fit"] == "weighted_moments"
