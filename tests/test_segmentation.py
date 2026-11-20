"""Refined measurements must follow body pixels, not connected thin artifacts."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nvsl_offline.background import Background
from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.geometry import weighted_body_ellipse
from nvsl_offline.segmentation import body_core_regions


def test_thin_attachment_does_not_reappear_after_bounded_growth():
    mask = np.zeros((100, 200), np.uint8)
    cv2.ellipse(mask, (120, 50), (5, 12), 0, 0, 360, 1, -1)
    cv2.line(mask, (20, 50), (120, 50), 1, 1)
    before = mask.copy()
    regions = body_core_regions(mask)
    assert len(regions) == 1
    region = regions[0]
    assert region.origin[0] > 105
    assert not region.truncated
    x, y = region.origin
    assert np.all(region.mask <= mask[y:y+region.mask.shape[0], x:x+region.mask.shape[1]])
    ellipse = weighted_body_ellipse(region.mask, region.mask*20, region.origin)
    np.testing.assert_allclose(ellipse[0], (120, 50), atol=1)
    assert ellipse[1][1] < 30
    np.testing.assert_array_equal(mask, before)


def test_competing_cores_never_claim_the_same_pixels():
    mask = np.zeros((60, 90), np.uint8)
    for center in [(30, 30), (50, 30)]:
        cv2.circle(mask, center, 5, 1, -1)
    cv2.line(mask, (30, 30), (50, 30), 1, 1)
    regions = body_core_regions(mask, growth=10)
    assert len(regions) == 2
    occupancy = np.zeros_like(mask, dtype=int)
    for region in regions:
        x, y = region.origin
        occupancy[y:y+region.mask.shape[0], x:x+region.mask.shape[1]] += region.mask
    assert occupancy.max() == 1


def test_erosion_cannot_disguise_roi_truncation():
    mask = np.zeros((60, 80), np.uint8)
    cv2.ellipse(mask, (40, 30), (5, 10), 0, 0, 360, 1, -1)
    cv2.line(mask, (0, 30), (40, 30), 1, 1)
    regions = body_core_regions(mask)
    assert len(regions) == 1 and regions[0].truncated
    assert regions[0].origin[0] > 0


def test_mask_weighted_center_and_orientation_use_saved_image_coordinates():
    mask = np.zeros((50, 50), np.uint8)
    cv2.ellipse(mask, (20, 25), (4, 10), 0, 0, 360, 1, -1)
    center, (w, h), theta = weighted_body_ellipse(mask, mask*20, (100, 200))
    np.testing.assert_allclose(center, (120, 225), atol=.1)
    assert w < h and min(theta, 180-theta) < 1


def test_detector_refines_center_before_candidate_selection():
    bg = Background.fit(np.full((3, 240, 240), 150, np.uint8))
    cal = SimpleNamespace(roi=lambda slot: [(0,0,120,120),(120,0,240,120),
                                           (0,120,120,240),(120,120,240,240)][slot])
    frame = np.full((240, 240, 3), 150, np.uint8)
    cv2.ellipse(frame, (80, 60), (5, 12), 0, 0, 360, (125,)*3, -1)
    cv2.line(frame, (20, 60), (80, 60), (125,)*3, 1)
    baseline = detect(frame, bg, cal, DetectionConfig(mode='normalized'))[0][0]
    refined = detect(frame, bg, cal, DetectionConfig(mode='normalized',segmentation='body_core'))[0][0]
    assert baseline.reason == refined.reason == 'accepted'
    assert abs(baseline.ellipse[0][0]-80) > 10
    np.testing.assert_allclose(refined.ellipse[0], (80, 60), atol=1)
    assert refined.shape_method == 'core_weighted_moments'


def test_no_core_is_missing_and_not_a_fallback_to_an_artifact():
    bg = Background.fit(np.full((3, 240, 240), 150, np.uint8))
    cal = SimpleNamespace(roi=lambda slot: [(0,0,120,120),(120,0,240,120),
                                           (0,120,120,240),(120,120,240,240)][slot])
    frame = np.full((240,240,3),150,np.uint8)
    cv2.line(frame, (20,60), (100,60), (100,)*3, 1)
    found = detect(frame,bg,cal,DetectionConfig(mode='normalized',segmentation='body_core'))[0][0]
    assert found.reason == 'no_core' and found.ellipse is None


def test_core_configuration_is_explicit_and_validated():
    from nvsl_offline.cli import parser
    assert DetectionConfig().segmentation == 'components'
    assert parser().parse_args(['track','audit','--output','new','--segmentation','body_core']).core_radius == 2
    for kwargs in [dict(core_radius=0),dict(core_radius=float('nan')),dict(core_growth=-1),
                   dict(core_growth=2.5),dict(segmentation='unknown'),
                   dict(segmentation='body_core',shape_fit='box'),
                   dict(min_concentration=.9), dict(ambiguity_darkness_ratio=2),
                   dict(segmentation='body_core', min_concentration=1.1),
                   dict(segmentation='body_core', ambiguity_darkness_ratio=float('inf'))]:
        with pytest.raises(ValueError):
            DetectionConfig(**kwargs).validate()


def test_concentration_and_darkness_resolve_different_competitors():
    bg = Background.fit(np.full((3, 240, 240), 150, np.uint8))
    cal = SimpleNamespace(roi=lambda slot: [(0, 0, 120, 120), (120, 0, 240, 120),
                                          (0, 120, 120, 240), (120, 120, 240, 240)][slot])
    frame = np.full((240, 240, 3), 150, np.uint8)
    cv2.rectangle(frame, (10, 10), (100, 15), (125,)*3, -1)
    cv2.ellipse(frame, (35, 40), (5, 8), 0, 0, 360, (70,)*3, -1)
    cv2.ellipse(frame, (75, 70), (5, 8), 0, 0, 360, (120,)*3, -1)

    def result(**options):
        return detect(frame, bg, cal, DetectionConfig(
            mode="normalized", segmentation="body_core", **options))[0][0]

    assert result().ellipse[0][1] < 20
    compact = result(min_concentration=.9)
    assert compact.reason == "ambiguous"
    assert compact.rejected_counts["low_concentration"] == 1
    resolved = result(min_concentration=.9, ambiguity_darkness_ratio=2)
    assert resolved.reason == "accepted"
    assert resolved.darkness_ratio == pytest.approx(80/30)
    np.testing.assert_allclose(resolved.ellipse[0], (35, 40), atol=.2)
    assert result(min_concentration=.9, ambiguity_darkness_ratio=3).reason == "ambiguous"

    # A single faint body stays eligible even with the darkness tie-break enabled.
    frame[:]=150
    cv2.ellipse(frame, (75, 70), (5, 8), 0, 0, 360, (120,)*3, -1)
    faint = result(min_concentration=.9, ambiguity_darkness_ratio=2)
    assert faint.reason == "accepted" and faint.darkness_ratio is None
    np.testing.assert_allclose(faint.ellipse[0], (75, 70), atol=.2)
