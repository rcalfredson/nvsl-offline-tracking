"""Lighting changes must be removed without explaining away foreground bodies."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nvsl_offline.background import Background
from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.illumination import global_offset


ROIS = [(0, 0, 128, 128), (128, 0, 256, 128),
        (0, 128, 128, 256), (128, 128, 256, 256)]


@pytest.mark.parametrize("offset", [-18, 0, 12])
def test_lighting_shift_preserves_faint_body_and_its_image_position(offset):
    bg = Background.fit(np.full((3, 256, 256), 150, np.uint8))
    cal = SimpleNamespace(roi=lambda slot: ROIS[slot])
    frame = np.full((256, 256, 3), 150+offset, np.uint8)
    # An elongated body also represents a fast-motion detection; no speed gate.
    cv2.ellipse(frame, (60, 60), (4, 15), 35, 0, 360, (136+offset,)*3, -1)
    original = frame.copy()
    results, residual, mask = detect(frame, bg, cal, DetectionConfig(
        mode="normalized", illumination="global_offset"))
    assert results[0].reason == "accepted"
    np.testing.assert_allclose(results[0].ellipse[0], (60, 60), atol=.7)
    assert all(r.ellipse is None for r in results[1:])
    assert results[0].illumination_offset == offset
    assert residual[200, 200] == 0 and mask[200, 200] == 0
    np.testing.assert_array_equal(frame, original)


def test_localized_dark_object_does_not_drive_global_brightness():
    residual = np.full((256, 256), 9., np.float32)
    residual[5:60, 5:60] += 70
    before = residual.copy()
    estimate = global_offset(residual, ROIS)
    assert estimate.offset == -9
    assert estimate.tile_mad == 0
    assert estimate.tiles == 64
    np.testing.assert_array_equal(residual, before)


def test_correction_prevents_selection_of_low_noise_wall_fragment():
    bg = Background.fit(np.full((3, 256, 256), 150, np.uint8))
    bg.noise[:] = 6
    cv2.ellipse(bg.noise, (110, 60), (3, 40), 0, 0, 360, 3, -1)
    cal = SimpleNamespace(roi=lambda slot: ROIS[slot])
    frame = np.full((256, 256, 3), 132, np.uint8)
    cv2.ellipse(frame, (50, 50), (4, 9), 20, 0, 360, (104,)*3, -1)
    baseline, _, _ = detect(frame, bg, cal, DetectionConfig(mode='normalized'))
    corrected, _, _ = detect(frame, bg, cal, DetectionConfig(
        mode='normalized', illumination='global_offset'))
    assert baseline[0].reason == corrected[0].reason == 'accepted'
    np.testing.assert_allclose(baseline[0].ellipse[0], (110, 60), atol=.6)
    np.testing.assert_allclose(corrected[0].ellipse[0], (50, 50), atol=.6)


def test_unequal_lighting_is_visible_as_tile_disagreement():
    residual = np.tile(np.arange(256, dtype=np.float32), (256, 1))
    assert global_offset(residual, ROIS).tile_mad > 0


def test_illumination_setting_is_explicit_and_validated():
    from nvsl_offline.cli import parser
    assert DetectionConfig().illumination == "none"
    for command in ['track', 'inspect']:
        argv = [command, 'audit', '--output', 'new']
        if command == 'inspect':
            argv += ['--frames', '0']
        assert parser().parse_args(argv).illumination == 'none'
        assert parser().parse_args([*argv, '--illumination', 'global_offset']).illumination == 'global_offset'
    with pytest.raises(ValueError, match="illumination"):
        DetectionConfig(illumination="unknown").validate()
