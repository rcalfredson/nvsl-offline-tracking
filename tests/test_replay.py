"""Comparison alignment, navigation, rendering and real headless video export."""

from dataclasses import asdict
import json
import pickle

import cv2
import numpy as np
import pytest

from test_baseline import recording
from nvsl_offline.recording import Calibration, FIELDS, Recording
from nvsl_offline.replay import (Comparison, Controls, FrameReader,
                                 compare, export_clip, parse_position)
from nvsl_offline.video import frames, probe


@pytest.fixture
def pair(tmp_path):
    video = recording(tmp_path)
    rec = Recording.load(video)
    payload = {key: value.copy() for key, value in rec.raw.items()}
    status = np.full(rec.shape, 3, np.uint8)
    status[0, 0] = 0
    for index in [2, 3, 4, 6, 7]:
        for key, value in zip(FIELDS, (28+7*index, 50, 8, 16, 20)):
            payload[key][0, index] = value
        status[0, index] = 2 if index == 4 else 1
    cal = Calibration.from_recording(rec, 240, 240, layout="large2", margin=3)
    payload["tracking_metadata"] = dict(source_recording=rec.sources(), configuration=dict(
        calibration=asdict(cal), detector=dict(mode="normalized")))
    payload["tracking_quality"] = dict(status=status, status_codes=dict(observed=0, recovered=1, interpolated=2, missing=3),
        reason=(status == 3).astype(np.uint8), reason_codes=dict(accepted=0, no_candidate=1))
    path = tmp_path/"selected.offline.trx"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return Comparison.load(path), path, payload


def test_clock_parser_and_exact_frame_positions():
    assert parse_position("5500", 7.5) == 5500
    assert parse_position("01:00", 7.5) == 450
    assert parse_position("00:01:00.2", 7.5) == 452
    for text in ["-1", "", "nan", "0:60", "1:-2", "0:inf", "1.5:00"]:
        with pytest.raises(ValueError):
            parse_position(text, 7.5)


def test_source_and_frame_alignment_are_enforced(pair):
    comp, path, payload = pair
    assert comp.variant_label == "normalized"
    assert comp.nframes == 9
    payload["ts"][2] += .1
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="identical fly/frame shapes and timestamps"):
        Comparison.load(path)


def test_recovery_navigation_excludes_interpolation_and_keeps_slot_boundaries(pair):
    comp, _, _ = pair
    np.testing.assert_array_equal(comp.starts(0, "original"), [1])
    np.testing.assert_array_equal(comp.starts(None, "original"), [0, 1])
    np.testing.assert_array_equal(comp.starts(0, "new"), [2, 6])
    assert comp.provenance(0, 4) == "interpolated"
    controls = Controls(0, 9, 0, 0)
    controls.key("n", comp)
    assert controls.index == 1 and controls.paused
    controls.key("r", comp)
    assert controls.index == 2
    controls.key("r", comp)
    assert controls.index == 6
    controls.key("b", comp)
    assert controls.index == 2
    controls.key("p", comp)
    assert controls.index == 1


def test_render_uses_same_pixels_and_never_fills_original_missing_values(pair):
    comp, _, _ = pair
    reader = FrameReader(comp.original.video)
    try:
        frame = reader.read(2)
        raw = comp.render(frame, 2, fly=0, scale=2, ellipses=False, centers=False)
        shown = comp.render(frame, 2, fly=0, scale=2)
        x0, y0, x1, y1 = comp.crop(0)
        height = (y1-y0)*2
        width = (x1-x0)*2
        # Common banner (36) and panel heading (48) precede identical image pixels.
        left = raw[84:84+height, :width]
        right = raw[84:84+height, width:]
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(shown[84:84+height, :width], left)
        assert np.any(shown[84:84+height, width:] != right)
        assert np.isnan(comp.original.arrays["x"][0, 2])
    finally:
        reader.close()


def test_random_and_backward_seek_match_sequential_decode(pair):
    comp, _, _ = pair
    reference = dict(frames(comp.original.video))
    reader = FrameReader(comp.original.video)
    try:
        for index in [7, 2, 3, 0, 8, 8]:
            np.testing.assert_array_equal(reader.read(index), reference[index])
    finally:
        reader.close()


def test_controls_seek_zoom_and_overlay_toggles(pair):
    comp, _, _ = pair
    controls = Controls(0, 9, 2, None)
    for key in "7g":
        controls.key(key, comp)
    assert controls.index == 7
    controls.key(",", comp)
    assert controls.index == 6
    controls.key("f", comp)
    assert controls.fly == 0
    controls.key("a", comp)
    assert controls.fly is None
    controls.key("e", comp)
    controls.key("c", comp)
    assert not controls.ellipses and not controls.centers
    for key in "99g":
        controls.key(key, comp)
    assert controls.index == 6 and controls.notice
    assert controls.key("q", comp) == "quit"


def test_snapshot_and_export_preserve_all_interval_frames(pair, tmp_path):
    comp, path, _ = pair
    snapshot = tmp_path/"review.png"
    compare(path, start="2", fly=0, scale=2, snapshot=snapshot)
    assert cv2.imread(str(snapshot)).shape[1] > 300
    assert json.loads(snapshot.with_suffix(".json").read_text())["frame"] == 2
    with pytest.raises(ValueError, match="already exists"):
        compare(path, start="2", snapshot=snapshot)
    output = tmp_path/"review.mp4"
    result = export_clip(comp, output, start=1, stop=5, fly=0, scale=2, speed=.5)
    assert result["exported_frames"] == 4
    assert probe(output).fps == pytest.approx(3.75)
    assert len(list(frames(output))) == 4
    with pytest.raises(ValueError, match="new .mp4"):
        export_clip(comp, output, start=1, stop=5)


def test_quality_missingness_cannot_contradict_measurements(pair):
    _, path, payload = pair
    payload["tracking_quality"]["status"][0, 2] = 3
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="missing masks disagree"):
        Comparison.load(path)


def test_player_loop_navigation_and_snapshot_without_a_desktop(pair, tmp_path, monkeypatch):
    import nvsl_offline.replay as replay
    comp, _, _ = pair
    monkeypatch.setenv("DISPLAY", ":mock")
    monkeypatch.setattr(replay.cv2, "getBuildInformation", lambda: "GUI: MOCK\n")
    monkeypatch.setattr(replay.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(replay.cv2, "createTrackbar", lambda *args: None)
    monkeypatch.setattr(replay.cv2, "setTrackbarPos", lambda *args: None)
    displayed = []
    monkeypatch.setattr(replay.cv2, "imshow", lambda title, image: displayed.append(image.shape))
    monkeypatch.setattr(replay.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(replay.cv2, "destroyAllWindows", lambda: None)
    keys = iter(map(ord, [".", "r", "s", "q"]))
    monkeypatch.setattr(replay.cv2, "waitKey", lambda delay: next(keys))
    replay.play(comp, start=0, stop=9, fly=0, snapshot_dir=tmp_path/"snapshots")
    metadata = list((tmp_path/"snapshots").glob("*.json"))
    assert len(metadata) == 1
    assert json.loads(metadata[0].read_text())["frame"] == 2
    assert len(displayed) == 4
