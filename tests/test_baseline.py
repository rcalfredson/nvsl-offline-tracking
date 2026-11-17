"""Behavioral/contract tests using independently constructed synthetic recordings."""

import pickle

import cv2
import numpy as np
import pytest

from nvsl_offline.background import Background, BackgroundConfig
from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.pipeline import audit, inspect, load_run, track
from nvsl_offline.recording import Calibration, Recording, round_acquisition
from nvsl_offline.video import decode_samples


def recording(tmp_path, nframes=9):
    video = tmp_path/"experiment.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 7.5, (240, 240))
    assert writer.isOpened()
    for frame_id in range(nframes):
        image = np.full((240, 240, 3), 150, np.uint8)
        # Different body locations let the global estimator observe the substrate.
        cv2.ellipse(image, (28+frame_id*7, 50), (4, 8), 20, 0, 360, (50, 50, 50), -1)
        writer.write(image)
    writer.release()
    geometry = {k: np.full((4, nframes), np.nan, np.float32) for k in ("x", "y", "w", "h", "theta")}
    for key, value in zip(("x", "y", "w", "h", "theta"), (28, 50, 8, 16, 20)):
        geometry[key][0, 0] = value
    # Preserve duplicate and nonmonotonic acquisition times rather than resampling.
    geometry["ts"] = np.arange(nframes, dtype=np.float32)/7.5
    geometry["ts"][2] = geometry["ts"][1]
    geometry["ts"][4] = geometry["ts"][3]-.01
    data = dict(protocol=dict(ct="large", tm=dict(x=5, y=5, fctr=.3),
                             frameNums=[dict(startPre=[0], startTrain=[2], startPost=[nframes]), {}]))
    for suffix, payload in [(".trx", geometry), (".data", data)]:
        with video.with_suffix(suffix).open("wb") as stream:
            pickle.dump(payload, stream, protocol=2)
    return video


@pytest.fixture
def scene(tmp_path):
    rec = Recording.load(recording(tmp_path))
    cal = Calibration.from_recording(rec, 240, 240, layout="large2", margin=3)
    samples = np.full((9, 240, 240), 150, np.uint8)
    bg = Background.fit(samples)
    return rec, cal, bg


def test_background_excludes_moving_dark_body_and_bright_outlier():
    samples = np.full((9, 30, 30), 100, np.uint8)
    samples[0, 4:12, 4:12] = 10
    samples[1, 4:12, 4:12] = 255
    bg = Background.fit(samples, BackgroundConfig(tile_rows=7))
    np.testing.assert_allclose(bg.estimate, 100)
    np.testing.assert_allclose(bg.median, 100)
    np.testing.assert_allclose(bg.noise, 3)


def test_normalized_detection_recovers_faint_image_supported_body(scene):
    rec, cal, bg = scene
    image = np.full((240, 240, 3), 150, np.uint8)
    cv2.ellipse(image, (50, 50), (4, 8), 20, 0, 360, (136, 136, 136), -1)
    absolute, _, _ = detect(image, bg, cal, DetectionConfig())
    normalized, _, _ = detect(image, bg, cal, DetectionConfig(mode="normalized"))
    assert absolute[0].ellipse is None
    assert normalized[0].reason == "accepted"
    np.testing.assert_allclose(normalized[0].ellipse[0], (50, 50), atol=.6)
    assert all(r.ellipse is None for r in normalized[1:])


def test_competing_objects_leave_nan_opportunity_instead_of_forcing_choice(scene):
    _, cal, bg = scene
    image = np.full((240, 240, 3), 150, np.uint8)
    for xy in [(35, 40), (70, 70)]:
        cv2.ellipse(image, xy, (4, 8), 20, 0, 360, (50, 50, 50), -1)
    result, _, _ = detect(image, bg, cal, DetectionConfig())
    assert result[0].reason == "ambiguous" and result[0].ellipse is None
    control, _, _ = detect(image, bg, cal, DetectionConfig(reject_ambiguous=False))
    assert control[0].reason == "accepted"


def test_no_motion_is_required_for_a_body_against_a_clean_background(scene):
    _, cal, bg = scene
    image = np.full((240, 240, 3), 150, np.uint8)
    cv2.ellipse(image, (50, 50), (4, 8), 20, 0, 360, (50, 50, 50), -1)
    assert all(detect(image, bg, cal, DetectionConfig())[0][0].reason == "accepted" for _ in range(3))


@pytest.mark.parametrize("angle", [0, 25, 90, 135])
def test_box2d_axis_swap_preserves_drawn_ellipse(angle):
    first = np.zeros((120, 120), np.uint8)
    second = first.copy()
    cv2.ellipse(first, ((60., 60.), (12., 28.), float(angle)), 255, -1)
    cv2.ellipse(second, ((60., 60.), (28., 12.), float(angle+90)), 255, -1)
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    assert intersection/union > .98


def test_legacy_geometry_requires_consistent_missing_masks(tmp_path):
    video = recording(tmp_path)
    path = video.with_suffix(".trx")
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    raw["x"][0, 1] = 30
    with path.open("wb") as stream:
        pickle.dump(raw, stream)
    with pytest.raises(ValueError, match="inconsistent geometry"):
        Recording.load(video)


def test_calibration_slots_and_acquisition_rounding(scene):
    rec, cal, _ = scene
    assert [round_acquisition(v) for v in [-1.5, -.5, .5, 1.5]] == [-2, -1, 1, 2]
    assert cal.floor(0)[0] < cal.floor(1)[0]
    assert cal.floor(0)[1] < cal.floor(2)[1]
    assert cal.roi(0)[2] <= cal.roi(1)[0]
    assert cal.roi(0)[3] <= cal.roi(2)[1]
    with pytest.raises(ValueError, match="does not fit"):
        Calibration.from_recording(rec, 120, 120, layout="large2")


@pytest.mark.parametrize("illumination", ["none", "global_offset"])
@pytest.mark.parametrize("segmentation", ["components", "body_core"])
def test_audit_review_companion_preserve_sources_and_frame_contract(tmp_path, illumination, segmentation):
    video = recording(tmp_path)
    original = Recording.load(video)
    sources = original.sources(hash_video=True)
    run = tmp_path/"audit"
    manifest = audit(video, run, layout="large2", margin=3, samples=9)
    assert manifest["decoded_frames"] == original.shape[1]
    assert manifest["recording"]["events"]["endpoint_events"] == 1
    assert manifest["recording"]["timestamps"]["zero_steps"] == 1
    assert manifest["recording"]["timestamps"]["negative_steps"] == 1
    review = tmp_path/"review"
    inspect(run, review, indices=[1, 5], configs=[DetectionConfig()])
    assert (review/"0000001_absolute.jpg").is_file()
    assert len((review/"annotations.csv").read_text().splitlines()) == 9
    output = tmp_path/"experiment.offline.trx"
    track(run, output, config=DetectionConfig(illumination=illumination,segmentation=segmentation), exclude_slots=[3])
    with output.open("rb") as stream:
        offline = pickle.load(stream)
    assert offline["x"].shape == original.shape
    assert offline["tracking_metadata"]["configuration"]["detector"]["illumination"] == illumination
    assert offline["tracking_metadata"]["configuration"]["detector"]["segmentation"] == segmentation
    assert offline["tracking_quality"]["illumination_offset"].shape == (original.shape[1],)
    assert np.isfinite(offline["tracking_quality"]["illumination_offset"]).all()
    np.testing.assert_array_equal(offline["ts"], original.raw["ts"])
    assert np.isnan(offline["x"][3]).all()
    assert not np.any(offline["tracking_quality"]["status"] == 2)
    assert np.isfinite(offline["x"][0]).any()
    assert offline["tracking_quality"]["status"][0, 0] == 0
    for key in ("x", "y", "w", "h", "theta"):
        assert np.array_equal(np.isnan(offline[key]), np.isnan(offline["x"]))
    assert original.sources(hash_video=True) == sources
    with pytest.raises(ValueError, match="destination must be new"):
        track(run, output)
    with pytest.raises(ValueError, match="destination must be new"):
        track(run, original.trx_path)
    with pytest.raises(FileExistsError):
        audit(video, run, layout="large2", samples=9)


def test_audit_rejects_header_alignment_mismatch(tmp_path):
    video = recording(tmp_path)
    with pytest.raises(ValueError, match="Video header"):
        decode_samples(video, expected_frames=100, count=3, sample_path=tmp_path/"samples.npy")
    assert not (tmp_path/"samples.npy").exists()


def test_review_rejects_changed_acquisition_source(tmp_path):
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    with video.with_suffix(".data").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        load_run(run)


def test_excess_event_indices_are_rejected_before_processing(tmp_path):
    video = recording(tmp_path)
    path = video.with_suffix(".data")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    data["protocol"]["frameNums"][0]["startTrain"] = [10]
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    with pytest.raises(ValueError, match="invalid saved-frame indices"):
        Recording.load(video)


def test_changed_background_is_rejected(tmp_path):
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    with (run/"background.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Background checksum"):
        load_run(run)


def test_failed_companion_write_leaves_no_output(tmp_path, monkeypatch):
    import nvsl_offline.pipeline as pipeline
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(pipeline.pickle, "dump", fail)
    output = tmp_path/"failed.offline.trx"
    with pytest.raises(OSError, match="simulated disk failure"):
        track(run, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".offline-*"))


def test_crop_truncation_rejected_without_using_motion_prediction(scene):
    _, cal, bg = scene
    image = np.full((240, 240, 3), 150, np.uint8)
    x0, y0, _, _ = cal.roi(0)
    cv2.ellipse(image, (x0+2, y0+25), (8, 10), 20, 0, 360, (50, 50, 50), -1)
    result, _, _ = detect(image, bg, cal, DetectionConfig())
    assert result[0].reason == "truncated"
    assert result[0].ellipse is None


def test_legacy_main_detector_unpickles_without_reference_import(tmp_path, monkeypatch):
    import __main__
    from nvsl_offline.recording import LegacyFlyDetector, load_pickle
    cls = type("FlyDetector", (), {"__module__": "__main__"})
    monkeypatch.setattr(__main__, "FlyDetector", cls, raising=False)
    detector = cls()
    detector.nchm = 4
    path = tmp_path/"legacy.data"
    with path.open("wb") as stream:
        pickle.dump(dict(flyDetector=detector), stream, protocol=2)
    monkeypatch.delattr(__main__, "FlyDetector")
    loaded = load_pickle(path)
    assert isinstance(loaded["flyDetector"], LegacyFlyDetector)
    assert loaded["flyDetector"].nchm == 4


def test_truncated_decode_cannot_publish_companion(tmp_path, monkeypatch):
    import nvsl_offline.pipeline as pipeline
    video = recording(tmp_path)
    run = tmp_path/"audit"
    audit(video, run, layout="large2", samples=9)
    original_frames = pipeline.frames
    def truncated_frames(path):
        for index, frame in original_frames(path):
            if index == 5:
                return
            yield index, frame
    monkeypatch.setattr(pipeline, "frames", truncated_frames)
    output = tmp_path/"truncated.offline.trx"
    with pytest.raises(ValueError, match="Video ended at"):
        track(run, output)
    assert not output.exists()
