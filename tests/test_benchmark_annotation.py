"""Exercise saved-frame annotation and resume with a mocked desktop."""

import csv
import importlib.util
from pathlib import Path

import cv2
import pytest

from test_replay import pair


spec = importlib.util.spec_from_file_location(
    "annotate_benchmark", Path(__file__).resolve().parents[1] / "scripts/annotate_benchmark.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_annotation_saves_target_click_and_resumes_unlabeled_rows(pair, tmp_path, monkeypatch):
    comparison, companion, _ = pair
    selection = tmp_path / "selection.csv"
    selection.write_text(
        f"companion,fly,frame,split\n{companion.name},0,2,tune\n{companion.name},1,4,evaluation\n"
    )
    output = tmp_path / "labels.csv"
    monkeypatch.setenv("DISPLAY", ":mock")
    monkeypatch.setattr(cv2, "getBuildInformation", lambda: "GUI: MOCK")
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    mouse = []
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, callback: mouse.append(callback))

    def interact(events):
        inputs = iter(events)

        def wait(delay):
            key, click = next(inputs)
            if click:
                mouse[-1](cv2.EVENT_LBUTTONDOWN, *click, 0, None)
            return ord(key) if key else -1

        monkeypatch.setattr(cv2, "waitKey", wait)
        tool.annotate(selection, output, scale=2)

    # A click while looking at a neighbor must not overwrite the target label.
    interact([("", (59, 101)), (".", None), ("", (90, 90)), ("q", None)])
    _, rows = tool.read_rows(output)
    left, top, _, _ = comparison.crop(0)
    assert (rows[0]["status"], rows[0]["x"], rows[0]["y"]) == (
        "visible", str(left+29), str(top+50)
    )
    assert rows[1]["status"] == ""

    interact([("a", None), ("q", None)])
    with output.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0] == rows[0]
    assert (saved[1]["status"], saved[1]["x"], saved[1]["y"]) == ("no_fly", "", "")
    assert saved[1]["split"] == "evaluation"

    selection.write_text(selection.read_text().replace(",1,4,", ",1,5,"))
    with pytest.raises(ValueError, match="does not match the selected rows"):
        tool.annotate(selection, output, scale=2)
