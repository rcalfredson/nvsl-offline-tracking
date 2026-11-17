"""Recorded-event partitions must cover frames once, with explicit boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('phase_comparison',
    Path(__file__).resolve().parents[1]/'scripts/compare_full_tracking_variants.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_phase_windows_partition_saved_frames_with_exclusive_stops():
    events=dict(startPre=[2,15],startTrain=[5,17],startPost=[10,22])
    windows=module.phase_windows(events,25)
    assert windows==[('startup',0,2),('pre_1',2,5),('training_1',5,10),
                     ('post_1',10,15),('pre_2',15,17),('training_2',17,22),('post_2',22,25)]
    coverage=np.zeros(25,int)
    for name,first,stop in windows:coverage[first:stop]+=1
    np.testing.assert_array_equal(coverage,1)


def test_empty_phases_do_not_duplicate_frames_and_invalid_metadata_fails():
    assert module.phase_windows(dict(startPre=[0],startTrain=[0],startPost=[5]),5)==[('training_1',0,5)]
    for events in [dict(startPre=[3],startTrain=[2],startPost=[5]),
                   dict(startPre=[0],startTrain=[2],startPost=[6]),
                   dict(startPre=[0],startTrain=[2.5],startPost=[5]),
                   dict(startPre=[0],startTrain=[],startPost=[5])]:
        with pytest.raises(ValueError):module.phase_windows(events,5)
