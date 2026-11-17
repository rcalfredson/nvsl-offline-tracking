"""Cohort selection must not silently mix genotypes or execute notebook code."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cohort_preparation',
    Path(__file__).resolve().parents[1]/'scripts/prepare_cohort_comparison.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_is_scoped_to_large_chamber_control_genotype():
    text = '''Flat AR-lower chamber old experiment
python analyze.py -v "old"
Large chamber tasks
Ctrl group: control
Flat AR-lower chamber current experiment
python analyze.py -v "/data/a[12]_*,/data/b_*" -f 0-1
UAS>> other genotype
Flat AR-lower chamber other genotype
python analyze.py -v "other"
'''
    selected = module.list_selection(text, 'Flat AR-lower chamber')
    assert selected['patterns'] == ['/data/a[12]_*', '/data/b_*']
    assert selected['command_line'] == 6
    with pytest.raises(ValueError):
        module.list_selection(text, 'Agarose AR-lower chamber')


def test_notebook_reads_literal_without_executing_cell(tmp_path):
    path = tmp_path/'notebook.ipynb'
    path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [
        "raise RuntimeError('must never execute')\nVIDEOS = '/data/a_*,/data/b_*'\n"]}]}))
    assert module.notebook_patterns(path, 'VIDEOS')['patterns'] == ['/data/a_*', '/data/b_*']
    with pytest.raises(ValueError):
        module.notebook_patterns(path, 'MISSING')


def test_incomplete_acquisition_is_held_and_complete_phases_pass():
    good = dict(startPre=[2,20,40], startTrain=[5,22,42], startPost=[15,35,55])
    assert module.phase_holds([good, good], 60) == []
    assert module.phase_holds([{}, {}], 65)
    assert module.phase_holds([good], 60)
    assert module.phase_holds([good, dict(startPre=[2,20],startTrain=[5,22],startPost=[15,21])],60)
