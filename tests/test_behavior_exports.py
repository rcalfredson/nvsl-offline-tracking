"""Protect metric pairing against repeated upstream CSV labels and missing cells."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('behavior_exports',
    Path(__file__).resolve().parents[1]/'scripts/compare_behavior_exports.py')
exports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exports)


def test_repeated_bucket_headers_remain_distinct_and_missing_cells_are_not_zero(tmp_path):
    path = tmp_path/'stats.csv'
    path.write_text('distance:\nvideo,fly,training 1 exp b1,#2,training 2 exp b1,#2\n'
                    'c51__example.avi,0,10,20,None,nan\n')
    cells = exports.stats_cells(path)
    assert len(cells) == 4
    assert cells[('distance:','c51__example.avi',0,3,'#2')] == 20
    assert np.isnan(cells[('distance:','c51__example.avi',0,5,'#2')])
    assert exports.change(float('nan'),2)['availability'] == 'newly_assessable'
    assert exports.change(2,float('nan'))['availability'] == 'no_longer_assessable'


def test_pair_or_schema_collisions_fail_instead_of_overwriting(tmp_path):
    path = tmp_path/'stats.csv'
    path.write_text('distance:\nvideo,fly,value\na.avi,0,1\na.avi,0,2\n')
    with pytest.raises(ValueError,match='Duplicate'):
        exports.stats_cells(path)
    path.write_text('distance:\nvideo,fly,value\na.avi,0,1,2\n')
    with pytest.raises(ValueError,match='width'):
        exports.stats_cells(path)


def test_comparable_commands_ignore_only_source_specific_paths():
    a = ['python','analyze.py','-v','a.avi','--rCC','15','--cross-experiment-sli-export','a.csv']
    b = ['python','analyze.py','-v','b.avi','--rCC','15','--cross-experiment-sli-export','b.csv']
    assert exports.comparable_command(a) == exports.comparable_command(b)
    b[5] = '12'
    assert exports.comparable_command(a) != exports.comparable_command(b)


def test_duplicate_sli_pair_keys_cannot_overwrite_another_recording(tmp_path):
    path = tmp_path/'sli.csv'
    path.write_text('pair_key,sli_t2_sb5\n2025-07-17::c52::pair0,0.2\n'
                    '2025-07-17::c52::pair0,0.4\n')
    with pytest.raises(ValueError, match='Duplicate SLI pair key'):
        exports.sli_rows(path)
    path.write_text('pair_key,sli_t2_sb5\n')
    with pytest.raises(ValueError, match='No SLI pair rows'):
        exports.sli_rows(path)


def test_behavior_sources_cannot_be_swapped_or_compared_against_themselves():
    exports.validate_source_order([dict(source='original'),dict(source='offline')])
    for sources in [('offline','original'),('original','original'),('offline','offline')]:
        with pytest.raises(ValueError, match='original analysis first'):
            exports.validate_source_order([dict(source=source) for source in sources])
