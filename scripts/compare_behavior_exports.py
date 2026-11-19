"""Pair upstream analysis exports by recording and physical experimental slot.

Retains duplicate CSV header labels using their column positions. A metric delta
is a sensitivity measurement, not a tracking-accuracy estimate.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from nvsl_offline.pipeline import create_run, csv_write
from nvsl_offline.recording import identity, json_write


def stats_cells(path):
    cells = {}
    heading = ''
    columns = None
    with Path(path).open(newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                columns = None
                continue
            if row[:2] == ['video', 'fly']:
                columns = row
            elif columns is not None and len(row) >= 2 and row[0].endswith('.avi'):
                if len(row) != len(columns):
                    raise ValueError(f'Inconsistent CSV width in {heading}')
                for index, (label, value) in enumerate(zip(columns[2:], row[2:]), 2):
                    key = (heading, Path(row[0]).name, int(row[1]), index, label)
                    if key in cells:
                        raise ValueError('Duplicate metric cell')
                    cells[key] = float('nan') if value.strip().lower() in ('none', 'nan', '') else float(value)
            elif columns is None:
                heading = row[0]
    return cells


def change(a, b):
    first, second = math.isfinite(a), math.isfinite(b)
    state = ('paired' if first and second else 'newly_assessable' if second else
             'no_longer_assessable' if first else 'missing_both')
    return dict(original=a if first else '', offline=b if second else '',
                delta=b-a if first and second else '', availability=state)


def comparable_command(argv):
    result = list(argv)
    for flag in ('-v','--cross-experiment-sli-export'):
        index = result.index(flag)
        result[index+1] = '<source-specific path>'
    return result


def sli_rows(path, recording_paths=None):
    rows = {}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['pair_key']
            if recording_paths is not None:
                video = recording_paths[row['closed_video']]
                key = f"{video}::pair{int(row['exp_fly_id'])}"
            if key in rows:
                raise ValueError(f'Duplicate SLI pair key: {key}')
            rows[key] = row
    if not rows:
        raise ValueError('No SLI pair rows to compare')
    return rows


def validate_source_order(reports):
    if [report.get('source') for report in reports] != ['original', 'offline']:
        raise ValueError('Expected original analysis first and offline analysis second')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original')
    p.add_argument('offline')
    p.add_argument('--output',required=True)
    args = p.parse_args()
    original, offline = Path(args.original), Path(args.offline)
    out = create_run(args.output)
    reports = [json.loads((d/'manifest.json').read_text()) for d in (original,offline)]
    validate_source_order(reports)
    if reports[0]['analysis'] != reports[1]['analysis']:
        raise ValueError('Analysis implementations differ')
    if comparable_command(reports[0]['command']) != comparable_command(reports[1]['command']):
        raise ValueError('Analysis settings differ')
    for directory in (original,offline):
        if not json.loads((directory/'result.json').read_text()).get('complete'):
            raise ValueError('Both analysis runs must have completed successfully')
    for before,after in zip(reports[0]['selection'],reports[1]['selection']):
        if before['acquisition'] != after['acquisition'] or before['acquisition_note'] != after['acquisition_note']:
            raise ValueError('Acquisition sources or notes differ')
    if len(reports[0]['selection']) != len(reports[1]['selection']):
        raise ValueError('Recording selections differ')
    sli = []
    sli_sources = [
        sli_rows(directory/'sli.csv', {entry['video']: entry['original_video']
                                       for entry in report['selection']})
        for directory, report in zip((original, offline), reports)
    ]
    sli_metrics = ['sli_t2_sb5','sli_t2_sb2_sb5_mean','exp_reward_pi_t2_sb5',
                   'yoked_reward_pi_t2_sb5','exp_reward_pi_t2_sb2_sb5_mean',
                   'yoked_reward_pi_t2_sb2_sb5_mean']
    for key in sorted(sli_sources[0].keys() | sli_sources[1].keys()):
        a,b = [s.get(key, {}) for s in sli_sources]
        for metric in sli_metrics:
            sli.append(dict(pair_key=key,metric=metric,
                **change(float(a[metric]) if a else math.nan, float(b[metric]) if b else math.nan),
                original_exported=bool(a), offline_exported=bool(b),
                original_valid_buckets=a.get('sli_t2_sb2_sb5_valid_bucket_count', ''),
                offline_valid_buckets=b.get('sli_t2_sb2_sb5_valid_bucket_count', '')))
    left,right = [stats_cells(d/'learning_stats.csv') for d in (original,offline)]
    def schema(cells):
        return {(section, index, label) for section, _, _, index, label in cells}

    if schema(left) != schema(right):
        raise ValueError('Metric export schemas differ')
    cells = [dict(section=key[0],video=key[1],experimental_slot=key[2],
                  column_index=key[3],column_label=key[4],
                  original_exported=key in left,offline_exported=key in right,
                  **change(left.get(key, math.nan),right.get(key, math.nan)))
             for key in sorted(left.keys() | right.keys())]
    csv_write(out/'sli_comparison.csv',sli,list(sli[0]))
    csv_write(out/'metric_comparison.csv',cells,list(cells[0]))
    json_write(out/'summary.json',dict(
        input_manifests=[identity(d/'manifest.json',content_hash=True) for d in (original,offline)],
        sli=sli,metric_cells=len(cells),
        metric_availability={name:sum(r['availability']==name for r in cells) for name in
            ['paired','newly_assessable','no_longer_assessable','missing_both']},
        interpretation='Source-sensitive behavioral outputs, not localization ground truth. Analysis preprocessing and standard eligibility filters retained.'))
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
