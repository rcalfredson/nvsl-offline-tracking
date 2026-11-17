"""Compare matched original, illumination-only and body-core analysis exports.

Uses unchanged acquisition history and identical upstream code/options. Missing
exported rows remain explicitly unassessed instead of disappearing in an inner
join. Metric deltas are sensitivity measurements, not accuracy estimates.
"""

import argparse
import json
from pathlib import Path

from compare_behavior_exports import change, comparable_command, sli_rows, stats_cells
from nvsl_offline.pipeline import create_run, csv_write
from nvsl_offline.recording import identity, json_write


def triple_values(values):
    rows={}
    for label,value in zip(('original','illumination','body_core'),values):
        rows[label]=change(value,value)['original']
    for first,second,label in [(0,2,'body_vs_original'),(1,2,'body_vs_illumination')]:
        delta=change(values[first],values[second])
        rows[label+'_delta']=delta['delta']
        rows[label+'_availability']=delta['availability']
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original')
    p.add_argument('illumination')
    p.add_argument('body_core')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    directories=[Path(getattr(args,label)) for label in ('original','illumination','body_core')]
    reports=[json.loads((d/'manifest.json').read_text()) for d in directories]
    if [r['source'] for r in reports]!=['original','offline','offline']:
        raise ValueError('Expected original, illumination offline, body-core offline sources')
    for report in reports[1:]:
        if report['analysis']!=reports[0]['analysis']:
            raise ValueError('Upstream analysis implementations differ')
        if comparable_command(report['command'])!=comparable_command(reports[0]['command']):
            raise ValueError('Analysis settings differ')
        if len(report['selection'])!=len(reports[0]['selection']):
            raise ValueError('Recording selections differ')
        for left,right in zip(reports[0]['selection'],report['selection']):
            if left['acquisition']!=right['acquisition'] or left['acquisition_note']!=right['acquisition_note']:
                raise ValueError('Acquisition sources or notes differ')
    for directory in directories:
        if not json.loads((directory/'result.json').read_text())['complete']:
            raise ValueError('All three analysis runs must be complete')
    out=create_run(args.output)
    exports=[sli_rows(d/'sli.csv') for d in directories]
    keys=sorted(set().union(*(set(rows) for rows in exports)))
    sli=[]
    metrics=['sli_t2_sb5','sli_t2_sb2_sb5_mean','exp_reward_pi_t2_sb5',
             'yoked_reward_pi_t2_sb5','exp_reward_pi_t2_sb2_sb5_mean','yoked_reward_pi_t2_sb2_sb5_mean']
    for key in keys:
        for metric in metrics:
            vals=[float(rows[key][metric]) if key in rows else float('nan') for rows in exports]
            sli.append(dict(pair_key=key,metric=metric,**triple_values(vals),
                original_exported=key in exports[0],illumination_exported=key in exports[1],
                body_core_exported=key in exports[2]))
    cells=[stats_cells(d/'learning_stats.csv') for d in directories]
    all_keys=sorted(set().union(*(set(rows) for rows in cells)))
    rows=[]
    for key in all_keys:
        values=[data.get(key,float('nan')) for data in cells]
        rows.append(dict(section=key[0],video=key[1],experimental_slot=key[2],
            column_index=key[3],column_label=key[4],**triple_values(values),
            original_exported=key in cells[0],illumination_exported=key in cells[1],body_core_exported=key in cells[2]))
    csv_write(out/'sli_comparison.csv',sli,list(sli[0]))
    csv_write(out/'metric_comparison.csv',rows,list(rows[0]))
    json_write(out/'summary.json',dict(input_manifests=[identity(d/'manifest.json',content_hash=True) for d in directories],
        sli=sli,metric_cells=len(rows),metric_availability={label:{state:sum(r[label+'_availability']==state for r in rows)
            for state in ('paired','newly_assessable','no_longer_assessable','missing_both')}
            for label in ('body_vs_original','body_vs_illumination')},
        missing_exported_cells={label:sum(not r[label+'_exported'] for r in rows) for label in ('original','illumination','body_core')},
        interpretation='Matched acquisition history and upstream settings; unexported values retained. Metric deltas do not identify the more accurate tracker.'))
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':main()
