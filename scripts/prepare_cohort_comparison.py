"""Freeze an explicit cohort selection and preflight acquisition sidecars/headers.

Does not execute list/notebook code, decode whole recordings, track, or run
behavioral analysis. Unusable recordings stay in the manifest with hold reasons.
"""

import argparse
import ast
from dataclasses import asdict
from datetime import date
import glob
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from nvsl_offline.background import BackgroundConfig
from nvsl_offline.detection import DetectionConfig
from nvsl_offline.pipeline import create_run, csv_write, producer
from nvsl_offline.recording import Calibration, Recording, identity, json_write
from nvsl_offline.video import probe


COHORTS = [
    dict(name='flat-ar-control', label='Flat AR-lower chamber',
         notebook_variable='FIGURE_4D_ANTENNAE_REMOVED_VIDEO_GLOBS', figure='4d'),
    dict(name='agarose-ar-control', label='Agarose AR-lower chamber',
         notebook_variable='FIGURE_4T_ANTENNAE_REMOVED_VIDEO_GLOBS', figure='4t'),
]


def list_selection(text, label):
    """Select within the control genotype's large-chamber section only."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == 'Large chamber tasks')
    start = next(i for i in range(start+1, len(lines)) if lines[i].startswith('Ctrl group:'))
    stop = next(i for i in range(start+1, len(lines)) if lines[i].startswith('UAS>>'))
    matches = [i for i in range(start, stop) if lines[i].startswith(label)]
    if len(matches) != 1:
        raise ValueError(f'Expected one control-genotype cohort heading: {label}')
    heading = matches[0]
    command_line = heading+1
    args = shlex.split(lines[command_line])
    if args.count('-v') != 1:
        raise ValueError('Expected one explicit -v argument immediately after cohort heading')
    patterns = [p.strip() for p in args[args.index('-v')+1].split(',')]
    if not patterns or any(not p or '|' in p for p in patterns):
        raise ValueError('Expected a single cohort with nonempty patterns')
    return dict(heading=lines[heading], genotype=lines[start],
                heading_line=heading+1, command_line=command_line+1,
                original_command=lines[command_line], patterns=patterns)


def notebook_patterns(path, variable):
    """Read literal assignment only; never execute a notebook cell."""
    values = []
    for index, cell in enumerate(json.loads(path.read_text())['cells']):
        if cell.get('cell_type') != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == variable for t in node.targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise ValueError('Expected a literal pattern string')
                values.append(dict(cell_index=index, variable=variable, patterns=value.split(',')))
    if len(values) != 1:
        raise ValueError(f'Expected one literal assignment for {variable}')
    return values[0]


def phase_holds(groups, nframes):
    reasons = []
    if not isinstance(groups, list) or len(groups) != 2:
        return ['Expected two experimental-fly event groups for analysis -f 0-1']
    for i, group in enumerate(groups):
        values = [group.get(k, []) for k in ('startPre', 'startTrain', 'startPost')]
        if not len(values[0]) == len(values[1]) == len(values[2]) or len(values[1]) < 2:
            reasons.append(f'Experimental fly {i}: incomplete phase boundaries for T1/T2')
            continue
        boundaries = [v for triple in zip(*values) for v in triple]
        if any(v < 0 or v > nframes for v in boundaries) or np.any(np.diff(boundaries) < 0):
            reasons.append(f'Experimental fly {i}: unordered/out-of-range phases')
        if any(values[2][t] <= values[1][t] for t in range(2)):
            reasons.append(f'Experimental fly {i}: empty training interval')
    return reasons


def examine(video):
    row = dict(video=video, disposition='hold', hold_reasons=[], warnings=[])
    try:
        rec = Recording.load(video)
        info = probe(rec.video)
        row.update(sources=rec.sources(), frames=rec.shape[1], slots=rec.shape[0],
                   video_header=asdict(info), original=rec.summary(),
                   phase_events=[{k: [int(v) for v in g.get(k, [])] for k in ('startPre','startTrain','startPost')}
                                 for g in rec.protocol['frameNums']],
                   ef2yc={str(k): int(v) for k,v in rec.protocol.get('ef2yc', {}).items()})
        row['hold_reasons'].extend(phase_holds(row['phase_events'], rec.shape[1]))
        if info.header_frames != rec.shape[1]:
            row['hold_reasons'].append('Video header/trajectory frame counts differ')
        if np.any(np.diff(rec.ts) < 0):
            row['hold_reasons'].append('Timestamps decrease')
        if rec.protocol.get('ef2yc') and rec.protocol['ef2yc'] != {0: 2, 1: 3}:
            row['hold_reasons'].append('Unexpected experimental/yoked mapping; inspect before analysis')
        row['calibration'] = asdict(Calibration.from_recording(
            rec, info.width, info.height, layout='large2', margin=24))
        empty = np.flatnonzero(np.isnan(rec.arrays['x']).all(axis=1)).tolist()
        if empty:
            row['warnings'].append(f'Original all-missing slots {empty}; occupancy is unconfirmed, not excluded')
        if np.any(np.diff(rec.ts) == 0):
            row['warnings'].append('Repeated timestamps')
        note = rec.video.with_name(rec.video.stem+'__note.txt')
        row['acquisition_note'] = identity(note, content_hash=True) if note.exists() else None
        if not row['hold_reasons']:
            row['disposition'] = 'ready_for_audit'
    except Exception as exc:
        row['hold_reasons'].append(f'{type(exc).__name__}: {exc}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', required=True)
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--date', required=True, type=date.fromisoformat, help='Plan date, YYYY-MM-DD')
    args = parser.parse_args()
    source, notebook = Path(args.list).resolve(), Path(args.notebook).resolve()
    source_id, notebook_id = identity(source, content_hash=True), identity(notebook, content_hash=True)
    selections = []
    for spec in COHORTS:
        selection = list_selection(source.read_text(), spec['label'])
        crosscheck = notebook_patterns(notebook, spec['notebook_variable'])
        if set(selection['patterns']) != set(crosscheck['patterns']):
            raise ValueError(f"List/notebook cohort disagreement: {spec['name']}")
        matched = {p: sorted(str(Path(v).resolve()) for v in glob.glob(p+'.avi'))
                   for p in selection['patterns']}
        if any(not paths for paths in matched.values()):
            raise ValueError(f'Unmatched video pattern(s): {[p for p,v in matched.items() if not v]}')
        selections.append(dict(**spec, selection=selection, notebook_crosscheck=crosscheck,
                               matches_by_pattern=matched,
                               videos=sorted({v for paths in matched.values() for v in paths})))
    if set(selections[0]['videos']) & set(selections[1]['videos']):
        raise ValueError('Flat and agarose cohorts overlap')
    config = DetectionConfig(mode='normalized', illumination='global_offset', segmentation='body_core')
    config.validate()
    root = Path(__file__).resolve().parents[1]
    validated_plan = root/'docs/body_core_full_check_plan.json'
    if asdict(config) != json.loads(validated_plan.read_text())['configuration']:
        raise ValueError('Configuration differs from the manually reviewed full-recording check')
    frozen_producer = producer()
    tracked = subprocess.check_output(['git','diff','HEAD','--','src/nvsl_offline'],cwd=root,text=True)
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard',
                                        'src/nvsl_offline'],cwd=root,text=True)
    if tracked or untracked:
        raise ValueError('Commit tracker package changes before freezing a cohort plan')
    out = create_run(args.output)
    frozen = dict(schema_version=1, date=args.date.isoformat(), producer=frozen_producer,
        tracker_package_matches_commit=True,
        validated_plan=identity(validated_plan, content_hash=True),
        detector=asdict(config), background=asdict(BackgroundConfig()),
        audit=dict(layout='large2', margin=24, samples=129, hash_video=False),
        contracts=dict(saved_frame_alignment=True, unresolved='NaN', interpolation=False,
                       source_sidecars='preserved', events='unchanged acquisition history'),
        manual_qa=dict(recordings=3, flies=12, accepted_for_reviewed_intervals=12,
                       scope='User completed fixed-window QA; not a full accuracy benchmark.'),
        input_list=source_id, notebook=notebook_id,
        execution='Preparation only: no tracking or behavioral analysis launched.')
    json_write(out/'freeze.json', frozen)
    for cohort in selections:
        jobs = []
        for index, video in enumerate(cohort.pop('videos')):
            print(f"{cohort['name']} {index+1}: {Path(video).name}", flush=True)
            jobs.append(dict(index=index, **examine(video)))
        cohort.update(schema_version=1, jobs=jobs, input_list=source_id, notebook=notebook_id,
                      freeze=str((out/'freeze.json').resolve()),
                      counts=dict(recordings=len(jobs), ready_for_audit=sum(
                          j['disposition']=='ready_for_audit' for j in jobs),
                          held=sum(j['disposition']=='hold' for j in jobs)))
        json_write(out/(cohort['name']+'.json'), cohort)
        csv_write(out/(cohort['name']+'.csv'), [dict(video=j['video'],frames=j.get('frames'),
                  disposition=j['disposition'],hold_reasons='; '.join(j['hold_reasons']),
                  warnings='; '.join(j['warnings'])) for j in jobs],
                  ['video','frames','disposition','hold_reasons','warnings'])
        print(cohort['name'], cohort['counts'], flush=True)
    if identity(source, content_hash=True) != source_id or identity(notebook, content_hash=True) != notebook_id:
        raise ValueError('Cohort definition changed during preparation')


if __name__ == '__main__':
    main()
