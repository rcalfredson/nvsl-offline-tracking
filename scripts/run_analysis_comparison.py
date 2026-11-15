"""Run upstream analysis against an explicitly staged tracking source.

Creates isolated local symlinks for video, unchanged .data, selected .trx and any
acquisition note file. Never replaces acquisition sidecars or edits upstream code.
The upstream checkout (including its compiled extensions) is an explicit external
dependency of this evaluation bridge, not a dependency of the offline tracker.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nvsl_offline.pipeline import assert_sources, create_run, producer
from nvsl_offline.recording import Recording, identity, json_write
from nvsl_offline.replay import Comparison


def analysis_identity(root):
    digest = hashlib.sha256()
    paths = [root/'analyze.py', *sorted((root/'src').rglob('*'))]
    for path in paths:
        if path.is_file() and path.suffix in ('.py', '.pyx', '.so'):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return dict(root=str(root), source_and_extensions_sha256=digest.hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True).splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('status',help='Full-recording-check status JSON with jobs/video/companion fields')
    p.add_argument('--analysis-root',required=True)
    p.add_argument('--source',choices=['original','offline'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root=Path(args.analysis_root).resolve(strict=True)
    state=json.loads(Path(args.status).read_text())
    out=create_run(args.output).resolve()
    # Upstream plotting assumes these working-directory subdirectories exist.
    (out/'imgs').mkdir()
    (out/'exports').mkdir()
    upstream=analysis_identity(root)
    selected=[]
    for i,job in enumerate(state['jobs']):
        rec=Recording.load(job['video'])
        if args.source=='offline':
            comparison=Comparison.load(job['companion'])
            if comparison.original.video != rec.video:
                raise ValueError('Companion belongs to another recording')
            trx=Path(job['companion']).resolve(strict=True)
        else:
            trx=rec.trx_path
        stage=out/'inputs'/f'{i+1:02d}'
        stage.mkdir(parents=True)
        video=stage/rec.video.name
        for link,target in [(video,rec.video),(video.with_suffix('.data'),rec.data_path),
                            (video.with_suffix('.trx'),trx)]:
            link.symlink_to(target)
        note=rec.video.with_name(rec.video.stem+'__note.txt')
        if note.exists():
            video.with_name(video.stem+'__note.txt').symlink_to(note)
        selected.append(dict(video=str(video), original_video=str(rec.video),
            acquisition=rec.sources(), selected_trx=identity(trx,content_hash=True),
            acquisition_note=identity(note,content_hash=True) if note.exists() else None))
    argv=[sys.executable,str(root/'analyze.py'),'-v',','.join(r['video'] for r in selected),
          '-f','0-1','--rCC','15','--num-trainings','2','--rpd',
          '--best-worst-trn','2','--sli-use-training-mean',
          '--sli-select-skip-first-sync-buckets','1','--sli-min-valid-sync-buckets','3',
          '--no-wall-cache','--hidePltTests','--cross-experiment-sli-export',str(out/'sli.csv')]
    manifest=dict(source=args.source,producer=producer(),analysis=upstream,
        selection=selected,command=argv,cache_policy='Separate working directories and staged source paths; wall cache disabled.',
        event_policy='Same acquisition .data and any note file in both variants; only selected .trx differs.',
        started=time.time(),completion_record='result.json')
    json_write(out/'manifest.json',manifest)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
             MPLCONFIGDIR=str(out/'mpl-config'),NUMBA_CACHE_DIR=str(out/'numba-cache'))
    with (out/'process.log').open('x') as log:
        completed=subprocess.run(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    for entry in selected:
        assert_sources(entry['acquisition'])
        if identity(Path(entry['selected_trx']['path']),content_hash=True)!=entry['selected_trx']:
            raise ValueError('Selected trajectory changed during analysis')
        if entry['acquisition_note'] is not None and identity(Path(entry['acquisition_note']['path']),content_hash=True)!=entry['acquisition_note']:
            raise ValueError('Acquisition note changed during analysis')
    if analysis_identity(root)!=upstream:
        raise ValueError('Upstream analysis source changed during processing')
    json_write(out/'result.json',dict(returncode=completed.returncode,finished=time.time(),
        complete=completed.returncode==0,stats_file=str(out/'learning_stats.csv'),sli_file=str(out/'sli.csv')))
    if completed.returncode:
        raise SystemExit(f'Analysis failed; see {out}/process.log')
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
