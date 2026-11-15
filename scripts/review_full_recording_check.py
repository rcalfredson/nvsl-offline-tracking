"""Create tracking diagnostics and reproducible image review points for full runs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from nvsl_offline.pipeline import create_run
from nvsl_offline.recording import json_write
from nvsl_offline.replay import Comparison, Controls, FrameReader, save_snapshot


def gap_ranges(values):
    edges=np.diff(np.r_[False,np.isnan(values),False].astype(int))
    ranges=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    return sorted([(int(a),int(b)) for a,b in ranges],key=lambda p:p[1]-p[0],reverse=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('status')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    state=json.loads(Path(args.status).read_text())
    if any(j['stage']!='complete' for j in state['jobs']):
        raise ValueError('All tracking jobs must be complete')
    out=create_run(args.output).resolve()
    report=dict(recordings=[],interpretation='Unannotated review selection. Missingness, disagreement and jumps do not establish tracking accuracy.')
    guide=['Full-recording check: review commands','',
           'Saved-frame and slot indices are zero-based. Commands open paused. Space plays; comma/period step. Source events and timestamps are unchanged.','']
    for job in state['jobs']:
        c=Comparison.load(job['companion'])
        root=out/Path(job['directory']).name
        root.mkdir()
        quality=c.variant.raw['tracking_quality']
        if quality['illumination_offset'].shape!=(c.nframes,):
            raise ValueError('Illumination metadata does not cover every saved frame')
        record=dict(video=str(c.original.video),companion=job['companion'],
                    frames=c.nframes,alignment='Comparison.load verified source identities, array shape and original timestamps',
                    slots=[],review_points=[])
        points={}
        def select(slot,frame,reason):
            if 0<=frame<c.nframes:
                points.setdefault((slot,frame),[]).append(reason)
        for slot in range(4):
            old,new=c.original.arrays['x'][slot],c.variant.arrays['x'][slot]
            missing_old,missing_new=np.isnan(old),np.isnan(new)
            common=~(missing_old|missing_new)
            displacement=np.hypot(old-c.variant.arrays['x'][slot],
                c.original.arrays['y'][slot]-c.variant.arrays['y'][slot])
            gaps=gap_ranges(old)
            new_gaps=gap_ranges(new)
            record['slots'].append(dict(slot=slot,
                original_missing=int(missing_old.sum()),offline_missing=int(missing_new.sum()),
                newly_finite=int((missing_old&~missing_new).sum()),newly_missing=int((~missing_old&missing_new).sum()),
                common_finite=int(common.sum()),common_disagreement_over_10px=int((displacement>10).sum()),
                original_longest_gaps=gaps[:3],offline_longest_gaps=new_gaps[:3],
                offline_reason_counts=dict(Counter(c.reason_names[int(value)] for value in quality['reason'][slot]))))
            for frame in [400,10000,54000,90000]:
                select(slot,frame,'predetermined distributed frame')
            if gaps:
                first,stop=gaps[0]
                for frame,which in [(first,'start'),((first+stop)//2,'middle'),(stop-1,'end')]:
                    select(slot,frame,f'longest original gap: {which}')
            if common.any():
                frame=int(np.nanargmax(displacement))
                select(slot,frame,'largest original/offline center disagreement; either source may be wrong')
            # A sustained new gap can escape the jump screen.
            newly_missing=~missing_old&missing_new
            new_loss_gaps=gap_ranges(np.where(newly_missing,np.nan,0))
            if new_loss_gaps:
                first,stop=new_loss_gaps[0]
                select(slot,(first+stop)//2,'longest run of newly missing measurements')
        if '2025-07-21' in str(c.original.video):
            for frame in [617,1005,1058]:
                select(0,frame,'previously reviewed startup rim-lock')
        with Path(job['comparison']).with_name('pairs.csv').open(newline='') as stream:
            pairs=list(csv.DictReader(stream))
        old_pairs={(r['slot'],r['mode'],r['first_jump'],r['second_jump']) for r in pairs if r['source']=='original'}
        record['offline_only_pairs']=[]
        for row in pairs:
            key=(row['slot'],row['mode'],row['first_jump'],row['second_jump'])
            if row['source']=='offline' and key not in old_pairs:
                record['offline_only_pairs'].append(row)
                select(int(row['slot']),int(row['first_jump'])+1,'offline-only suspicious pair; visual adjudication required')
        reader=FrameReader(c.original.video)
        try:
            for (slot,frame),reasons in sorted(points.items(),key=lambda p:(p[0][1],p[0][0])):
                image=c.render(reader.read(frame),frame,fly=slot,scale=1.)
                target=root/f'frame{frame:07d}-slot{slot}.png'
                controls=Controls(0,c.nframes,frame,slot)
                save_snapshot(target,image,c,controls)
                record['review_points'].append(dict(frame=frame,slot=slot,reasons=reasons,snapshot=str(target)))
        finally:
            reader.close()
        offset=quality['illumination_offset']
        record['illumination_offset_percentiles']=dict(zip(['p0','p50','p95','p100'],map(float,np.percentile(offset,[0,50,95,100]))))
        report['recordings'].append(record)
        guide += [f"**{c.original.video.stem} — {job['role']}**",'']
        for slot in range(4):
            gaps=gap_ranges(c.original.arrays['x'][slot])
            if not gaps:
                guide += [f'Slot {slot}: no original gaps.','']
                continue
            first,stop=gaps[0]
            command=f"PYTHONPATH=src {shlex.quote(sys.executable)} -m nvsl_offline compare {shlex.quote(job['companion'])} --fly {slot} --start {max(0,first-15)} --speed 0.5"
            guide += [f"Slot {slot}, longest original gap: {first}–{stop-1}. Midpoint: {(first+stop)//2}.",'','```bash',command,'```','']
        guide += ['Offline-only suspicious pair indices (first jump / second jump):','']
        if record['offline_only_pairs']:
            for row in record['offline_only_pairs']:
                guide.append(f"- slot {row['slot']}, {row['mode']}: {row['first_jump']} / {row['second_jump']}")
        else:
            guide.append('None.')
        guide.append('')
        print(f"Reviewed artifact selection: {c.original.video.name}, {len(points)} snapshots",flush=True)
    json_write(out/'summary.json',report)
    (out/'review.md').write_text('\n'.join(guide)+'\n')
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
