"""Compare original, illumination-only and body-core trajectories by recorded phase.

Input is a batch status with completed jobs, companion and illumination_companion.
Event boundaries come from unchanged acquisition metadata; interpolation is used
only in diagnostic jump scoring. Review images are not manual accuracy labels.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from nvsl_offline.detection import DetectionConfig
from nvsl_offline.jumps import score_positions
from nvsl_offline.pipeline import create_run, csv_write, producer, write_image
from nvsl_offline.recording import FIELDS, identity, json_write
from nvsl_offline.replay import Comparison, FrameReader


def phase_windows(events, nframes):
    """Disjoint [start, stop) partition, including acquisition startup and final post."""
    pre, train, post = [events[key] for key in ('startPre','startTrain','startPost')]
    if not len(pre) == len(train) == len(post) or not pre:
        raise ValueError('Training event lists must be nonempty and equal length')
    boundaries=[('startup',0)]
    for index, values in enumerate(zip(pre,train,post),1):
        for name, value in zip(('pre','training','post'),values):
            if int(value)!=value or not 0<=value<=nframes:
                raise ValueError('Invalid acquisition phase index')
            boundaries.append((f'{name}_{index}',int(value)))
    boundaries.append(('end',nframes))
    if any(b[1]<a[1] for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Acquisition phase boundaries are not ordered')
    return [(a[0],a[1],b[1]) for a,b in zip(boundaries,boundaries[1:]) if b[1]>a[1]]


def gap_ranges(missing):
    edges=np.diff(np.r_[False,missing,False].astype(int))
    return sorted(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)),key=lambda p:p[1]-p[0],reverse=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('status')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    state=json.loads(Path(args.status).read_text())
    if not state['jobs'] or any(j['stage']!='complete' for j in state['jobs']):
        raise ValueError('All batch jobs must be complete')
    out=create_run(args.output).resolve()
    phases,totals,changes,reviews,pair_rows=[],[],[],[],[]
    for job in state['jobs']:
        body=Comparison.load(job['companion'])
        old=Comparison.load(job['illumination_companion'])
        if body.original.sources()!=old.original.sources():
            raise ValueError('Acquisition sources differ')
        meta=body.variant.raw['tracking_metadata']; previous=old.variant.raw['tracking_metadata']
        a=DetectionConfig(**previous['configuration']['detector'])
        b=DetectionConfig(**meta['configuration']['detector'])
        if b.segmentation!='body_core' or a.segmentation!='components' or replace(
                b,segmentation=a.segmentation,core_radius=a.core_radius,core_growth=a.core_growth)!=a:
            raise ValueError('Comparison changes parameters beyond body-core segmentation')
        if meta['background_sha256']!=previous['background_sha256']:
            raise ValueError('Variants use different backgrounds')
        variants={'original':body.original,'illumination':old.variant,'body_core':body.variant}
        protocol=body.original.protocol
        event_lists=protocol['frameNums']
        if not isinstance(event_lists,list):raise ValueError('Expected per-experimental-fly frameNums')
        nexp=sum(bool(events) for events in event_lists)
        mapping=protocol.get('ef2yc') or {}
        owner={i:i for i in range(nexp)}
        owner.update({mapping.get(i,i+nexp):i for i in range(nexp)})
        points={}
        def select(slot,frame,reason):
            if 0<=frame<body.nframes:points.setdefault((slot,int(frame)),[]).append(reason)
        for slot in range(body.original.shape[0]):
            windows=phase_windows(event_lists[owner[slot]],body.nframes)
            scores={}
            for label,rec in variants.items():
                x,y=rec.arrays['x'][slot],rec.arrays['y'][slot]
                score,pairs=score_positions(x,y,open_loop=protocol.get('pt')=='openLoop')
                scores[label]=pairs
                totals.append(dict(video=str(body.original.video),slot=slot,variant=label,**score))
                for name,first,stop in windows:
                    missing=np.isnan(x[first:stop]);gaps=gap_ranges(missing)
                    row=dict(video=str(body.original.video),slot=slot,variant=label,phase=name,
                        first=first,stop=stop,frames=stop-first,missing_frames=int(missing.sum()),
                        missing_fraction=float(missing.mean()),longest_gap_frames=int(gaps[0][1]-gaps[0][0]) if gaps else 0,
                        raw_suspicious_pairs=sum(e['mode']=='raw' and first<=e['first_jump'] and e['second_jump']+1<stop for e in pairs),
                        analysis_suspicious_pairs=(sum(e['mode']=='analysis' and first<=e['first_jump'] and e['second_jump']+1<stop for e in pairs)
                            if score['analysis_state']=='eligible' else None))
                    phases.append(row)
                pair_rows.extend(dict(video=str(body.original.video),slot=slot,variant=label,**e) for e in pairs)
            old_x,new_x=old.variant.arrays['x'][slot],body.variant.arrays['x'][slot]
            old_missing,new_missing=np.isnan(old_x),np.isnan(new_x)
            displacement=np.hypot(old_x-new_x,old.variant.arrays['y'][slot]-body.variant.arrays['y'][slot])
            common=np.isfinite(displacement)
            changes.append(dict(video=str(body.original.video),slot=slot,
                newly_finite=int((old_missing&~new_missing).sum()),newly_missing=int((~old_missing&new_missing).sum()),
                common_finite=int(common.sum()),center_difference_median=float(np.median(displacement[common])) if common.any() else None,
                center_difference_max=float(displacement[common].max()) if common.any() else None,
                center_difference_over_10px=int((displacement>10).sum())))
            for frame in [400,10000,54000,90000]:select(slot,frame,'distributed control')
            for name,first,stop in windows:select(slot,(first+stop)//2,f'{name} midpoint')
            for mask,reason in [(old_missing&~new_missing,'newly finite'),(~old_missing&new_missing,'newly missing')]:
                gaps=gap_ranges(mask)
                if gaps:select(slot,(gaps[0][0]+gaps[0][1])//2,'longest '+reason+' run')
            if common.any():select(slot,np.nanargmax(displacement),'largest illumination/body-core center disagreement')
            old_pairs={(e['mode'],e['first_jump'],e['second_jump']) for e in scores['illumination']}
            for e in scores['body_core']:
                if (e['mode'],e['first_jump'],e['second_jump']) not in old_pairs:
                    select(slot,e['first_jump']+1,'body-core-only pair (exact-index comparison)')
                    select(slot,e['second_jump']+1,'body-core-only pair return')
        if '2025-07-17' in job['video']:
            for slot,frame in [(2,1113),(3,293),(1,54002),(3,61592)]:select(slot,frame,'pilot case')
        if '2025-07-27' in job['video']:
            for slot,frame in [(2,362),(3,387)]:select(slot,frame,'pilot case')
        directory=out/Path(job['video']).stem;directory.mkdir()
        reader=FrameReader(body.original.video)
        try:
            for (slot,index),reasons in sorted(points.items(),key=lambda p:(p[0][1],p[0][0])):
                frame=reader.read(index);x0,y0,x1,y1=body.crop(slot);panels=[]
                for label,rec,color in [('original',body.original,(0,220,255)),('illumination',old.variant,(80,255,80)),('body core',body.variant,(255,100,255))]:
                    values=[rec.arrays[k][slot,index] for k in FIELDS];image=frame.copy()
                    valid=np.isfinite(values).all()
                    if valid:cv2.ellipse(image,((values[0],values[1]),(values[2],values[3]),values[4]),color,1)
                    image=cv2.copyMakeBorder(image[y0:y1,x0:x1],40,0,0,0,cv2.BORDER_CONSTANT)
                    title=label+(': measured' if valid else ': missing')
                    if not valid and 'tracking_quality' in rec.raw:
                        q=rec.raw['tracking_quality'];names={v:k for k,v in q['reason_codes'].items()}
                        title=label+': '+names[int(q['reason'][slot,index])]
                    cv2.putText(image,title,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                    cv2.putText(image,f'frame {index}; slot {slot}',(3,32),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);panels.append(image)
                path=directory/f'frame{index:07d}-slot{slot}.png';write_image(path,np.hstack(panels))
                reviews.append(dict(video=job['video'],slot=slot,frame=index,reasons=reasons,image=str(path)))
        finally:reader.close()
        print(f"Compared {Path(job['video']).name}; {len(points)} review images",flush=True)
    csv_write(out/'phase_comparison.csv',phases,list(phases[0]))
    csv_write(out/'tracking_comparison.csv',totals,list(totals[0]))
    csv_write(out/'measurement_changes.csv',changes,list(changes[0]))
    if pair_rows:csv_write(out/'pairs.csv',pair_rows,list(pair_rows[0]))
    json_write(out/'summary.json',dict(producer=producer(),input_status=identity(Path(args.status),content_hash=True),
        tracking=totals,phases=phases,measurement_changes=changes,review_points=reviews,
        interpretation='Raw missingness and diagnostic jumps, not accuracy labels. Phase windows partition saved frames; phase pair counts require all endpoints inside the window.'))
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':main()
