"""Evaluate frame-local body-core segmentation on explicit saved-frame intervals.

Plan entries: audit, label, slots, intervals [start, stop), snapshots; optional
baseline_companion for parity checks. Baseline is normalized + global_offset.
Includes a centroid-only diagnostic ablation on unchanged accepted components.
Outputs interval measurements, not full-length .trx files or accuracy labels.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from nvsl_offline.detection import DetectionConfig, detect
from nvsl_offline.geometry import weighted_body_ellipse
from nvsl_offline.jumps import jump_pairs
from nvsl_offline.pipeline import assert_sources, create_run, csv_write, load_run, producer, write_image
from nvsl_offline.recording import FIELDS, identity, json_write, load_pickle
from nvsl_offline.replay import FrameReader


def values(ellipse):
    if ellipse is None:
        return np.full(5, np.nan)
    return np.array([*ellipse[0], *ellipse[1], ellipse[2]])


def centroid_ablation(result, residual, mask):
    if result.ellipse is None:
        return None
    contour = result.candidates[0].contour
    x, y, w, h = cv2.boundingRect(contour)
    component = np.zeros((h, w), np.uint8)
    cv2.drawContours(component, [contour-np.array([[[x,y]]])], -1, 1, -1)
    component &= (mask[y:y+h, x:x+w]>0).astype(np.uint8)
    return weighted_body_ellipse(component, residual[y:y+h, x:x+w], (x,y))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--output',required=True)
    parser.add_argument('--core-radius',type=float,default=2.)
    parser.add_argument('--core-growth',type=int,default=2)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    base=DetectionConfig(mode='normalized',illumination='global_offset')
    core=replace(base,segmentation='body_core',core_radius=args.core_radius,core_growth=args.core_growth)
    core.validate()
    out=create_run(args.output)
    report=dict(producer=producer(),plan=plan,baseline_config=asdict(base),core_config=asdict(core),recordings=[],
        interpretation='Development intervals; raw diagnostic pairs and availability are not localization accuracy. Centroid-only is a geometry ablation with baseline acceptance unchanged.')
    start=time.monotonic()
    for item in plan['recordings']:
        rec,manifest,bg,cal=load_run(item['audit'])
        saved=load_pickle(Path(item['baseline_companion'])) if item.get('baseline_companion') else None
        if saved is not None:
            stored=DetectionConfig(**saved['tracking_metadata']['configuration']['detector'])
            if stored != base:
                raise ValueError('Stored parity companion differs from baseline configuration')
        directory=out/item['label'];directory.mkdir()
        reader=FrameReader(rec.video)
        rows,summaries=[],[]
        parity_error=0.
        try:
            for first,stop in item['intervals']:
                if not 0<=first<stop<=rec.shape[1]:
                    raise ValueError('Invalid saved-frame interval')
                arrays={name:np.full((4,stop-first,5),np.nan) for name in ['baseline','body_core','centroid_only']}
                for index in range(first,stop):
                    image=reader.read(index)
                    old,residual,mask=detect(image,bg,cal,base)
                    new,_,_=detect(image,bg,cal,core)
                    for slot in range(4):
                        arrays['baseline'][slot,index-first]=values(old[slot].ellipse)
                        arrays['body_core'][slot,index-first]=values(new[slot].ellipse)
                        centroid=centroid_ablation(old[slot],residual,mask)
                        arrays['centroid_only'][slot,index-first]=values(centroid)
                        if saved is not None:
                            expected=np.array([saved[k][slot,index] for k in FIELDS])
                            np.testing.assert_allclose(arrays['baseline'][slot,index-first],expected,
                                                       atol=1e-3,rtol=1e-5,equal_nan=True)
                            if np.isfinite(expected).all():
                                parity_error=max(parity_error,float(np.max(np.abs(expected-arrays['baseline'][slot,index-first]))))
                        for name,result in [('baseline',old[slot]),('body_core',new[slot])]:
                            selected=result.candidates[0] if result.ellipse is not None else None
                            rows.append(dict(frame=index,slot=slot,variant=name,reason=result.reason,
                                candidate_count=len(result.candidates),
                                core_pixels=int(selected.core_mask.sum()) if selected is not None and selected.core_mask is not None else 0,
                                **dict(zip(FIELDS,arrays[name][slot,index-first]))))
                        if index in item.get('snapshots',[]) and slot in item['slots']:
                            x0,y0,x1,y1=cal.roi(slot)
                            panels=[]
                            for ellipse,title,color in [(old[slot].ellipse,'baseline: '+old[slot].reason,(80,255,80)),
                                (centroid,'centroid only',(0,220,255)),(new[slot].ellipse,'body core: '+new[slot].reason,(255,100,255))]:
                                panel=image.copy()
                                if ellipse is not None:cv2.ellipse(panel,ellipse,color,1)
                                panel=panel[y0:y1,x0:x1]
                                panel=cv2.copyMakeBorder(panel,40,0,0,0,cv2.BORDER_CONSTANT)
                                cv2.putText(panel,title,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                                cv2.putText(panel,f'frame {index}; slot {slot}',(3,32),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                                panels.append(panel)
                            write_image(directory/f'frame{index:07d}-slot{slot}.png',np.hstack(panels))
                    if index%500==0:
                        print(f"{item['label']}: {index}/{stop}",flush=True)
                for slot in range(4):
                    summary=dict(start=first,stop=stop,slot=slot)
                    for name,a in arrays.items():
                        a=a[slot]
                        _,pairs=jump_pairs(a[:,0],a[:,1])
                        summary[name]=dict(missing_frames=int(np.isnan(a[:,0]).sum()),
                            raw_pairs=[dict(p,first_jump=p['first_jump']+first,second_jump=p['second_jump']+first) for p in pairs])
                    old,new=arrays['baseline'][slot],arrays['body_core'][slot]
                    common=np.isfinite(old[:,0])&np.isfinite(new[:,0])
                    movement=np.hypot(old[common,0]-new[common,0],old[common,1]-new[common,1])
                    summary.update(newly_missing=int((np.isfinite(old[:,0])&np.isnan(new[:,0])).sum()),
                        newly_finite=int((np.isnan(old[:,0])&np.isfinite(new[:,0])).sum()),
                        center_difference_median=float(np.median(movement)) if len(movement) else None,
                        center_difference_max=float(movement.max()) if len(movement) else None)
                    summaries.append(summary)
                np.savez_compressed(directory/f'frames-{first}-{stop}.npz',frames=np.arange(first,stop),**arrays)
                print(f"{item['label']}: completed {first}:{stop}",flush=True)
        finally:
            reader.close()
        assert_sources(manifest['sources'])
        csv_write(directory/'measurements.csv',rows,list(rows[0]))
        report['recordings'].append(dict(label=item['label'],video=str(rec.video),audit=str(Path(item['audit']).resolve()),
            sources=manifest['sources'],baseline_companion=identity(Path(item['baseline_companion']),content_hash=True) if saved is not None else None,
            baseline_reproduction_max_error=parity_error if saved is not None else None,intervals=summaries))
    report['seconds']=time.monotonic()-start
    if producer()['package_sha256']!=report['producer']['package_sha256']:
        raise ValueError('Tracker source changed during experiment')
    json_write(out/'summary.json',report)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
