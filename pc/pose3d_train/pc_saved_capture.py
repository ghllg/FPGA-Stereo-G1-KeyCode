"""PC reference: saved board JPEG/CSV -> two existing YOLO candidates -> paired COCO17.

Never opens a camera. Right points are explicitly interpolated between recorded
exposures, not presented as hardware-synchronized measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mocap.io import COCO17, write_json
from mocap.pose import PoseEstimator
from mocap.ensemble import fuse_pose
from .geometry import Stereo


def extract(source, calibration, model, output):
    source, output, model = map(Path, (source, output, model))
    if output.exists():
        raise ValueError('Output exists; choose a new directory.')
    if not model.is_file():
        raise ValueError('Provide the saved reference weight file.')
    stereo = Stereo(calibration)
    records = []
    for side in ('left', 'right'):
        with (source/(side+'.csv')).open(encoding='utf-8-sig') as f:
            rows = [{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)]
        times = np.array([row['source_timestamp_us']*1000 for row in rows],np.int64)
        if len(times)<2 or np.any(np.diff(times)<=0):
            raise ValueError('Capture source timestamps must strictly increase.')
        records.append((rows,times))
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    metadata = dict(schema='pose2d_manifest/1.0',state='incomplete',source_session=str(source.resolve()),
        camera_ids=stereo.calibration['camera_ids'],joint_names=list(COCO17),
        clock_id='saved_board_source_timestamp_us:'+source.name,unit='original_image_pixels',mirrored=False,
        calibration_id=stereo.calibration['calibration_id'],calibration_hash=stereo.hash,
        source_role='offline PC reference; not board PP-TinyPose or PL output',
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        exposure_sync_verified=False,hardware_accessed=False,
        temporal_alignment=dict(method='offline_right_bracket_linear_interpolation',right_timestamp_subtract_s=0.,
                                maximum_bracket_gap_ms=75.,hardware_sync=False,exposure_sync_verified=False))
    files=[]
    for refine,label in [(True,'person'),(False,'full')]:
        estimator=PoseEstimator(model,device='cpu',refine_person=refine)
        metadata.update(model_id=estimator.model_id,person_roi_refinement=refine,state='incomplete')
        target=output/(label+'.jsonl');files.append(target)
        write_json(target.with_suffix('.meta.json'),metadata)
        predictions={}
        for side,(rows,times) in zip(('left','right'),records):
            for offset in range(0,len(rows),6):
                batch=rows[offset:offset+6];images=[];paths=[]
                for record in batch:
                    path=source/f'{side}_{record["capture_index"]}.jpg'
                    image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)
                    if image is None or [image.shape[1],image.shape[0]]!=stereo.calibration['image_size']:
                        raise ValueError('Missing, corrupt or wrong-resolution saved image: '+str(path))
                    images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));paths.append(path)
                for record,path,pred in zip(batch,paths,estimator.predict(images)):
                    pred.update(camera_id=side,frame_index=record['capture_index'],
                                timestamp_ns=record['source_timestamp_us']*1000,
                                source_image=str(path.resolve()),source_record=record,
                                source_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    predictions[side,record['capture_index']]=pred
                if offset%60==0:print(label,side,min(offset+6,len(rows)),len(rows),flush=True)
        write_json(output/(label+'_predictions.json'),list(predictions.values()))
        paired=[];skipped=[]
        for record,timestamp in zip(*records[0]):
            if record['capture_index']<6:continue
            index=int(np.searchsorted(records[1][1],timestamp))
            if index==0 or index==len(records[1][1]):continue
            lo,hi=records[1][0][index-1:index+1]
            times=records[1][1][index-1:index+1]
            if times[1]-times[0]>75_000_000:
                skipped.append(dict(left_frame=record['capture_index'],reason='right_bracket_gap',right_times_ns=times.tolist()))
                continue
            left=predictions['left',record['capture_index']]
            a,b=[predictions['right',r['capture_index']] for r in (lo,hi)]
            if any(v['person_count']!=1 for v in (left,a,b)):
                skipped.append(dict(left_frame=record['capture_index'],reason='person_count'))
                continue
            alpha=float((timestamp-times[0])/(times[1]-times[0]))
            alignment=dict(metadata['temporal_alignment'],right_source_frames=[lo['capture_index'],hi['capture_index']],
                           right_original_timestamps_ns=times.tolist(),right_interpolation_alpha=alpha)
            right=dict(camera_id='right',image_size=left['image_size'],frame_index=lo['capture_index'],
                       timestamp_ns=int(timestamp),interpolation_source_views=[a,b],interpolation_alpha=alpha,joints=[])
            for u,v in zip(a['joints'],b['joints']):
                valid=u['valid'] and v['valid']
                right['joints'].append(dict(xy=((1-alpha)*np.array(u['xy'])+alpha*np.array(v['xy'])).tolist() if valid else None,
                    confidence=min(u['confidence'],v['confidence']),valid=valid,estimated=True))
            paired.append(dict(schema='pose2d/1.0',seq=len(paired),joint_names=list(COCO17),model_id=estimator.model_id,
                timestamp_ns=int(timestamp),pair_delta_ms=0.,views=[left,right],temporal_alignment=alignment))
        if not paired:raise ValueError('No usable saved-image pairs.')
        target.write_text(''.join(json.dumps(row)+'\n' for row in paired),encoding='utf-8')
        metadata.update(state='complete',rows=len(paired),skipped=skipped)
        write_json(target.with_suffix('.meta.json'),metadata)
    # Both candidates use identical exposures and the same model. Keep all
    # existing consistency bounds; the final geometry gate remains five pixels.
    fuse_pose(*files,calibration,output/'pose2d.jsonl')
    meta=json.loads((output/'pose2d.meta.json').read_text(encoding='utf-8'))
    meta['pose_ensemble']['candidate_labels']={'full_frame':'primary person-refined YOLO', 'person_roi':'alternate full-image YOLO'}
    write_json(output/'pose2d.meta.json',meta)
    return output/'pose2d.jsonl'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','calibration','model','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    extract(args.source,args.calibration,args.model,args.output)
