"""Auditable saved FPGA heatmaps -> DARK -> stereo completion -> checked BVH.

Two passes of the same PP weights: fixed initial crop, then pose-guided crop.
Inference outputs must already exist; this entry never opens a camera or UART.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from mocap.io import COCO17, write_json
from mocap.ensemble import fuse_pose
from .geometry import Stereo
from .heatmap_decode import dark_decode, guided_roi
from .pipeline import replay


def read_points(directory, name):
    path = Path(directory)/(name+'.json')
    record = json.loads(path.read_text())
    if (record.get('backend') != 'pl-stream' or record.get('pl_acceleration') is not True
            or record.get('source_size') != [1280,720] or record.get('keypoint_order') != 'COCO17'
            or record.get('coordinate_space') != 'source_image_pixels'
            or record.get('stream_dispatches') != 110):
        raise ValueError('Expected original full-network FPGA inference: '+str(path))
    heat_path = Path(str(path)+'.heatmap')
    raw = heat_path.read_bytes()
    if len(raw) != 17*32*24*4:
        raise ValueError('Truncated heatmap: '+str(heat_path))
    heat = np.frombuffer(raw, dtype='<f4').reshape(17,32,24)
    return dark_decode(heat, record['roi_xywh']), record, hashlib.sha256(raw).hexdigest()


def extract(source, roi_dir, guided_dir, binding, calibration, output, start_index=6):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory.')
    stereo = Stereo(calibration)
    provenance = json.loads(Path(binding).read_text())
    if not provenance.get('hardware_accessed') or not provenance.get('boot_id') or not provenance.get('original_model_sha256'):
        raise ValueError('Missing hardware and weight binding.')
    records = {}
    for side in ('left','right'):
        with (source/(side+'.csv')).open(encoding='utf-8-sig') as stream:
            records[side] = [{k:int(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        times = [row['source_timestamp_us'] for row in records[side]]
        if len(times)<2 or np.any(np.diff(times)<=0):
            raise ValueError('Source timestamps must strictly increase.')
    output.mkdir(parents=True)
    alignment = dict(method='offline_right_bracket_linear_interpolation', right_timestamp_subtract_s=0.,
                     maximum_bracket_gap_ms=75., hardware_sync=False, exposure_sync_verified=False)
    model_id = 'PP-TinyPose:'+provenance['original_model_sha256'][:16]+':pl-stream:DARK'
    meta = dict(schema='pose2d_manifest/1.0', state='incomplete', source_session=str(source.resolve()),
        camera_ids=stereo.calibration['camera_ids'], joint_names=list(COCO17),
        clock_id='saved_board_source_timestamp_us:'+source.name, unit='original_image_pixels', mirrored=False,
        model_id=model_id, model_sha256=provenance['original_model_sha256'],
        calibration_id=stereo.calibration['calibration_id'], calibration_hash=stereo.hash,
        temporal_alignment=alignment, source_role='actual FPGA inference on saved JPEGs; PC DARK decoding',
        source_2d_backend='pl-stream', hardware_inference_binding=provenance, live_acquisition_validated=False,
        exposure_sync_verified=False, frame_selection=dict(first_left_index=start_index,person_filter=False),
        quality_claim='Development-clip replay; no independent real 3D ground truth.')
    roi_points = {}
    targets = {}
    passes = [('roi',roi_dir)] + ([('guided',guided_dir)] if guided_dir is not None else [])
    for label, directory in passes:
        predictions = {}
        for side in ('left','right'):
            for record in records[side]:
                name = f'{side}_{record["capture_index"]}'
                points, inference, heat_hash = read_points(directory, name)
                if label == 'roi':
                    if inference['roi_xywh'] != [435,180,405,540]:
                        raise ValueError('Initial crop changed.')
                    roi_points[name] = points
                elif inference['roi_xywh'] != guided_roi(roi_points[name]):
                    raise ValueError('Guided crop does not match first-pass pose: '+name)
                image_path = source/(name+'.jpg')
                predictions[side,record['capture_index']] = dict(camera_id=side,image_size=[1280,720],
                    frame_index=record['capture_index'],timestamp_ns=record['source_timestamp_us']*1000,
                    source_image=str(image_path.resolve()),source_image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    source_record=record,roi_xywh=inference['roi_xywh'],heatmap_sha256=heat_hash,
                    board_forward_ms=inference['infer_ms'],decode_method='dark',joints=[dict(xy=[float(x),float(y)],
                        confidence=float(np.clip(score,0,1)),raw_heatmap_score=float(score),
                        valid=bool(score>=.35 and np.isfinite([x,y]).all() and 0<=x<1280 and 0<=y<720)) for x,y,score in points])
        times = np.array([r['source_timestamp_us']*1000 for r in records['right']],np.int64)
        rows, skipped = [], []
        for record in records['left']:
            if record['capture_index']<start_index:
                continue
            timestamp=record['source_timestamp_us']*1000
            index=int(np.searchsorted(times,timestamp))
            if index==0 or index==len(times) or times[index]-times[index-1]>75_000_000:
                skipped.append(dict(left_frame=record['capture_index'],reason='no_right_bracket_within_75ms'))
                continue
            ends=[predictions['right',records['right'][i]['capture_index']] for i in (index-1,index)]
            alpha=float((timestamp-times[index-1])/(times[index]-times[index-1]))
            right=dict(camera_id='right',image_size=[1280,720],frame_index=ends[0]['frame_index'],timestamp_ns=timestamp,
                interpolation_source_views=ends,interpolation_alpha=alpha,joints=[])
            for a,b in zip(ends[0]['joints'],ends[1]['joints']):
                valid=a['valid'] and b['valid']
                right['joints'].append(dict(xy=((1-alpha)*np.array(a['xy'])+alpha*np.array(b['xy'])).tolist() if valid else None,
                    confidence=min(a['confidence'],b['confidence']),valid=valid,estimated=True))
            aligned=dict(alignment,right_source_frames=[e['frame_index'] for e in ends],
                right_original_timestamps_ns=[e['timestamp_ns'] for e in ends],right_interpolation_alpha=alpha)
            rows.append(dict(schema='pose2d/1.0',seq=len(rows),timestamp_ns=timestamp,model_id=model_id,joint_names=list(COCO17),
                pair_delta_ms=0.,temporal_alignment=aligned,views=[predictions['left',record['capture_index']],right]))
        if not rows:
            raise ValueError('No usable source pairs.')
        target=output/(label+'.jsonl');targets[label]=target
        target.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
        write_json(target.with_suffix('.meta.json'),dict(meta,state='complete',rows=len(rows),skipped=skipped,crop_method=label))
    if guided_dir is None:
        return targets['roi']
    fused=output/'pose2d.jsonl'
    fuse_pose(targets['guided'],targets['roi'],calibration,fused)
    return fused


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','roi-dir','guided-dir','binding','calibration','checkpoint','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--start-index',type=int,default=6)
    args=parser.parse_args()
    pose=extract(args.source,args.roi_dir,args.guided_dir,args.binding,args.calibration,args.output,args.start_index)
    replay(pose,args.calibration,args.checkpoint,Path(args.output)/'replay')
