"""Compare exact original frames against existing PC YOLO results, not GT."""
import argparse
import json
import math
from pathlib import Path
import statistics
from run_recorded_board import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board_record',type=Path)
    parser.add_argument('pc_pose',type=Path)
    args=parser.parse_args();out=args.board_record.resolve()
    assert not (out/'pc_pose_comparison.json').exists()
    meta=json.loads(args.pc_pose.with_suffix('.meta.json').read_text(encoding='utf-8'))
    proof=json.loads((out/'recovered_result.json').read_text(encoding='utf-8'))
    inputs=json.loads((Path(proof['inputs'])/'manifest.json').read_text(encoding='utf-8'))
    assert meta['state']=='complete' and Path(meta['source_session']).resolve()==Path(inputs['source_session']).resolve()
    assert not meta.get('temporal_alignment'), 'Use original frames, not interpolated poses'
    lookup={}
    for line in args.pc_pose.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        for view in row['views']:
            key=(view['camera_id'],view['frame_index'])
            assert key not in lookup
            assert not view.get('temporal_alignment') and not any(j.get('estimated') for j in view['joints'])
            lookup[key]=view
    packets=[json.loads(line) for line in (out/'board_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    errors={mode:[] for mode in ('cpu','pl-stream')};details=[]
    for packet,image in zip(packets,inputs['images']):
        view=lookup[(packet['camera_id'],packet['frame_id'])]
        assert view['timestamp_ns']==packet['capture_timestamp_ns'] and view['image_size']==packet['source_size']
        suffix=image['file'][:-4]
        board={mode:json.loads((out/(mode.replace('-','_')+'_'+suffix+'.json')).read_text()) for mode in errors}
        for joint,reference in enumerate(view['joints']):
            if not reference['valid'] or reference['confidence']<.3:continue
            item=dict(file=image['file'],camera_id=packet['camera_id'],frame_id=packet['frame_id'],joint=joint,
                      pc_xy=reference['xy'],pc_confidence=reference['confidence'],backends={})
            for mode in errors:
                point=board[mode]['keypoints'][joint]
                distance=math.dist(reference['xy'],point[:2]);errors[mode].append(distance)
                item['backends'][mode]=dict(distance_px=distance,xy=point[:2],confidence=point[2])
            details.append(item)
    measurements={}
    for mode,values in errors.items():
        ordered=sorted(values)
        measurements[mode]=dict(joints=len(values),mean_px=statistics.mean(values),median_px=statistics.median(values),
            p95_px=ordered[math.ceil(.95*len(ordered))-1],max_px=max(values),over_30_px=sum(x>30 for x in values))
    result=dict(complete=True,exact_frame_timestamp_and_camera_matches=len(packets),
        pc_model_id=meta['model_id'],pc_pose_sha256=sha(args.pc_pose),pc_meta_sha256=sha(args.pc_pose.with_suffix('.meta.json')),
        board_pose_sha256=sha(out/'board_pose.jsonl'),reference_confidence_threshold=.3,
        unit='original_1920x1080_image_pixels',ground_truth=False,models_identical=False,
        measurements=measurements,details=details,
        scope='Same recorded physical camera/frame/timestamp, existing original PC YOLO versus actual board TinyPose. Different networks and ROI/postprocessing; no human GT, no precision acceptance or causal quantization attribution.')
    (out/'pc_pose_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(measurements,indent=2))

if __name__=='__main__':main()
