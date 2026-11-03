"""Prepare real recorded stereo images for board inference, without a pose model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import cv2

REPO=Path(__file__).resolve().parents[3]
U02=REPO/'ultrascale_plus/u02'
sys.path.insert(0,str(REPO/'windows_camera'))
sys.path.insert(0,str(REPO/'windows_vision'))
from training_data import iter_training_pairs
from mocap.calibration import load_calibration

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('calibration',type=Path)
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--every',type=int,default=120)
    args=parser.parse_args()
    assert 1<=args.count<=30 and args.every>=1
    session=args.session.resolve()
    metadata=json.loads((session/'session.json').read_text(encoding='utf-8'))
    calibration=load_calibration(args.calibration,meta=metadata)
    output=U02/('reports/stereo_recorded_inputs_'+str(time.time_ns()))
    output.mkdir()
    files=['session.json','pairs.csv','left_timestamps.csv','right_timestamps.csv','left.mkv','right.mkv']
    result=dict(complete=False,input_mode='recorded',source_session=str(session),
        source_sha256={name:sha(session/name) for name in files},
        calibration_sha256=sha(args.calibration),camera_ids=calibration['camera_ids'],
        source_size=calibration['image_size'],hardware_sync=False,exposure_timestamp_verified=False,
        timestamp_source=metadata['timestamp_source_requested'],
        timestamp_fallback_note=metadata['timestamp_fallback_note'],
        clock_id='windows_dshow:'+session.name,
        calibration_physical_mount_verified=False,pose_model_on_pc=False,images=[])
    try:
        for index,pair in enumerate(iter_training_pairs(session,max_delta_ms=10)):
            if index%args.every:continue
            number=len(result['images'])//2
            for side in range(2):
                name=('left','right')[side]
                rgb=pair[name+'_rgb']
                assert list(rgb.shape[1::-1])==calibration['image_size']
                filename=f'p{number}_{name}.jpg'
                ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                assert ok
                (output/filename).write_bytes(encoded.tobytes())
                # Fixed whole-body ROIs for this recorded take, no detector.
                roi=[750,100,675,900] if side==0 else [650,50,675,900]
                result['images'].append(dict(file=filename,sha256=sha(output/filename),
                    pair_id=number,camera_id=result['camera_ids'][side],
                    frame_id=pair[name+'_frame_index'],capture_timestamp_ns=pair[name+'_source_ns'],
                    source_size=calibration['image_size'],roi_xywh=roi,
                    roi_source='explicit_fixed_recorded_take_crop',
                    raw_pair_delta_ms=pair['right_minus_left_ms']))
            print('REAL_STEREO_PAIR_PREPARED',number,pair['left_frame_index'],pair['right_frame_index'],flush=True)
            if number+1>=args.count:break
        assert len(result['images'])==args.count*2
        shutil.copyfile(args.calibration,output/'calibration.json')
        shutil.copyfile(session/'session.json',output/'source_session.json')
        result.update(complete=True,scope='Actual raw timestamp-matched recorded images only; no board inference or exposure synchronization proof. Historical pose interpolation/constant offset not applied.')
    finally:
        (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('STEREO_INPUT_RECORD',output,flush=True)

if __name__=='__main__':main()
