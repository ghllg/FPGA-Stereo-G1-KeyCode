"""Select real right video frames using a hash-bound offline content offset."""
import argparse
import bisect
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import av
import cv2
from prepare_recorded_pairs import REPO,U02,sha

sys.path.insert(0,str(REPO/'windows_camera'))
from training_data import VideoReader

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('alignment',type=Path)
    args=parser.parse_args();proof=json.loads(args.alignment.read_text(encoding='utf-8'))
    assert proof['complete'] and not proof['new_live_capture_applicable']
    previous=Path(proof['source_inputs']);manifest=json.loads((previous/'manifest.json').read_text(encoding='utf-8'))
    assert sha(previous/'manifest.json')==proof['source_manifest_sha256']
    assert sha(previous/'calibration.json')==proof['calibration_sha256']
    session=Path(proof['source_session'])
    assert all(sha(session/name)==digest for name,digest in proof['source_sha256'].items())
    with (session/'right_timestamps.csv').open(encoding='utf-8',newline='') as file:
        frames=list(csv.DictReader(file))
    stamps=[int(row['source_timestamp_ns']) for row in frames]
    assert all(b>a for a,b in zip(stamps,stamps[1:]))
    offset=proof['right_timestamp_subtract_ns'];assert type(offset) is int
    digest=sha(args.alignment)
    output=U02/('reports/stereo_content_inputs_'+str(time.time_ns()));output.mkdir()
    result=copy.deepcopy(manifest);result.update(complete=False,images=[],
        alignment_proof_sha256=digest,alignment_source=str(args.alignment),
        pairing_clock_id=manifest['clock_id']+':offline_content:'+digest[:16],
        pairing_method='offline_content_offset_nearest_actual_frame')
    previous_right=-1
    try:
        with av.open(str(session/'right.mkv')) as video:
            reader=VideoReader(video)
            for original in manifest['images'][::2]:
                left=copy.deepcopy(original)
                query=left['capture_timestamp_ns']+offset
                hi=bisect.bisect_left(stamps,query)
                choices=[i for i in (hi-1,hi) if 0<=i<len(stamps)]
                index=min(choices,key=lambda i:abs(stamps[i]-query))
                assert abs(stamps[index]-query)<=10_000_000
                frame=int(frames[index]['frame_index']);assert frame>previous_right
                previous_right=frame
                rgb=reader.at(frame)
                assert list(rgb.shape[1::-1])==manifest['source_size']
                right=copy.deepcopy(manifest['images'][left['pair_id']*2+1])
                right.update(frame_id=frame,capture_timestamp_ns=stamps[index])
                ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                (output/right['file']).write_bytes(encoded.tobytes());right['sha256']=sha(output/right['file'])
                shutil.copyfile(previous/left['file'],output/left['file']);assert sha(output/left['file'])==left['sha256']
                for image,correction in [(left,0),(right,offset)]:
                    image.update(pairing_timestamp_ns=image['capture_timestamp_ns']-correction,
                        pairing_clock_id=result['pairing_clock_id'],time_alignment=dict(
                            method=result['pairing_method'],proof_sha256=digest,offset_ns=correction))
                    image['raw_pair_delta_ms']=(right['capture_timestamp_ns']-left['capture_timestamp_ns'])/1e6
                    image['corrected_pair_delta_ms']=(right['capture_timestamp_ns']-offset-left['capture_timestamp_ns'])/1e6
                    result['images'].append(image)
                print('REAL_CONTENT_PAIR',left['frame_id'],right['frame_id'],left['corrected_pair_delta_ms'],flush=True)
        shutil.copyfile(previous/'calibration.json',output/'calibration.json')
        shutil.copyfile(previous/'source_session.json',output/'source_session.json')
        shutil.copyfile(args.alignment,output/'alignment.json')
        result.update(complete=True,scope='Actual frames selected using reproduced offline content offset; unchanged left JPEGs/ROIs/calibration. Raw timestamps retained, no 2D interpolation or PC neural inference, no exposure synchronization proof.')
    finally:
        (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('CONTENT_STEREO_INPUT_RECORD',output)

if __name__=='__main__':main()
