"""Recheck existing content-offset evidence for this exact saved take only."""
import argparse
import json
from pathlib import Path
import sys
import time
from prepare_recorded_pairs import REPO,U02,sha

sys.path.insert(0,str(REPO/'windows_vision'))
from mocap.calibration import load_calibration
from mocap.temporal import estimate_offset

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('original_pc_pose',type=Path)
    args=parser.parse_args()
    inputs=args.inputs.resolve();manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    meta=json.loads(args.original_pc_pose.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert meta['state']=='complete' and not meta.get('temporal_alignment')
    assert Path(meta['source_session']).resolve()==Path(manifest['source_session']).resolve()
    session=Path(manifest['source_session'])
    assert all(sha(session/name)==digest for name,digest in manifest['source_sha256'].items())
    calibration=load_calibration(inputs/'calibration.json')
    assert meta['camera_ids']==calibration['camera_ids']
    rows=[json.loads(line) for line in args.original_pc_pose.read_text(encoding='utf-8').splitlines()]
    assert all(not row.get('temporal_alignment') for row in rows)
    offset,fit=estimate_offset(rows,calibration)
    output=U02/('reports/stereo_recorded_alignment_'+str(time.time_ns()));output.mkdir()
    result=dict(complete=True,source_session=str(session),source_sha256=manifest['source_sha256'],
        source_inputs=str(inputs),source_manifest_sha256=sha(inputs/'manifest.json'),
        calibration_sha256=sha(inputs/'calibration.json'),
        original_pc_pose_sha256=sha(args.original_pc_pose),reference_model_id=meta['model_id'],
        reused_alignment_source_sha256=sha(REPO/'windows_vision/mocap/temporal.py'),
        right_timestamp_subtract_ns=round(offset*1e9),fit=fit,
        image_pairing_action='Select nearest actual right video frame at left raw timestamp plus offset; do not interpolate 2D joints',
        exposure_sync_verified=False,new_live_capture_applicable=False,
        scope='Reproduced content/geometry offset using existing PC results, only for hash-bound saved take. No new PC pose inference, no accepted FPGA stereo or exposure synchronization.')
    (output/'alignment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(fit,indent=2));print('RECORDED_ALIGNMENT_RECORD',output)

if __name__=='__main__':main()
