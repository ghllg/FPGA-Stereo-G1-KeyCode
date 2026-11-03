"""Prepare/retrain the existing completion MLP only with verified 1080p geometry."""
import argparse
import hashlib
import json
from pathlib import Path

from .abi9_bridge import check_live_calibration
from .geometry import Stereo


def plan(calibration, archive):
    path=Path(calibration)
    cal=json.loads(path.read_text(encoding='utf-8-sig'))
    blockers=[]
    try:check_live_calibration(cal,10.)
    except ValueError as error:blockers.append(str(error))
    if cal.get('accepted_for_training') is not True:
        blockers.append('Calibration has not been accepted for training.')
    if 'diagnostic' in str(cal.get('quality',{}).get('acceptance','')).lower():
        blockers.append('Calibration quality still declares diagnostic use.')
    if not cal.get('camera_physical_ids') or len(cal['camera_physical_ids'])!=2:
        blockers.append('Physical left/right camera identities are missing.')
    archive=Path(archive)
    if not archive.is_file():blockers.append('Existing LAFAN1 archive is unavailable.')
    stereo=Stereo(path)
    return dict(profile='1080p_14p24_20261019',ready=not blockers,blockers=blockers,
                calibration=str(path.resolve()),calibration_hash=stereo.hash,
                calibration_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                archive=str(archive.resolve()),new_human_training_videos_required=0,
                epochs=180,batch_size=512,threads=4,seed=20261020,
                split={'train':[1,2,3],'validation':[4],'test':[5]},
                pixel_noise_sigma_range=[1.,5.],noise_scope='Initial synthetic noise in native 1080p pixels; not measured camera-error calibration.',
                observed_reprojection_limit_px=5.,completion_reprojection_limit_px=20.,
                test_set_used_for_selection=False)


def execute(calibration,archive,output):
    proposal=plan(calibration,archive)
    if not proposal['ready']:
        raise ValueError('Training blocked: '+'; '.join(proposal['blockers']))
    from .data_prepare import prepare
    from .train import train
    from .evaluate import evaluate
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_text(json.dumps(proposal,indent=2),encoding='utf8')
    prepare(archive,calibration,output/'data',seed=proposal['seed'],candidate_one_hip=True)
    train(output/'data',output/'model',epochs=180,batch_size=512,threads=4,
          seed=proposal['seed'],single_view_ray=True,reject_conflicting_view=True)
    return evaluate(output/'data',calibration,output/'model/checkpoint.pt',output/'evaluation.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:
        if not args.output:parser.error('--execute requires a new --output directory')
        result=execute(args.calibration,args.archive,args.output)
    else:
        result=plan(args.calibration,args.archive)
    print(json.dumps(result,ensure_ascii=False,indent=2))
