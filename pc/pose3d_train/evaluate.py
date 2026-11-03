"""Held-out synthetic truth evaluation; never interpret real pseudo labels as ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .geometry import Stereo, BONES
from .infer import complete_missing
from .model import load_model
from mocap.io import write_json


def metrics(prediction, truth, mask):
    errors = np.linalg.norm(prediction-truth, axis=-1)
    selected = errors[mask]
    bone_error = np.abs(np.linalg.norm(prediction[:, BONES[:, 0]]-prediction[:, BONES[:, 1]], axis=-1)
                        - np.linalg.norm(truth[:, BONES[:, 0]]-truth[:, BONES[:, 1]], axis=-1))
    bone_mask = mask[:, BONES[:, 0]] & mask[:, BONES[:, 1]]
    return dict(valid_joints=int(mask.sum()), total_joints=int(mask.size), coverage=float(mask.mean()),
                mpjpe_mm=float(selected.mean()*1000) if selected.size else None,
                p95_joint_error_mm=float(np.percentile(selected, 95)*1000) if selected.size else None,
                pck50_of_all_targets=float(((errors <= .05) & mask).mean()),
                mean_bone_length_error_mm=float(bone_error[bone_mask].mean()*1000) if bone_mask.any() else None)


def evaluate(data_dir, calibration, checkpoint, output):
    stereo = Stereo(calibration)
    model, saved = load_model(checkpoint)
    manifest = json.loads((Path(data_dir)/'manifest.json').read_text(encoding='utf-8'))
    if stereo.hash != saved['calibration_hash'] or stereo.hash != manifest['calibration_hash']:
        raise ValueError('Evaluation calibration does not match training.')
    if manifest.get('feature_policy')!=saved.get('feature_policy'):
        raise ValueError('Evaluation feature policy does not match training.')
    with np.load(Path(data_dir)/'test.npz', allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
    torch.set_num_threads(4)
    with torch.inference_mode():
        prediction = torch.cat([model(x, b) for x, b in zip(torch.from_numpy(a['x']).split(2048),
                                                           torch.from_numpy(a['base']).split(2048))]).numpy()
    hybrid, valid, completed = complete_missing(stereo, prediction, a['base'], a['observed'], a['pixels'], a['confidence'],
                                               single_hip_anchor=saved.get('single_hip_anchor',False),
                                               single_view_ray=saved.get('single_view_ray',False),
                                               conflict_model=model if saved.get('reject_conflicting_view',False) else None)
    _, clean_base, clean_observed, _ = stereo.features(a['clean_pixels'], np.ones_like(a['confidence']))
    all_mask = np.ones_like(a['observed'])
    report = dict(schema='pose3d_evaluation/1.0', truth='held-out LAFAN1 subject5; synthetic camera projections',
                  samples=len(prediction), calibration_hash=stereo.hash, checkpoint_epoch=saved['epoch'],
                  single_view_ray=saved.get('single_view_ray',False),
                  reject_conflicting_view=saved.get('reject_conflicting_view',False),
                  provisional_calibration=saved.get('provisional_calibration', False),
                  calibration_accepted_for_training=saved.get('calibration_accepted_for_training'),
                  clean_geometry=metrics(clean_base, a['xyz'], clean_observed),
                  noisy_geometry=metrics(a['base'], a['xyz'], a['observed']),
                  raw_model_all=metrics(prediction, a['xyz'], all_mask),
                  raw_model_observed=metrics(prediction, a['xyz'], a['observed']),
                  raw_model_missing=metrics(prediction, a['xyz'], ~a['observed']),
                  root_fill_missing_baseline=metrics(a['base'], a['xyz'], ~a['observed']),
                  hybrid=metrics(hybrid, a['xyz'], valid),
                  hybrid_completed=metrics(hybrid, a['xyz'], completed),
                  observed_joints_unchanged=bool(np.array_equal(hybrid[a['observed']], a['base'][a['observed']])),
                  real_accuracy_validated=False,
                  limitations=['Synthetic occlusion/noise is not a measured YOLO error distribution.',
                               'Only 12 body joints are evaluated; no facial or joint-twist ground truth.',
                               'Subject split prevents frame leakage; action classes may overlap.',
                               'Models and thresholds have not been validated for robot control.'])
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evaluate(args.data, args.calibration, args.checkpoint, args.output)
