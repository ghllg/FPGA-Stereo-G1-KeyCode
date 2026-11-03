"""Train on subjects 1-3; select a checkpoint using only subject 4."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .geometry import BONES
from .model import PoseMLP
from mocap.io import write_json


def load_split(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in ('x', 'base', 'xyz', 'observed')}


def train(data_dir, output, epochs=60, batch_size=512, threads=4, seed=20261013, single_view_ray=False, reject_conflicting_view=False):
    if min(epochs, batch_size, threads) < 1:
        raise ValueError('epochs, batch_size and threads must be positive.')
    data_dir, output = Path(data_dir), Path(output)
    manifest = json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    manifest_sha256 = hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest()
    if manifest.get('state') != 'complete':
        raise ValueError('Dataset has not finished preparation.')
    if reject_conflicting_view and (not single_view_ray or manifest.get('feature_policy')!='geometry_candidate_v1'):
        raise ValueError('Conflicting-view policy requires the validated candidate and ray policies.')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    data, val = load_split(data_dir/'train.npz'), load_split(data_dir/'validation.npz')
    model = PoseMLP()
    model.mean.copy_(data['x'].mean(dim=0))
    model.scale.copy_(data['x'].std(dim=0).clamp(min=.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    bones = torch.from_numpy(BONES)
    history, best, best_epoch = [], float('inf'), 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(data['x']))
        total = 0.
        for index in order.split(batch_size):
            pred = model(data['x'][index], data['base'][index])
            target = data['xyz'][index]
            weight = torch.where(data['observed'][index], 1., 3.)
            position_loss = (F.smooth_l1_loss(pred, target, reduction='none', beta=.05).mean(dim=-1)*weight).mean()
            pred_bones = (pred[:, bones[:, 0]]-pred[:, bones[:, 1]]).norm(dim=-1)
            target_bones = (target[:, bones[:, 0]]-target[:, bones[:, 1]]).norm(dim=-1)
            loss = position_loss + .05*F.smooth_l1_loss(pred_bones, target_bones, beta=.05)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(index)
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model(x, b) for x, b in zip(val['x'].split(2048), val['base'].split(2048))])
            error = (predictions-val['xyz']).norm(dim=-1)
            missing = ~val['observed']
            score = float(error[missing].mean())
            observed_score = float(error[val['observed']].mean())
        row = dict(epoch=epoch, training_loss=total/len(order), validation_missing_mpjpe_m=score,
                   validation_observed_mpjpe_m=observed_score, elapsed_s=time.perf_counter()-started)
        history.append(row)
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(schema='pose3d_mlp/1.0', state_dict=model.state_dict(), epoch=epoch,
                            calibration_hash=manifest['calibration_hash'], calibration_id=manifest['calibration_id'],
                            data_archive_sha256=manifest['archive_sha256'], seed=seed, joints=manifest['joints'],
                            data_manifest_sha256=manifest_sha256, training_roi_xywh=manifest.get('roi_xywh'),
                            training_xy_bounds_m=manifest['xy_bounds_m'],
                            provisional_calibration=manifest.get('provisional_calibration', False),
                            feature_policy=manifest.get('feature_policy'),single_hip_anchor=manifest.get('single_hip_anchor',False),
                            single_view_ray=single_view_ray,
                            reject_conflicting_view=reject_conflicting_view,
                            calibration_accepted_for_training=manifest.get('calibration_accepted_for_training'),
                            validation_missing_mpjpe_m=best, inference_policy='preserve_observed_complete_missing'), output/'checkpoint.pt')
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
        write_json(output/'history.json', history)
    report = dict(schema='pose3d_training_run/1.0', state='complete', device='cpu', threads=threads,
                  seed=seed, epochs=epochs, best_epoch=best_epoch, batch_size=batch_size,
                  single_view_ray=single_view_ray,
                  reject_conflicting_view=reject_conflicting_view,
                  parameters=sum(p.numel() for p in model.parameters()), dataset=str(data_dir.resolve()),
                  data_manifest_sha256=manifest_sha256, training_roi_xywh=manifest.get('roi_xywh'),
                  training_xy_bounds_m=manifest['xy_bounds_m'],
                  provisional_calibration=manifest.get('provisional_calibration', False),
                  calibration_accepted_for_training=manifest.get('calibration_accepted_for_training'),
                  validation_missing_mpjpe_m=best, elapsed_s=time.perf_counter()-started,
                  test_set_used_for_selection=False)
    write_json(output/'training_report.json', report)
    write_json(output/'normalization.json', dict(mean=model.mean.tolist(), scale=model.scale.tolist(),
                                                calibration_hash=manifest['calibration_hash']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--single-view-ray', action='store_true')
    parser.add_argument('--reject-conflicting-view', action='store_true')
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.threads, single_view_ray=args.single_view_ray,
          reject_conflicting_view=args.reject_conflicting_view)
