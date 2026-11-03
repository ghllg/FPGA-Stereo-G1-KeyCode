"""LAFAN1 BVH -> calibrated synthetic stereo, split by subject before augmentation."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from .bvh import read_bvh
from .geometry import Stereo, BODY_NAMES, LAFAN_BODY
from mocap.io import write_json


def split_for(name):
    subject = int(re.search(r'_subject([1-5])\.bvh$', name)[1])
    return 'train' if subject <= 3 else ('validation' if subject == 4 else 'test')


def roi_visible(pixels, roi_xywh):
    x, y, width, height = roi_xywh
    return ((pixels[..., 0] >= x) & (pixels[..., 0] < x + width)
            & (pixels[..., 1] >= y) & (pixels[..., 1] < y + height))


def completion_variants(arrays, stereo, rng):
    """Append single-hip occlusions within the already assigned subject split."""
    indices = np.flatnonzero(rng.random(len(arrays['x'])) < .4)
    extra = {key: value[indices].copy() for key, value in arrays.items()}
    for index, hip in enumerate(rng.integers(6, 8, len(indices))):
        extra['confidence'][index, :, hip] = 0.
    joined = {key: np.concatenate([value, extra[key]]) for key, value in arrays.items()}
    joined['x'], joined['base'], joined['observed'], _ = stereo.features(
        joined['pixels'], joined['confidence'], candidate_base=True)
    keep = joined['observed'][:, 6:8].any(axis=1) & (joined['observed'].sum(axis=1) >= 6)
    return {key: value[keep] for key, value in joined.items()}


def prepare(archive, calibration, output, per_sequence=1000, seed=20261013,
            xy_bounds=(-.45, .85, .5, 2.8), allow_provisional_calibration=False, candidate_one_hip=False,
            roi_xywh=None):
    if per_sequence < 1:
        raise ValueError('per_sequence must be positive.')
    if (len(xy_bounds) != 4 or not np.isfinite(xy_bounds).all()
            or xy_bounds[0] >= xy_bounds[1] or xy_bounds[2] >= xy_bounds[3]):
        raise ValueError('xy_bounds must be finite xmin < xmax, ymin < ymax in world metres.')
    stereo = Stereo(calibration)
    if roi_xywh is not None:
        width, height = stereo.calibration['image_size']
        if (len(roi_xywh) != 4 or not np.isfinite(roi_xywh).all()
                or min(roi_xywh[:2]) < 0 or min(roi_xywh[2:]) <= 0
                or roi_xywh[0] + roi_xywh[2] > width or roi_xywh[1] + roi_xywh[3] > height):
            raise ValueError('roi_xywh must be a positive rectangle inside the source image.')
    provisional = (stereo.calibration.get('diagnostic_only', False)
                   or stereo.calibration.get('accepted_for_training') is False)
    if provisional and not allow_provisional_calibration:
        raise ValueError('Provisional calibration requires explicit experimental training opt-in.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    completion_rng = np.random.default_rng(seed + 100004)
    groups = {name: [] for name in ('train', 'validation', 'test')}
    manifest = dict(schema='pose3d_training_data/1.0', state='incomplete', seed=seed,
                    archive=str(Path(archive).resolve()), archive_sha256='',
                    calibration_id=stereo.calibration['calibration_id'], calibration_hash=stereo.hash,
                    provisional_calibration=bool(provisional),
                    calibration_accepted_for_training=stereo.calibration.get('accepted_for_training'),
                    world_reference=stereo.calibration.get('world_reference'),
                    xy_bounds_m=list(xy_bounds),
                    coordinate_system=stereo.calibration['world_coordinate_system'], unit='m',
                    joints=list(BODY_NAMES), mapping=dict(zip(BODY_NAMES, LAFAN_BODY)),
                    subjects={'train': [1, 2, 3], 'validation': [4], 'test': [5]}, sequences=[],
                    noise={'pixel_sigma_range': [1., 5.], 'per_view_dropout': .08, 'both_view_dropout': .05},
                    limitations=['Synthetic projections, not real camera recordings.',
                                 'No facial landmarks are supervised or invented.',
                                 'Fixed calibration; augmentation changes body position/yaw/scale, not camera geometry.',
                                 'Independent frame augmentation; this dataset must not train temporal losses.',
                                 'Both hips and at least six stereo body joints must be reliable.'])
    if candidate_one_hip:
        manifest.update(feature_policy='geometry_candidate_v1', single_hip_anchor=True,
                        completion_augmentation=dict(one_hip_variant_fraction=.4, seed=seed+100004))
        manifest['limitations'][-1] = 'One reliable hip and at least six stereo body joints required for completion.'
    if roi_xywh is not None:
        manifest['roi_xywh'] = list(roi_xywh)
        manifest['roi_visibility_policy'] = 'Zero confidence if clean or noisy projection lies outside the ROI; keep source pixel coordinates.'
    with Path(archive).open('rb') as file:
        manifest['archive_sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
    write_json(output/'manifest.json', manifest)
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.lower().endswith('.bvh'))
        for seq, name in enumerate(members):
            names, parents, pos, dt = read_bvh(z.read(name).decode('utf-8'))
            if len(names) != 22 or dt < .03 or dt > .034:
                raise ValueError(f'Unexpected LAFAN1 skeleton/frame rate: {name}')
            indices = np.linspace(0, len(pos)-1, min(len(pos), per_sequence), dtype=int)
            body = pos[indices][:, [names.index(n) for n in LAFAN_BODY]].copy()
            # BVH is Y-up centimetres; reader already converted centimetres to metres.
            floor = np.percentile(pos[:, [names.index('LeftFoot'), names.index('RightFoot')], 1], 1)
            body[..., 1] -= floor
            body[..., [0, 2]] -= pos[indices, names.index('Hips')][:, None, [0, 2]]
            world = body[..., [2, 0, 1]]
            count = len(world)
            yaw = rng.uniform(-np.pi, np.pi, count)
            c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            x, y = world[..., 0].copy(), world[..., 1].copy()
            world[..., 0], world[..., 1] = c*x-s*y, s*x+c*y
            world *= rng.uniform(.8, 1.15, (count, 1, 1))
            world[..., 0] += rng.uniform(*xy_bounds[:2], (count, 1))
            world[..., 1] += rng.uniform(*xy_bounds[2:], (count, 1))
            clean, depth = stereo.project(world)
            width, height = stereo.calibration['image_size']
            in_frame = ((depth > .3).all(axis=(1, 2)) & (clean[..., 0] >= 8).all(axis=(1, 2))
                        & (clean[..., 0] < width-8).all(axis=(1, 2)) & (clean[..., 1] >= 8).all(axis=(1, 2))
                        & (clean[..., 1] < height-8).all(axis=(1, 2)))
            sigma = rng.uniform(1., 5., (count, 2, 1, 1))
            pixels = clean + rng.normal(size=clean.shape)*sigma
            confidence = rng.uniform(.75, 1., clean.shape[:-1])
            missing = rng.random(confidence.shape) < .08
            missing |= (rng.random((count, 1, 12)) < .05)
            missing[:, :, 6:8] = False
            confidence[missing] = 0.
            if roi_xywh is not None:
                clean, pixels = clean.astype(np.float32), pixels.astype(np.float32)
                visible_in_roi = roi_visible(clean, roi_xywh) & roi_visible(pixels, roi_xywh)
                confidence[~visible_in_roi] = 0.
            features, base, observed, _ = stereo.features(pixels, confidence)
            keep = in_frame & observed[:, 6:8].all(axis=1) & (observed.sum(axis=1) >= 6)
            split = split_for(name)
            arrays = dict(x=features, base=base, xyz=world.astype(np.float32), observed=observed,
                          pixels=pixels.astype(np.float32), confidence=confidence.astype(np.float32),
                          clean_pixels=clean.astype(np.float32),
                          sequence=np.full(count, seq, dtype=np.int16), source_frame=indices.astype(np.int32))
            if roi_xywh is not None:
                arrays['roi_visible'] = visible_in_roi
            retained = {key: value[keep] for key, value in arrays.items()}
            if candidate_one_hip:
                retained = completion_variants(retained, stereo, completion_rng)
            groups[split].append(retained)
            manifest['sequences'].append(dict(name=name, split=split, source_frames=len(pos),
                                              sampled=count, retained=len(retained['x']), frame_time_s=dt))
            print(f'{seq+1}/{len(members)} {Path(name).name}: {keep.sum()}/{count} -> {split}', flush=True)
    for split, parts in groups.items():
        joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        if len(joined['x']) == 0:
            raise ValueError(f'No usable samples for {split}.')
        np.savez_compressed(output/f'{split}.npz', **joined)
    manifest['counts'] = {split: sum(len(part['x']) for part in parts) for split, parts in groups.items()}
    manifest['state'] = 'complete'
    write_json(output/'manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-sequence', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--xy-bounds', type=float, nargs=4, default=(-.45, .85, .5, 2.8),
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    parser.add_argument('--allow-provisional-calibration', action='store_true',
                        help='Experimental training only; does not mark calibration as accepted.')
    parser.add_argument('--candidate-one-hip', action='store_true',
                        help='Use bounded triangulation candidates and train single-hip occlusion variants.')
    parser.add_argument('--roi-xywh', type=int, nargs=4,
                        help='Fixed detector crop in source pixels, applied to both cameras.')
    args = parser.parse_args()
    prepare(args.archive, args.calibration, args.output, args.per_sequence, args.seed,
            args.xy_bounds, args.allow_provisional_calibration, args.candidate_one_hip, args.roi_xywh)
