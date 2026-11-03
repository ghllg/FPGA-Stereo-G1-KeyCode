"""Causal COCO17 -> existing recovered-BVH convention; reuse frozen repo code."""
from pathlib import Path
import sys
import warnings
import os

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
sys.path.insert(0, str(ROOT / 'upstream/windows_vision'))
from shared.skeleton_protocol import validate, COCO17
from mocap.lafan import NAMES, PARENTS, CORE, WORLD_TO_BVH, extend_coco, offsets_from_targets, fit_rotations

WORLD = 'world_right_handed_X_forward_Y_left_Z_up'
# Existing BVH decoder applies Rx(90). The upstream exporter uses (Y,Z,X),
# so a root-only Ry(90) cancels its otherwise extra horizontal -90deg heading.
ROOT_ALIGNMENT = Rotation.from_euler('Y', 90, degrees=True)


def support_relative_hip_height(points):
    # A lifted swing foot is not the support plane. Averaging ankle heights
    # mistakes a one-foot lift for a crouch and bends both robot knees.
    # This relative proxy preserves bilateral crouching; it is not ground truth.
    return float(np.mean(points[[11,12],2])-np.min(points[[15,16],2]))


def bvh_header(offsets):
    lines = ['HIERARCHY']
    def node(j, level):
        indent = '  ' * level
        lines.extend([indent + ('ROOT ' if j == 0 else 'JOINT ') + NAMES[j], indent + '{',
                      indent + '  OFFSET ' + ' '.join(f'{v*100:.8f}' for v in offsets[j]),
                      indent + '  ' + ('CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation'
                                        if j == 0 else 'CHANNELS 3 Zrotation Yrotation Xrotation')])
        children = np.flatnonzero(PARENTS == j)
        for child in children:
            node(int(child), level+1)
        if not len(children):
            lines.extend([indent+'  End Site', indent+'  {', indent+'    OFFSET 0 0 0', indent+'  }'])
        lines.append(indent + '}')
    node(0, 0)
    return '\n'.join(lines)


class LiveBVH:
    def __init__(self, warmup=10, warmup_seconds=0.):
        if type(warmup) is not int or warmup < 1:
            raise ValueError('warmup must be a positive number of valid frames')
        self.warmup = warmup
        if not np.isfinite(warmup_seconds) or not 0 <= warmup_seconds <= 5:
            raise ValueError('warmup_seconds must be finite and between 0 and 5')
        self.warmup_seconds = warmup_seconds
        self.prefix_start_ns = None
        self.warmup_frames_used = 0
        self.prefix = []
        self.offsets = self.header = self.identity = self.origin_ns = None
        self.previous_ns = self.previous_seq = -1
        self.torso_length = None
        self.received = self.invalid = self.emitted = 0

    def push(self, packet):
        validate(packet)
        if packet['coordinate_system'] != WORLD:
            raise ValueError('Require calibrated world X-forward/Y-left/Z-up, in metres')
        identity = tuple(packet[k] for k in ('session_id', 'clock_id', 'calibration_id', 'model_id'))
        if self.identity is not None and identity != self.identity:
            raise ValueError('Session/clock/calibration/model changed: start a new input session')
        stamp, seq = packet['timestamp_ns'], packet['seq']
        if stamp <= self.previous_ns or seq <= self.previous_seq:
            raise ValueError('Source timestamps and seq must strictly increase')
        self.identity = identity
        self.previous_ns, self.previous_seq = stamp, seq
        self.received += 1
        points = np.array([j['xyz'] if j['valid'] else [np.nan]*3 for j in packet['joints']])
        if not np.isfinite(points[list(CORE)]).all():
            self.invalid += 1
            return None
        try:
            targets, basis = extend_coco(points, self.torso_length)
        except ValueError:
            self.invalid += 1
            return None
        if self.offsets is None:
            if self.prefix_start_ns is None:
                self.prefix_start_ns = stamp
            self.prefix.append(targets)
            if len(self.prefix) < self.warmup or (stamp-self.prefix_start_ns)/1e9 < self.warmup_seconds:
                return None
            # Prefix only, frozen before first output. No later bone-length drift.
            self.offsets = offsets_from_targets(np.array(self.prefix))
            self.torso_length = float(np.percentile([np.linalg.norm(q[11]-q[0]) for q in self.prefix], 90))
            self.header = bvh_header(self.offsets)
            self.warmup_frames_used = len(self.prefix)
            self.prefix.clear()
            targets, basis = extend_coco(points, self.torso_length)
        rotations, fk = fit_rotations(targets, self.offsets, basis)
        rotations[0] = ROOT_ALIGNMENT.as_matrix() @ rotations[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            euler = Rotation.from_matrix(rotations).as_euler('ZYX', degrees=True)
        if self.origin_ns is None:
            self.origin_ns = stamp
        values = np.r_[ROOT_ALIGNMENT.apply(targets[0])*100, euler.ravel()]
        if not np.isfinite(values).all():
            raise ValueError('Non-finite BVH frame')
        self.emitted += 1
        return dict(type='frame', seq=self.emitted-1, source_time=(stamp-self.origin_ns)/1e9,
                    values=values.tolist(), timestamp_ns=stamp,
                    source_hip_height_m=support_relative_hip_height(points),
                    fit_p95_m=float(np.percentile(np.linalg.norm(fk-targets, axis=1), 95)),
                    source_estimated=any(j['estimated'] for j in packet['joints']))

    def init_message(self, name='Live COCO17 -> BVH'):
        if self.header is None:
            raise ValueError('Waiting for valid calibration prefix')
        leg_length = np.mean([np.linalg.norm(self.offsets[a])+np.linalg.norm(self.offsets[b])
                              for a,b in ((2,3),(6,7))])
        return dict(type='init', header=self.header, unit='cm', up='Y', name=name,
                    source_leg_length_m=float(leg_length))


class PairedStereo:
    """Adapt latest paired_coco17 wire format without re-pairing or ROI scaling."""
    def __init__(self, calibration_path, camera_ids, allow_diagnostic=False):
        from mocap.calibration import load_calibration
        from mocap.reconstruction import CausalFilter
        sys.path.insert(0, str(ROOT/'upstream/handoff_1080p_12p68_20261017/pc'))
        from receive_paired_pose import validate_message
        self.validate_message = validate_message
        self.calibration = load_calibration(calibration_path)
        if list(camera_ids) != self.calibration['camera_ids']:
            raise ValueError('Explicit logical left/right bindings must equal calibration camera_ids in order')
        if self.calibration.get('world_coordinate_system') != WORLD or self.calibration.get('world_from_cam0') is None:
            raise ValueError('Ground/world calibration is required')
        quality = self.calibration.get('quality', {}).get('acceptance', '')
        if 'diagnostic_only' in quality and not allow_diagnostic:
            raise ValueError('Diagnostic calibration: validate current physical mount first; replay may use --allow-diagnostic')
        self.filter = CausalFilter(.04)
        self.identity = None
        self.previous = None

    def push(self, row):
        from mocap.reconstruction import triangulate
        packet = row.get('message', row)
        self.validate_message(packet, packet['session_id'], tuple(self.calibration['image_size']))
        identity = (packet['session_id'], packet['boot_id'])
        if self.identity is not None and identity != self.identity:
            raise ValueError('Board restarted/session changed: reconnect as a new G1 session')
        views = packet['cameras']
        stamps = [v['capture_timestamp_us']*1000 for v in views]
        if self.previous is not None and (packet['pair_id'] <= self.previous[0] or
                any(a <= b for a, b in zip(stamps, self.previous[1]))):
            raise ValueError('Duplicate or out-of-order board pair')
        self.identity, self.previous = identity, (packet['pair_id'], stamps)
        # Preserve board validity, then apply original triangulation confidence gate.
        points = [[dict(xy=p[:2], confidence=min(1., max(0., p[2])), valid=p[3])
                   for p in v['keypoints']] for v in views]
        joints = triangulate(*points, self.calibration)
        stamp = max(stamps)
        self.filter.apply(joints, stamp)
        return validate(dict(schema='human_skeleton/1.0', unit='m', joint_names=list(COCO17),
            seq=packet['pair_id'], timestamp_ns=stamp, session_id=str(packet['session_id']),
            source='u02_paired_coco17', clock_id='v4l2_monotonic:'+packet['boot_id'],
            coordinate_system=WORLD, calibration_id=self.calibration['calibration_id'],
            model_id='PP-TinyPose:'+str(packet.get('model_sha256', 'board-version-unreported')),
            joints=joints, hardware_sync=False, pair_delta_us=packet['pair_delta_us']))
