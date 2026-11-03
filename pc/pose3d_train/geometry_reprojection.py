"""Same calibrated geometry for synthetic training and real JSONL inference."""
import hashlib
import json
import cv2
import numpy as np
from mocap.calibration import load_calibration

BODY = np.arange(5, 17)
BODY_NAMES = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
LAFAN_BODY = ('LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
              'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot')
BONES = np.array([(0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
                  (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)])
WORLD = 'world_right_handed_X_forward_Y_left_Z_up'


def fingerprint(calibration):
    keys = ('calibration_id', 'camera_ids', 'image_size', 'K', 'distortion',
            'R_cam0_to_cam1', 't_cam0_to_cam1_m', 'world_from_cam0', 'world_coordinate_system')
    data = {key: calibration.get(key) for key in keys}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Stereo:
    def __init__(self, calibration, max_reprojection_px=5.):
        if max_reprojection_px not in (5., 6., 8.):
            raise ValueError("Use a declared 5, 6 or 8 px reconstruction policy.")
        self.max_reprojection_px = float(max_reprojection_px)
        self.calibration = load_calibration(calibration) if not isinstance(calibration, dict) else calibration
        cal = self.calibration
        if cal.get('world_coordinate_system') != WORLD or cal.get('world_from_cam0') is None:
            raise ValueError('A metre-based calibrated world coordinate system is required.')
        self.k = np.array(cal['K'], dtype=float)
        self.d = [np.array(d, dtype=float) for d in cal['distortion']]
        self.r = np.array(cal['R_cam0_to_cam1'])
        self.t = np.array(cal['t_cam0_to_cam1_m'])
        self.w = np.array(cal['world_from_cam0'])
        self.rv = [np.zeros(3), cv2.Rodrigues(self.r)[0]]
        self.tv = [np.zeros(3), self.t]
        self.p = [np.column_stack([np.eye(3), np.zeros(3)]), np.column_stack([self.r, self.t])]
        self.hash = fingerprint(cal)

    def project(self, xyz):
        """World metres [N,J,3] -> original distorted pixels [N,2,J,2]."""
        xyz = np.asarray(xyz, dtype=float)
        cam = (xyz - self.w[:3, 3]) @ self.w[:3, :3]
        pixels = [cv2.projectPoints(cam.reshape(-1, 3), r, t, k, d)[0].reshape(*cam.shape[:2], 2)
                  for r, t, k, d in zip(self.rv, self.tv, self.k, self.d)]
        depth = np.stack([cam[..., 2], (cam @ self.r.T + self.t)[..., 2]], axis=1)
        return np.stack(pixels, axis=1), depth

    def features(self, xy, confidence, candidate_base=False):
        """Returns features, geometry with missing joints at root, observed mask and ray diagnostics."""
        xy, confidence = np.asarray(xy, dtype=float), np.asarray(confidence, dtype=float)
        if xy.ndim != 4 or xy.shape[1:] != (2, 12, 2) or confidence.shape != xy.shape[:-1]:
            raise ValueError('Expected [frames,2,12,2] pixels and [frames,2,12] confidence.')
        visible = (confidence >= .35) & np.isfinite(xy).all(axis=-1)
        confidence = np.where(visible, confidence, 0.)
        safe = np.where(visible[..., None], xy, 0.)
        und = np.stack([cv2.undistortPoints(safe[:, i].reshape(-1, 1, 2), self.k[i], self.d[i]).reshape(-1, 12, 2)
                        for i in range(2)], axis=1)
        rays = np.concatenate([und, np.ones((*und.shape[:-1], 1))], axis=-1)
        rays[:, 1] = rays[:, 1] @ self.r
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        angle = np.degrees(np.arccos(np.clip(np.sum(rays[:, 0] * rays[:, 1], axis=-1), -1, 1)))
        h = cv2.triangulatePoints(self.p[0], self.p[1], und[:, 0].reshape(-1, 2).T, und[:, 1].reshape(-1, 2).T).T
        finite_h = abs(h[:, 3]) > 1e-9
        cam = (h[:, :3] / np.where(finite_h, h[:, 3], 1.)[:, None]).reshape(-1, 12, 3)
        world = cam @ self.w[:3, :3].T + self.w[:3, 3]
        projected, depth = self.project(world)
        error = np.linalg.norm(projected - safe, axis=-1)
        good = (visible.all(axis=1) & finite_h.reshape(-1, 12) & (depth > 0).all(axis=1)
                & (np.linalg.norm(cam, axis=-1) < 20) & (angle >= 1.) & (error.max(axis=1) <= self.max_reprojection_px)
                & np.isfinite(world).all(axis=-1))
        hip_good = good[:, 6:8]
        root = np.sum(np.where(hip_good[..., None], world[:, 6:8], 0.), axis=1) / np.maximum(hip_good.sum(axis=1), 1)[:, None]
        usable = good
        if candidate_base:
            # A bounded, untrusted triangulation is an MLP input only. It never
            # changes the reliable-observation mask or bypasses completion gates.
            usable = good | (visible.all(axis=1) & finite_h.reshape(-1,12) & (depth>.1).all(axis=1)
                      & (angle>=1.) & (error.max(axis=1)<=20.) & np.isfinite(world).all(axis=-1)
                      & (np.linalg.norm(world-root[:,None],axis=-1)<2.))
        base = np.where(usable[..., None], world, root[:, None])
        masked_und = np.where(visible[..., None], und, 0.)
        features = np.concatenate([masked_und.reshape(-1, 48), confidence.reshape(-1, 24),
                                   visible.reshape(-1, 24), (base - root[:, None]).reshape(-1, 36),
                                   good, root], axis=1).astype(np.float32)
        return features, base.astype(np.float32), good, {'reprojection_px': error, 'depth': depth, 'visible': visible}
