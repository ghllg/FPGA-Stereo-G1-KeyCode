"""P3e ablation: preserve feedback state; halve only final lateral rotation."""
import numpy as np
from scipy.spatial.transform import Rotation


class PositionFeedback:
    def __init__(self):
        self.origin = None
        self.filtered = np.zeros(2)

    def apply(self, refs, target, actual, velocity, dt):
        if self.origin is None:
            self.origin = actual[:2]-target[:2]
        error = target[:2]+self.origin-actual[:2]
        desired = 3.*error-.7*velocity[:2]
        desired *= min(1., np.deg2rad(30.)/max(np.linalg.norm(desired), 1e-12))
        self.filtered += (desired-self.filtered)*(1.-np.exp(-dt/.10))
        rotation = Rotation.from_rotvec([-.5*self.filtered[1], self.filtered[0], 0.])
        modified = dict(refs)
        values = refs["body_rot"]
        modified["body_rot"] = (rotation*Rotation.from_quat(values.reshape(-1,4))).as_quat().reshape(values.shape).astype(np.float32)
        return modified
