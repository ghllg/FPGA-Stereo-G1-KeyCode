"""Causal, opt-in arm-length check against earlier clear observations.

This is an estimated correction, not a new measurement. A short arm history
cannot resolve occlusion or establish absolute anatomical accuracy.
"""
import numpy as np
from scipy.optimize import least_squares


class ArmStabilizer:
    def __init__(self, stereo):
        self.stereo = stereo
        self.reset()

    def reset(self):
        self.lengths = {j: [] for j in (7, 8, 9, 10)}
        self.previous = {}
        self.timestamp = None

    def fit(self, points, pixels, scores, lengths, previous):
        shoulder = points[0]
        # Match the geometry visibility floor. A low-confidence image point
        # must not pull a completed joint back toward a discarded detection.
        weights = np.sqrt(np.clip((scores-.35)/.25, 0., 1.))

        def residual(flat):
            arm = flat.reshape(2, 3)
            projected, _ = self.stereo.project(arm[None])
            image_error = ((projected[0]-pixels)*weights[..., None]/8.).ravel()
            # Pseudo-Huber image loss prevents one misidentified view dominating.
            image_error = np.sign(image_error)*np.sqrt(2*(np.sqrt(1+image_error**2)-1))
            bone_error = np.array([np.linalg.norm(arm[0]-shoulder),
                                   np.linalg.norm(arm[1]-arm[0])])-lengths
            terms = [image_error, bone_error/.015]
            if previous is not None:
                terms.append(((arm-previous)/.15).ravel())
            return np.concatenate(terms)

        starts = [points[1:].ravel()]
        if previous is not None:
            starts.append(previous.ravel())
        trials = [least_squares(residual, start, max_nfev=35,
                               ftol=1e-4, xtol=1e-4, gtol=1e-4) for start in starts]
        best = min(trials, key=lambda trial: np.dot(trial.fun, trial.fun))
        return best.x.reshape(2, 3)

    def update(self, joints, views, timestamp_ns):
        if self.timestamp is not None and timestamp_ns-self.timestamp > 250_000_000:
            self.reset()
        self.timestamp = timestamp_ns
        for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
            ids = (shoulder, elbow, wrist)
            if not all(joints[j]['valid'] for j in ids):
                self.previous.pop(wrist, None)
                continue
            points = np.array([joints[j]['xyz'] for j in ids])
            scores = np.array([[v['joints'][j]['confidence'] if v['joints'][j]['valid'] else 0.
                                for j in ids] for v in views])
            current = np.linalg.norm(np.diff(points, axis=0), axis=1)
            for k, j in enumerate((elbow, wrist), 1):
                if (joints[j]['reason'] == joints[ids[k-1]]['reason'] == 'observed'
                        and scores[:, k-1:k+1].min() >= .6):
                    self.lengths[j].append(float(current[k-1]))
                    del self.lengths[j][:-30]
            ready = min(len(self.lengths[j]) for j in (elbow, wrist)) >= 5
            if ready and scores[:, 0].min() >= .6 and (scores[:, 1:] > .2).all():
                lengths = np.array([np.median(self.lengths[j]) for j in (elbow, wrist)])
                # Fade by geometric disagreement and image confidence. A score
                # changing from .3500 to .3499 must not switch a correction off.
                weight = np.clip((np.max(abs(current-lengths)/lengths)-.05)/.20, 0., 1.)
                weight = weight*weight*(3.-2.*weight)
                trust = np.clip((scores[:, 1:].min()-.2)/.1, 0., 1.)
                weight *= trust*trust*(3.-2.*trust)
                # Each fitted joint needs at least one measured ray. Without
                # it, bone length and the previous pose can invent a direction.
                visible = np.clip((scores[:, 1:].max(axis=0).min()-.35)/.05, 0., 1.)
                weight *= visible*visible*(3.-2.*visible)
                if weight > 0:
                    pixels = np.array([[v['joints'][j]['raw_xy'] for j in (elbow, wrist)] for v in views])
                    fitted = self.fit(points, pixels, scores[:, 1:], lengths, self.previous.get(wrist))
                    fitted_lengths = np.linalg.norm(np.diff(np.vstack([points[0], fitted]), axis=0), axis=1)
                    _, depth = self.stereo.project(fitted[None])
                    if (np.isfinite(fitted).all() and (depth > .1).all()
                            and (abs(fitted_lengths-lengths) <= .15*lengths).all()):
                        for k, j in enumerate((elbow, wrist)):
                            original = dict(joints[j])
                            blended = points[k+1]+weight*(fitted[k]-points[k+1])
                            joints[j].update(xyz=blended.tolist(), estimated=True, confidence=.2,
                                reason='arm_length_constrained', correction='arm_length_fit_v3_reliability',
                                uncorrected_joint=original, reference_bone_lengths_m=lengths.tolist(),
                                correction_blend=float(weight))
            self.previous[wrist] = np.array([joints[j]['xyz'] for j in (elbow, wrist)])
