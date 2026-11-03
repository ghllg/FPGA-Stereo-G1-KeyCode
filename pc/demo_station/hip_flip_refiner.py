"""Bounded, post-refinement same-frame hip TTA; baseline ROI history is untouched."""
import copy
import hashlib
import time

import numpy as np

from .pose_refiner import PoseRefiner, INPUT_SIZE
from mocap.reconstruction import triangulate
COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def flip_input(tensor):
    return np.ascontiguousarray(tensor[..., ::-1])


def merge_flip_logits(original_x, original_y, flipped_x, flipped_y):
    # MMPose flip_vectors: swap left/right channels; reverse X bins only.
    restored_x = flipped_x[:, COCO_FLIP, ::-1]
    restored_y = flipped_y[:, COCO_FLIP, :]
    return (original_x + restored_x) * .5, (original_y + restored_y) * .5


class _SessionCapture:
    # A per-call class with closures retains these arrays when cyclic GC is off.
    def __init__(self, session, input_name, capture):
        self.session=session;self.input_name=input_name;self.capture=capture

    def run(self, names, feeds):
        outputs=self.session.run(names, feeds)
        self.capture.update(tensor=feeds[self.input_name], outputs=outputs)
        return outputs


class HipFlipRefiner(PoseRefiner):
    hybrid_hip_flip_check = True

    def refine(self, image, points, *, source_image_sha256, frame_index, **kwargs):
        self._current_attempts = []
        result = super().refine(image, points, **kwargs)
        if not hasattr(self, 'pair_attempts'):
            self.pair_attempts = {}
        self.pair_attempts[kwargs['camera_id']] = dict(timestamp_ns=kwargs['timestamp_ns'],
            frame_index=frame_index, source_image_sha256=source_image_sha256,
            image_size=list(image.shape[1::-1]), attempts=self._current_attempts)
        return result

    def _infer(self, image, bbox, padding):
        original_session = self.session
        capture = {}

        self.session = _SessionCapture(original_session, self.input_name, capture)
        try:
            result = super()._infer(image, bbox, padding)
        finally:
            self.session = original_session
        capture['crop'] = result['crop']
        self._current_attempts.append(capture)
        return result


def decode(outputs, crop, image_size):
    x, y = outputs
    if x.shape != (1, 17, 384) or y.shape != (1, 17, 512) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid TTA SimCC outputs')
    bins = np.stack((x.argmax(-1), y.argmax(-1)), axis=-1).astype(np.float32)
    # Keep current deployed mean score, separating TTA from the rejected score change.
    scores = .5 * (x.max(-1) + y.max(-1))
    bins[scores <= 0] = -1
    scale, center = np.array(crop['scale']), np.array(crop['center'])
    xy = bins[0] / 2. / INPUT_SIZE * scale + center - scale / 2
    width, height = image_size
    edge = (bins[0,:,0] < 2) | (bins[0,:,0] >= 382) | (bins[0,:,1] < 2) | (bins[0,:,1] >= 510)
    valid = ((scores[0] >= .35) & ~edge & (xy[:,0] >= 0) & (xy[:,0] < width)
             & (xy[:,1] >= 0) & (xy[:,1] < height))
    return [dict(xy=p.tolist() if good else None, raw_xy=p.tolist(), confidence=min(float(score), 1.),
                 raw_simcc_score=float(score), valid=bool(good), pose_candidate_source='same_frame_hip_flip_tta')
            for p, score, good in zip(xy, scores[0], valid)]


def recover_hips(row, refiner, calibration):
    """At most one additional forward per view; only previously failed hips change."""
    started = time.perf_counter()
    original_views = row['views']
    original_geometry = triangulate(original_views[0]['joints'], original_views[1]['joints'], calibration)
    targets = [j for j in (11, 12) if original_geometry[j]['reason'] == 'reprojection_error']
    audit = dict(triggered=bool(targets), targets=targets, adopted=[], extra_forward_calls=0,
        views=[], source_seq=row['seq'], timestamp_ns=row['timestamp_ns'], inference_ms=0., total_ms=0.,
        original_geometry=original_geometry, roi_history_feedback=False,
        point_origins={v['camera_id']:['baseline_rtmpose'] * 17 for v in original_views})
    result = copy.deepcopy(row)
    result['hip_flip_recovery'] = audit
    if not targets:
        audit.update(reason='no_hip_reprojection_failure', total_ms=(time.perf_counter()-started)*1000)
        return result
    attempt_indices = [set(v['refinement']['joint_roi_attempt_indices'][j] for j in targets) for v in original_views]
    if any(len(indices) != 1 for indices in attempt_indices):
        audit.update(reason='mixed_hip_source_rois_exceeds_two_forward_budget', total_ms=(time.perf_counter()-started)*1000)
        return result
    candidate_views = copy.deepcopy(original_views)
    selected = []
    for vi, (view, indices) in enumerate(zip(original_views, attempt_indices)):
        index = next(iter(indices))
        binding = refiner.pair_attempts[view['camera_id']]
        if any(binding[key] != view[key] for key in ('timestamp_ns', 'image_size', 'frame_index', 'source_image_sha256')):
            raise ValueError('Hip TTA cache differs from the current source image identity')
        attempt = binding['attempts'][index]
        if attempt['crop'] != view['refinement']['roi_attempts'][index]['crop']:
            raise ValueError('Hip TTA cache differs from the original selected ROI')
        selected.append((index, attempt))
    for vi, (view, (index, attempt)) in enumerate(zip(original_views, selected)):
        before = time.perf_counter()
        flipped = refiner.session.run(refiner.output_names, {refiner.input_name: flip_input(attempt['tensor'])})
        elapsed = (time.perf_counter()-before)*1000
        audit['extra_forward_calls'] += 1
        audit['inference_ms'] += elapsed
        merged = merge_flip_logits(*attempt['outputs'], *flipped)
        proposed = decode(merged, attempt['crop'], view['image_size'])
        for j in targets:
            candidate_views[vi]['joints'][j] = proposed[j]
        audit['views'].append(dict(camera_id=view['camera_id'], timestamp_ns=view['timestamp_ns'],
            frame_index=view['frame_index'], source_image_sha256=view['source_image_sha256'],
            baseline_attempt=index, crop=attempt['crop'], input_tensor_sha256=hashlib.sha256(attempt['tensor'].tobytes()).hexdigest(),
            original_hips={str(j):view['joints'][j] for j in targets}, proposed_hips={str(j):proposed[j] for j in targets}, inference_ms=elapsed))
    proposed_geometry = triangulate(candidate_views[0]['joints'], candidate_views[1]['joints'], calibration)
    audit['proposed_geometry'] = {str(j):proposed_geometry[j] for j in targets}
    for j in targets:
        if proposed_geometry[j]['valid']:
            for vi in (0, 1):
                result['views'][vi]['joints'][j] = candidate_views[vi]['joints'][j]
                audit['point_origins'][original_views[vi]['camera_id']][j] = 'same_frame_hip_flip_tta'
            audit['adopted'].append(j)
    # An optional correction never replaces an original geometric observation.
    for j in range(17):
        if j not in audit['adopted']:
            if any(result['views'][vi]['joints'][j] != original_views[vi]['joints'][j] for vi in (0,1)):
                raise ValueError('Hip TTA changed an untargeted or unaccepted joint')
    final_geometry = triangulate(result['views'][0]['joints'], result['views'][1]['joints'], calibration)
    if any(old['valid'] and not new['valid'] for old, new in zip(original_geometry, final_geometry)):
        raise ValueError('Hip TTA invalidated an original geometric observation')
    audit.update(reason='completed_same_frame_flip_trial', total_ms=(time.perf_counter()-started)*1000)
    if audit['extra_forward_calls'] > 2:
        raise ValueError('Hip TTA exceeded the two-forward budget')
    return result
