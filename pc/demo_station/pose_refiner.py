"""Fixed RTMPose-m refinement of an FPGA COCO17 bounding box.

Affine and SimCC processing follow rtmlib (Apache-2.0); RGB normalization
follows this model's bundled SDK pipeline. See models/rtmpose_m/source_manifest.json.
The caller owns JPEG decoding, camera/frame identity, and original image size checks.
"""
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from .person_detector import PersonDetector, make_session


MODEL_SHA256 = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
MODEL_ID = 'rtmpose-m-simcc-coco17:' + MODEL_SHA256
BBOX_SOURCE = 'fpga_coco17_fullbody_with_recovery_v1'
INPUT_SIZE = (192, 256)
MEAN = np.array((123.675, 116.28, 103.53))
STD = np.array((58.395, 57.12, 57.375))
TRACK_MAX_AGE_NS = 350_000_000
DETECTION_INTERVAL_NS = 500_000_000


def _full_body(usable):
    return (np.count_nonzero(usable) >= 8 and all(np.any(usable[a:b])
            for a, b in ((5, 7), (11, 13), (13, 15), (15, 17))))


def _bbox(points):
    return np.r_[points.min(axis=0), points.max(axis=0)] if len(points) >= 2 else None


def _action_bbox(image, roi_xywh, include_bounds=None):
    height, width = image.shape[:2]
    x, y, w, h = roi_xywh if roi_xywh is not None else (0, 0, width, height)
    bbox = np.array([x, y, x + w, y + h], dtype=float)
    if include_bounds is not None:
        # Recovery must retain all image pixels visible to the first affine crop.
        bbox[:2] = np.minimum(bbox[:2], include_bounds[:2])
        bbox[2:] = np.maximum(bbox[2:], include_bounds[2:])
    return np.clip(bbox, [0, 0, 0, 0], [width, height, width, height])


def _proposal_issue(bbox, usable, previous):
    if bbox is None or np.any(bbox[2:] <= bbox[:2]):
        return 'insufficient_or_degenerate_fpga_bbox'
    if not _full_body(usable):
        return 'incomplete_fpga_body_coverage'
    if previous is not None:
        size, old_size = bbox[2:] - bbox[:2], previous[2:] - previous[:2]
        if size[1] < old_size[1] * .6 or size.prod() < old_size.prod() * .35:
            return 'fpga_bbox_collapsed'
        distance = np.linalg.norm((bbox[:2] + bbox[2:] - previous[:2] - previous[2:]) * .5)
        if distance > max(old_size) * .5 or np.any(size > old_size * 2.5):
            return 'fpga_bbox_jump'
    return None


def _affine(image, center, scale):
    scale = scale.copy()
    if scale[0] > scale[1] * .75:
        scale[1] = scale[0] / .75
    else:
        scale[0] = scale[1] * .75
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0., -.5 * scale[0]])
    direction = src[0] - src[1]
    src[2] = src[1] + np.r_[-direction[1], direction[0]]
    dst[0], dst[1] = [96., 128.], [96., 32.]
    direction = dst[0] - dst[1]
    dst[2] = dst[1] + np.r_[-direction[1], direction[0]]
    warp = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, warp, INPUT_SIZE, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0), scale


class PoseRefiner:
    def __init__(self, model_path, sha256=MODEL_SHA256, *, backend='cpu'):
        started = time.perf_counter()
        if sha256 != MODEL_SHA256:
            raise ValueError('Only the verified RTMPose-m model is supported')
        path = Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('RTMPose model SHA256 mismatch')
        self.backend = backend
        self.session = make_session(path, backend, 4)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[1:] != [3, 256, 192]:
            raise ValueError('Unexpected RTMPose input shape')
        if len(outputs) != 2 or [o.shape[-1] for o in outputs] != [384, 512]:
            raise ValueError('Unexpected RTMPose SimCC output shape')
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in outputs]
        self.detector = PersonDetector(backend=backend)
        self._states = {}
        self.initialization_ms = (time.perf_counter() - started) * 1000

    def warmup(self):
        """Compile/upload both networks before live input, without ROI state."""
        started = time.perf_counter()
        self.session.run(self.output_names, {self.input_name: np.zeros((1, 3, 256, 192), np.float32)})
        pose_ms = (time.perf_counter() - started) * 1000
        return dict(backend=self.backend, initialization_ms=self.initialization_ms,
                    pose_ms=pose_ms, detector_ms=self.detector.warmup())

    def refine(self, image, raw_keypoints, *, camera_id=None, timestamp_ns=None, roi_xywh=None):
        started = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError('Expected a decoded BGR uint8 HWC image')
        points = np.asarray(raw_keypoints, dtype=float)
        if points.shape != (17, 3):
            raise ValueError('Expected seventeen original FPGA [x, y, confidence] points')
        height, width = image.shape[:2]
        if roi_xywh is not None and (len(roi_xywh) != 4 or not np.isfinite(roi_xywh).all() or min(roi_xywh[2:]) <= 0):
            raise ValueError('Expected finite positive action ROI xywh')
        state = None
        if camera_id is not None and timestamp_ns is not None:
            if not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
                raise ValueError('Expected nonnegative integer source timestamp_ns')
            state = self._states.get(camera_id)
            if state is None or state['shape'] != image.shape or timestamp_ns <= state['timestamp_ns'] or timestamp_ns - state['timestamp_ns'] > 2_000_000_000:
                state = dict(shape=image.shape, timestamp_ns=timestamp_ns, good_bbox=None,
                             good_ns=None, detection_ns=None, needs_detection=False, identity_rejected=False)
                self._states[camera_id] = state
            state['timestamp_ns'] = timestamp_ns
        previous = state['good_bbox'] if state is not None and state['good_ns'] is not None and timestamp_ns - state['good_ns'] <= TRACK_MAX_AGE_NS else None
        usable = (np.isfinite(points).all(axis=1) & (points[:, 2] >= .35)
                  & (points[:, 0] >= 0) & (points[:, 0] < width)
                  & (points[:, 1] >= 0) & (points[:, 1] < height))
        fpga_bbox = _bbox(points[usable, :2])
        issue = _proposal_issue(fpga_bbox, usable, previous)
        if state is not None and state['needs_detection']:
            issue = issue or 'previous_crop_quality_failure'
        bbox_ms = (time.perf_counter() - started) * 1000
        bbox, roi_source, detection = fpga_bbox, 'fpga', None
        if issue:
            if previous is not None and not state['needs_detection']:
                bbox, roi_source = previous.copy(), 'tracked'
            elif state is None or state['detection_ns'] is None or timestamp_ns - state['detection_ns'] >= DETECTION_INTERVAL_NS:
                reference = state['good_bbox'] if state is not None and state['good_ns'] is not None and (state['identity_rejected'] or timestamp_ns - state['good_ns'] < 2_000_000_000) else None
                detection = self.detector.detect(image, reference_bbox=reference, roi_xywh=roi_xywh)
                bbox = np.asarray(detection['bbox'], dtype=float) if detection['bbox'] is not None else None
                roi_source = 'detector'
                if state is not None:
                    state['detection_ns'] = timestamp_ns
                    if detection.get('reason') == 'person_does_not_match_previous_roi':
                        state['identity_rejected'] = True
                    elif bbox is not None:
                        state['identity_rejected'] = False
                    state['needs_detection'] = state['identity_rejected']
            else:
                bbox, roi_source = (previous.copy(), 'tracked') if previous is not None and not state['identity_rejected'] else (None, 'unavailable')
            identity_rejected = (state is not None and state['identity_rejected']) or (detection is not None and detection.get('reason') == 'person_does_not_match_previous_roi')
            if identity_rejected:
                bbox, roi_source, issue = None, 'unavailable', 'person_does_not_match_previous_roi'
            elif bbox is None:
                bbox, roi_source = _action_bbox(image, roi_xywh), 'action_roi_reacquire'
        timing = dict(bbox=bbox_ms,
                      detection=detection['total_ms'] if detection else 0.,
                      preprocess=0., inference=0., postprocess=0., reacquire=0., total=0.)
        result = dict(status='bbox_unavailable', reason=issue,
                      keypoints=None, valid=[False] * 17, bbox=bbox.tolist() if bbox is not None else None,
                      bbox_used_indices=np.flatnonzero(usable).tolist(), crop=None, timing_ms=timing,
                      fpga_bbox=fpga_bbox.tolist() if fpga_bbox is not None else None,
                      roi_source=roi_source, recovery_reason=issue, detector=detection,
                      joint_rejection_reasons=['no_usable_roi'] * 17, backend=self.backend,
                      model_id=MODEL_ID, model_sha256=MODEL_SHA256, bbox_source=BBOX_SOURCE)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            timing['total'] = (time.perf_counter() - started) * 1000
            return result
        prediction = self._infer(image, bbox, 1. if roi_source == 'action_roi_reacquire' else 1.25)
        joint_roi_attempt_indices = [0] * 17
        attempts = [dict(roi_source=roi_source, bbox=bbox.tolist(), valid=prediction['valid'],
                         crop=prediction['crop'], joint_rejection_reasons=prediction['joint_rejection_reasons'], timing_ms=prediction['timing_ms'].copy())]
        for key in ('preprocess', 'inference', 'postprocess'):
            timing[key] += prediction['timing_ms'][key]
        boundary = any(reason == 'crop_boundary_peak' for reason in prediction['joint_rejection_reasons'][5:])
        if roi_source != 'action_roi_reacquire' and (boundary or (roi_source == 'detector' and not _full_body(prediction['valid']))):
            retry_started = time.perf_counter()
            bbox, roi_source = _action_bbox(image, roi_xywh, prediction['crop']['bounds']), 'action_roi_reacquire'
            first_prediction = prediction
            prediction = self._infer(image, bbox, 1.)
            timing['reacquire'] = (time.perf_counter() - retry_started) * 1000
            for key in ('preprocess', 'inference', 'postprocess'):
                timing[key] += prediction['timing_ms'][key]
            attempts.append(dict(roi_source=roi_source, bbox=bbox.tolist(), valid=prediction['valid'],
                                 crop=prediction['crop'], joint_rejection_reasons=prediction['joint_rejection_reasons'], timing_ms=prediction['timing_ms'].copy()))
            # Both attempts observe this same image; recovery fills rejected points only.
            joint_roi_attempt_indices = [0 if valid else 1 for valid in first_prediction['valid']]
            for key in ('keypoints', 'valid', 'joint_rejection_reasons'):
                prediction[key] = [first_prediction[key][i] if source == 0 else prediction[key][i]
                                   for i, source in enumerate(joint_roi_attempt_indices)]
            result['recovery_reason'] = 'crop_boundary_or_incomplete_detected_pose'
        result.update({key: value for key, value in prediction.items() if key != 'timing_ms'})
        result.update(status='ok', reason=None, bbox=bbox.tolist(), roi_source=roi_source,
                      roi_attempts=attempts, joint_roi_attempt_indices=joint_roi_attempt_indices)
        if state is not None:
            valid = np.asarray(result['valid'])
            reliable = _full_body(valid) and bool(np.all(valid[15:17]))
            if reliable:
                new_bbox = _bbox(np.asarray(result['keypoints'])[valid, :2])
                if new_bbox is not None and np.all(new_bbox[2:] > new_bbox[:2]):
                    state['good_bbox'], state['good_ns'] = new_bbox, timestamp_ns
                    state['needs_detection'] = False
            elif boundary or not _full_body(valid):
                state['needs_detection'] = True
        timing['total'] = (time.perf_counter() - started) * 1000
        return result

    def _infer(self, image, bbox, padding):
        height, width = image.shape[:2]
        timing = {}
        before = time.perf_counter()
        center = (bbox[:2] + bbox[2:]) * .5
        scale = (bbox[2:] - bbox[:2]) * padding
        resized, scale = _affine(image, center, scale)
        normalized = (resized[:, :, ::-1] - MEAN) / STD
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)[None]
        timing['preprocess'] = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        timing['inference'] = (time.perf_counter() - before) * 1000
        if len(outputs) != 2 or [o.shape for o in outputs] != [(1, 17, 384), (1, 17, 512)]:
            raise ValueError('Expected exactly seventeen RTMPose SimCC joints')
        if not all(np.isfinite(o).all() for o in outputs):
            raise ValueError('Nonfinite RTMPose output')
        before = time.perf_counter()
        simcc_x, simcc_y = outputs
        locs = np.stack((np.argmax(simcc_x, axis=-1), np.argmax(simcc_y, axis=-1)), axis=-1).astype(np.float32)
        scores = .5 * (simcc_x.max(axis=-1) + simcc_y.max(axis=-1))
        locs[scores <= 0.] = -1
        xy = (locs / 2.0) / INPUT_SIZE * scale + center - scale / 2
        keypoints = np.column_stack((xy[0], scores[0]))
        valid = ((scores[0] >= .35) & (xy[0, :, 0] >= 0) & (xy[0, :, 0] < width)
                 & (xy[0, :, 1] >= 0) & (xy[0, :, 1] < height))
        edge = ((locs[0, :, 0] < 2) | (locs[0, :, 0] >= 382)
                | (locs[0, :, 1] < 2) | (locs[0, :, 1] >= 510))
        reasons = [None if good else 'low_confidence_or_outside_image' for good in valid]
        for index in np.flatnonzero((scores[0] >= .35) & edge):
            reasons[index] = 'crop_boundary_peak'
        valid &= ~edge
        bounds = np.r_[center - scale / 2, center + scale / 2]
        timing['postprocess'] = (time.perf_counter() - before) * 1000
        return dict(keypoints=keypoints.tolist(), valid=valid.tolist(), joint_rejection_reasons=reasons,
                    timing_ms=timing, crop=dict(center=center.tolist(), scale=scale.tolist(), bounds=bounds.tolist(),
                    outside_image=bool(np.any(bounds[:2] < 0) or bounds[2] > width or bounds[3] > height)))
