"""On-demand person reacquisition with the official RTMDet-nano SDK export.

Preprocessing and output scaling follow the accompanying pipeline.json.
The model is loaded at startup, but inference runs only when the ROI is lost.
"""
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np


MODEL_SHA256 = 'b8585adb6d1352796f6e86337e09c3459b830645a7e7192d9381670bea2facbd'
MODEL_ID = 'rtmdet-nano-person:' + MODEL_SHA256


def make_session(model_path, backend, threads):
    import onnxruntime as ort
    if backend not in ('cpu', 'dml'):
        raise ValueError('Expected backend cpu or dml')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ['CPUExecutionProvider']
    if backend == 'dml':
        if 'DmlExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Requested DirectML backend is unavailable')
        options.enable_mem_pattern = False
        # Leave CPU time to reconstruction and G1 while DirectML is idle.
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def _iou(boxes, reference):
    low = np.maximum(boxes[:, :2], reference[:2])
    high = np.minimum(boxes[:, 2:], reference[2:])
    intersect = np.maximum(high - low, 0).prod(axis=1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    ref_area = np.prod(reference[2:] - reference[:2])
    return intersect / np.maximum(area + ref_area - intersect, 1.)


class PersonDetector:
    def __init__(self, model_path=None, *, backend='cpu'):
        path = Path(model_path) if model_path else Path(__file__).parent / 'models' / 'rtmdet_nano' / 'end2end.onnx'
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('RTMDet model SHA256 mismatch')
        self.session = make_session(path, backend, 1)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if self.output_names != ['dets', 'labels']:
            raise ValueError('Unexpected RTMDet SDK outputs')

    def warmup(self):
        started = time.perf_counter()
        self.session.run(self.output_names, {self.input_name: np.zeros((1, 3, 320, 320), np.float32)})
        return (time.perf_counter() - started) * 1000

    def detect(self, image, reference_bbox=None, roi_xywh=None):
        started = time.perf_counter()
        height, width = image.shape[:2]
        ratio = min(320 / width, 320 / height)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        padded = np.full((320, 320, 3), 114, np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
        tensor = (padded.astype(np.float32) - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
        dets, labels = self.session.run(self.output_names, {self.input_name: np.ascontiguousarray(tensor.transpose(2, 0, 1))[None]})
        dets, labels = np.asarray(dets)[0], np.asarray(labels)[0]
        if dets.ndim != 2 or dets.shape[1] != 5 or labels.shape != dets.shape[:1]:
            raise ValueError('Unexpected RTMDet detection shape')
        keep = np.isfinite(dets).all(axis=1) & (labels == 0) & (dets[:, 4] >= .5)
        dets = dets[keep].copy()
        dets[:, [0, 2]] *= width / resized_width
        dets[:, [1, 3]] *= height / resized_height
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, width)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, height)
        dets = dets[((dets[:, 2:4] - dets[:, :2]) >= [16, 32]).all(axis=1)]
        selected = None
        reason = 'no_person'
        if len(dets):
            if reference_bbox is not None:
                overlap = _iou(dets[:, :4], np.asarray(reference_bbox))
                if overlap.max() >= .1:
                    selected = int(np.argmax(overlap))
                else:
                    reason = 'person_does_not_match_previous_roi'
            elif roi_xywh is not None:
                roi = np.array([roi_xywh[0], roi_xywh[1], roi_xywh[0] + roi_xywh[2], roi_xywh[1] + roi_xywh[3]])
                overlap = _iou(dets[:, :4], roi)
                if overlap.max() > 0:
                    selected = int(np.argmax(overlap * dets[:, 4]))
                else:
                    reason = 'no_person_in_action_region'
            else:
                selected = int(np.argmax(dets[:, 4]))
        return dict(bbox=dets[selected, :4].tolist() if selected is not None else None,
                    confidence=float(dets[selected, 4]) if selected is not None else None,
                    reason=None if selected is not None else reason, candidate_count=len(dets),
                    model_id=MODEL_ID, model_sha256=MODEL_SHA256,
                    total_ms=(time.perf_counter() - started) * 1000)
