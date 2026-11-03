"""FPGA-guided RTMPose candidate; original identity and 3D geometry stay enforced."""
import hashlib
import json
from pathlib import Path
import queue
import time
import traceback

import cv2
import numpy as np
from demo_station.core import write_json
from demo_station.jpeg_pair_decoder import decode_pair
from demo_station.hybrid_receiver import ReceivedJpeg, image_identity
from demo_station.hybrid_candidate import activate_runtime
from pose3d_train.board_adapter import PacketDecoder
from pose3d_train.online_pair20 import OnlineSession
from demo_station.pair20_performance_receiver import validate_message
from demo_station.pose_refiner import MODEL_ID, MODEL_SHA256, PoseRefiner


def verified_images(record):
    """Validate both exact JPEG identities; saved replays also verify file hashes."""
    message = record['message']
    images = record.get('raw_images', [])
    if len(images) != 2 or [item.get('side') for item in images] != ['left', 'right']:
        raise ValueError('Exactly one left and one right raw JPEG are required.')
    payloads = record.get('_verified_jpegs')
    if payloads is not None and (type(payloads) is not tuple or len(payloads) != 2):
        raise ValueError('Exactly two immutable received JPEG payloads are required.')
    encoded = []
    for index, (item, camera) in enumerate(zip(images, message['cameras'])):
        expected = dict(side=camera['camera_id'], boot_id=message['boot_id'],
                        session_id=message['session_id'], pair_id=message['pair_id'],
                        frame_id=camera['frame_id'], capture_timestamp_us=camera['capture_timestamp_us'],
                        source_size=camera['source_size'])
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError('Raw JPEG identity differs from the FPGA camera frame.')
        path = Path(item['path'])
        if not path.is_absolute():
            raise ValueError('Raw JPEG path must be absolute.')
        if payloads is None:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item.get('sha256'):
                raise ValueError('Raw JPEG SHA256 mismatch.')
        else:
            payload = payloads[index]
            if (type(payload) is not ReceivedJpeg or payload.identity != image_identity(item)
                    or payload.sha256 != item.get('sha256') or len(payload.data) != item.get('jpeg_bytes')):
                raise ValueError('Received JPEG payload differs from its persisted evidence identity.')
            data = payload.data
        if not data.startswith(b'\xff\xd8'):
            raise ValueError('Raw image must be a JPEG.')
        encoded.append(data)
    return decode_pair(tuple(encoded), [camera['source_size'] for camera in message['cameras']])


def refine_row(row, record, refiner, *, hip_flip_check=False, calibration=None):
    if hip_flip_check:
        if calibration is None: raise ValueError("Hip flip refinement requires original calibration")
        refiner.pair_attempts = {}
    images = verified_images(record)
    fpga_views = row['views']
    views, details = [], []
    for view, camera, image, raw_image in zip(fpga_views, record['message']['cameras'], images, record['raw_images']):
        context = dict(camera_id=view['camera_id'], timestamp_ns=view['timestamp_ns'], roi_xywh=view['roi_xywh'])
        if hip_flip_check:
            context.update(source_image_sha256=raw_image['sha256'], frame_index=view['frame_index'])
        result = refiner.refine(image, [point[:3] for point in camera['keypoints']], **context)
        if result['model_id'] != MODEL_ID or result['model_sha256'] != MODEL_SHA256:
            raise ValueError('Refinement model identity mismatch.')
        backend = result.get('backend', getattr(refiner, 'backend', 'cpu'))
        if backend not in ('cpu', 'dml'):
            raise ValueError('Unknown refinement execution backend.')
        joints = []
        if result['status'] == 'bbox_unavailable':
            joints = [dict(xy=None, confidence=0., valid=False,
                           reason='refinement_bbox_unavailable') for _ in range(17)]
        elif result['status'] == 'ok':
            points = np.asarray(result['keypoints'], dtype=float)
            flags = result['valid']
            reasons = result.get('joint_rejection_reasons', [None] * 17)
            if points.shape != (17, 3) or len(flags) != 17 or not np.isfinite(points).all():
                raise ValueError('Malformed refined keypoints.')
            if len(reasons) != 17 or any(reason is not None and not isinstance(reason, str) for reason in reasons):
                raise ValueError('Malformed refinement rejection reasons.')
            width, height = view['image_size']
            for (x, y, score), flag, reason in zip(points.tolist(), flags, reasons):
                admissible = score >= .35 and 0 <= x < width and 0 <= y < height
                if type(flag) is not bool or (flag and not admissible) or score < 0:
                    raise ValueError('Refined valid flag contradicts score/image bounds.')
                joints.append(dict(xy=[x, y] if flag else None, confidence=min(score, 1.),
                                   valid=flag, raw_xy=[x, y], raw_simcc_score=score))
                if not flag and reason: joints[-1]['reason'] = reason
        else:
            raise ValueError('Unknown refinement status.')
        detail = {key: value for key, value in result.items() if key not in ('keypoints', 'valid')}
        details.append(detail)
        views.append(dict(view, joints=joints, refinement=detail,
                          source_image=raw_image['path'], source_image_sha256=raw_image['sha256'],
                          roi_source=result.get('roi_source', 'fpga_coarse_input'), source_2d_backend='pc-onnxruntime-'+backend))
    refined = dict(row, views=views, model_id=MODEL_ID, source_fpga_model=row['model_id'],
                source_fpga_views=fpga_views,
                refinement=dict(mode='fpga_guided_rtmpose', model_sha256=MODEL_SHA256,
                                source_timestamps_preserved=True,
                                raw_images=record['raw_images'], views=details))
    if hip_flip_check:
        from .hip_flip_refiner import recover_hips
        return recover_hips(refined, refiner, calibration)
    return refined


class HybridOnlineSession(OnlineSession):
    def __init__(self, calibration, checkpoint, binding, output, *, max_delta_ms=20., backend='cpu', hip_flip_check=False, max_reprojection_px=5.):
        if max_delta_ms != 20.:
            raise ValueError('The hybrid candidate requires the declared 20 ms pairing policy.')
        if type(hip_flip_check) is not bool: raise ValueError('hip_flip_check must be boolean')
        self.hybrid_hip_flip_check = hip_flip_check
        self.hip_flip_counts = dict(trigger_pairs=0, adopt_pairs=0, adopted_hips=0, extra_forward_calls=0)
        cv2.setNumThreads(1)
        activate_runtime(backend)
        refiner_class = PoseRefiner
        if hip_flip_check:
            from .hip_flip_refiner import HipFlipRefiner
            refiner_class = HipFlipRefiner
        self.refiner = refiner_class(Path(__file__).parent / 'models' / 'rtmpose_m' / 'end2end.onnx', backend=backend)
        self.refinement_preparation = self.refiner.warmup()
        self.refinement_backend = backend
        self.refinement_timings = []
        self.bbox_unavailable = dict(left=0, right=0)
        self.submitted = 0
        self.drop_events = []
        super().__init__(calibration, checkpoint, binding, output, max_delta_ms=max_delta_ms, queue_capacity=1,
                         max_reprojection_px=max_reprojection_px)

    def submit(self, record):
        super().submit(record)
        self.submitted += 1

    def _record_drop(self, record):
        super()._record_drop(record)
        message = record['message']
        self.drop_events.append(dict(reason='superseded_by_latest_complete_pair',
            boot_id=message['boot_id'], session_id=message['session_id'], pair_id=message['pair_id'],
            pc_receive_ns=record['pc_receive_ns'], cameras=[{key: camera[key] for key in
                ('camera_id','frame_id','capture_timestamp_us')} for camera in message['cameras']],
            raw_images=record['raw_images']))

    def run(self):
        decoder = None
        try:
            with (self.output / 'skeleton3d.jsonl').open('x', encoding='utf-8') as skeleton, (self.output / 'pose2d.jsonl').open('x', encoding='utf-8') as poses:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        record = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    started = time.perf_counter_ns()
                    message = record['message']
                    session_id = self.binding.get('session_id', message['session_id'])
                    validate_message(message, session_id, self.engine.stereo.calibration['image_size'], 20000)
                    if decoder is None:
                        self.binding.setdefault('session_id', message['session_id'])
                        decoder = PacketDecoder(self.engine.stereo, self.binding)
                        write_json(self.output / 'session_binding.json', self.binding)
                    original = decoder.decode(record, self.count + 1)
                    if original is None:
                        raise ValueError('Expected a validated original FPGA pose packet.')
                    row = refine_row(original, record, self.refiner, hip_flip_check=self.hybrid_hip_flip_check,
                                     calibration=self.engine.stereo.calibration)
                    refined = time.perf_counter_ns()
                    packet = self.engine.process(row, 'zynq_v4l2_monotonic:' + self.binding['boot_id'], str(session_id))
                    done = time.perf_counter_ns()
                    packet.update(pc_receive_ns=record['pc_receive_ns'], pc_complete_ns=done,
                                  receive_to_3d_ms=(done - record['pc_receive_ns']) / 1e6,
                                  queue_dropped=self.dropped, source_fpga_model=row['source_fpga_model'],
                                  refinement=row['refinement'], refinement_ms=(refined - started) / 1e6,
                                  hybrid_compute_ms=(done - started) / 1e6)
                    if self.hybrid_hip_flip_check:
                        audit = row['hip_flip_recovery']
                        packet['hip_flip_recovery'] = audit
                        self.hip_flip_counts['trigger_pairs'] += int(audit['triggered'])
                        self.hip_flip_counts['adopt_pairs'] += int(bool(audit['adopted']))
                        self.hip_flip_counts['adopted_hips'] += len(audit['adopted'])
                        self.hip_flip_counts['extra_forward_calls'] += audit['extra_forward_calls']
                    self.count += 1
                    self.timings.append(packet['compute_ms'])
                    self.latencies.append(packet['receive_to_3d_ms'])
                    self.refinement_timings.append(packet['refinement_ms'])
                    for view in row['views']:
                        self.bbox_unavailable[view['camera_id']] += int(view['refinement']['status'] == 'bbox_unavailable')
                    poses.write(json.dumps(row, allow_nan=False) + '\n'); poses.flush()
                    skeleton.write(json.dumps(packet, allow_nan=False) + '\n'); skeleton.flush()
                    self.latest = packet
                    try:
                        write_json(self.output / 'latest.json', packet)
                    except PermissionError:
                        self.preview_updates_skipped += 1
        except Exception:
            self.error = traceback.format_exc()
            (self.output / 'error.txt').write_text(self.error, encoding='utf-8')

    def finish(self):
        report = super().finish()
        report.update(hybrid_hip_flip_check=self.hybrid_hip_flip_check,hip_flip_counts=self.hip_flip_counts,
                      mode='fpga_guided_rtmpose', source_2d_model=MODEL_ID, refinement_backend=self.refinement_backend,
                      refinement_preparation=self.refinement_preparation,
                      fpga_model_sha256=self.binding['model_sha256'], raw_fpga_preserved=True,
                      raw_geometry_preserved=True, geometry_scope='Original calibration; explicit reconstruction_reprojection_limit_px applies to features and triangulation; refined 2D inputs.',
                      source_timestamps_preserved=True, bbox_unavailable=self.bbox_unavailable,
                      submitted_frames=self.submitted, drop_events=self.drop_events,
                      processing_complete=self.count+self.dropped==self.submitted,
                      refinement_p95_ms=float(np.percentile(self.refinement_timings, 95)) if self.refinement_timings else None)
        write_json(self.output / 'summary.json', report)
        return report
