"""Convert saved board packets using an explicit physical session binding; no hardware IO."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .geometry import Stereo
from mocap.io import COCO17, read_json, write_json


class PacketDecoder:
    """Stateful validation; a foreign session or reordered frame is rejected."""
    def __init__(self, stereo, bound):
        cal = stereo.calibration
        if (bound.get('schema') != 'board_session_binding/1.0'
                or bound.get('calibration_hash') != stereo.hash
                or not cal.get('camera_physical_ids')
                or bound.get('camera_physical_ids') != cal['camera_physical_ids']):
            raise ValueError('Session binding must match calibration and physical camera order.')
        model_hash = bound.get('model_sha256', '')
        if len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            raise ValueError('Binding requires the verified 2D model SHA256.')
        if not bound.get('boot_id') or bound.get('session_id') is None:
            raise ValueError('Binding requires boot_id and session_id from this recording.')
        self.cal, self.bound, self.model_hash = cal, bound, model_hash
        self.model_id = 'PP-TinyPose:' + model_hash[:16]
        self.previous = self.previous_pair = -1
        self.rows = self.skipped = self.capped = 0
        self.exposure_flags = set()
        self.hardware_messages = True

    def decode(self, record, line_number):
        cal, bound, model_hash, model_id = self.cal, self.bound, self.model_hash, self.model_id
        msg = record.get('message', record)
        if msg.get('schema') != 2 or msg.get('message_type') != 'paired_coco17':
            self.skipped += 1
            return None
        self.hardware_messages = self.hardware_messages and msg.get('backend') == 'pl-stream' and msg.get('whole_network_on_board') is True
        if msg.get('boot_id') != bound['boot_id'] or msg.get('session_id') != bound['session_id']:
            raise ValueError(f'Foreign boot/session at line {line_number}; do not mix recordings.')
        if msg.get('keypoint_order') != 'COCO17' or msg.get('model_sha256', model_hash) != model_hash:
            raise ValueError('Joint order or model binding mismatch.')
        pair_id = msg.get('pair_id')
        if type(pair_id) is not int or pair_id <= self.previous_pair:
            raise ValueError('Pair ids must strictly increase; duplicates/restarts require a separate file.')
        self.previous_pair = pair_id
        cameras = msg.get('cameras', [])
        if [v.get('camera_id') for v in cameras] != cal['camera_ids']:
            raise ValueError('Camera order mismatch.')
        threshold = msg.get('valid_confidence_threshold')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid source confidence threshold.')
        views = []
        for camera in cameras:
            if 'roi_xywh' in bound:
                expected_roi = bound['roi_xywh']
                if isinstance(expected_roi,dict): expected_roi = expected_roi.get(camera['camera_id'])
                if camera.get('roi_xywh') != expected_roi:
                    raise ValueError('Camera ROI differs from the session binding.')
            if (camera.get('source_size') != cal['image_size']
                    or camera.get('coordinate_space') != 'source_image_pixels'
                    or camera.get('timestamp_source') != 'v4l2_monotonic_unverified_exposure'
                    or camera.get('model_sha256', model_hash) != model_hash):
                raise ValueError('Original image coordinates, resolution, clock and model must match.')
            stamp, frame = camera.get('capture_timestamp_us'), camera.get('frame_id')
            if type(stamp) is not int or stamp < 0 or type(frame) is not int or frame < 0:
                raise ValueError('Invalid source timestamp/frame id.')
            points = camera.get('keypoints', [])
            if len(points) != 17:
                raise ValueError('Expected exactly 17 keypoints.')
            joints = []
            for point in points:
                if len(point) != 4:
                    raise ValueError('Expected x,y,heatmap_score,valid.')
                x, y, score, valid = point
                if (type(valid) is not bool or any(type(v) not in (int,float) or not math.isfinite(v) for v in [x,y,score])
                        or score < 0):
                    raise ValueError('Invalid keypoint fields.')
                if valid and (score < threshold or not 0 <= x < cal['image_size'][0] or not 0 <= y < cal['image_size'][1]):
                    raise ValueError('Valid flag contradicts source threshold/image bounds.')
                joints.append(dict(xy=[x,y] if valid else None, confidence=min(score,1.), valid=valid,
                                   raw_heatmap_score=score, raw_xy=[x,y]))
                self.capped += int(score > 1)
            views.append(dict(camera_id=camera['camera_id'], image_size=cal['image_size'],
                              frame_index=frame, timestamp_ns=stamp*1000, joints=joints,
                              roi_xywh=camera.get('roi_xywh')))
        delta_us = (views[1]['timestamp_ns']-views[0]['timestamp_ns'])//1000
        if msg.get('pair_delta_us') != delta_us:
            raise ValueError('Packet delta disagrees with original capture timestamps.')
        timestamp = views[0]['timestamp_ns']
        if timestamp <= self.previous:
            raise ValueError('Left source timestamps must strictly increase.')
        self.previous = timestamp
        self.exposure_flags.add(bool(msg.get('exposure_sync_verified', False)))
        row = dict(schema='pose2d/1.0', seq=self.rows, source_pair_id=pair_id,
                         source_line=line_number, joint_names=list(COCO17), model_id=model_id,
                         timestamp_ns=timestamp, pair_delta_ms=delta_us/1000, views=views)
        self.rows += 1
        return row


def convert(source, calibration, binding, output):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix('.meta.json').exists():
        raise ValueError('Output exists; use a new path.')
    stereo = Stereo(calibration)
    bound = read_json(binding)
    cal = stereo.calibration
    decoder = PacketDecoder(stereo, bound)
    model_hash, model_id = decoder.model_hash, decoder.model_id
    rows = []
    for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            row = decoder.decode(json.loads(line), line_number)
            if row is not None: rows.append(row)
    exposure_flags, capped, skipped = decoder.exposure_flags, decoder.capped, decoder.skipped
    hardware_messages = decoder.hardware_messages
    if not rows:
        raise ValueError('No paired_coco17 rows.')
    meta = dict(schema='pose2d_manifest/1.0', state='complete', rows=len(rows), camera_ids=cal['camera_ids'],
                image_size=cal['image_size'], calibration_id=cal['calibration_id'],
                calibration_hash=stereo.hash, model_id=model_id, model_sha256=model_hash,
                source_session=str(source.parent / str(bound['session_id'])), source_file=str(source.resolve()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), session_binding=bound,
                clock_id='zynq_v4l2_monotonic:'+bound['boot_id'], mirrored=False,
                unit='original_image_pixels', raw_timestamps_preserved=True,
                exposure_sync_verified=False, reported_exposure_flags=sorted(exposure_flags),
                score_values_capped=capped, non_pose_rows_skipped=skipped,
                temporal_alignment=None,
                note='Explicit external session binding, not physical revalidation. Raw times unchanged; inference rejects pairs over 10 ms. Heatmap scores capped only for bounded features; raw values retained.')
    meta.update(source_2d_backend='pl-stream' if hardware_messages else None,
                hardware_inference_binding=bound if hardware_messages else None,
                live_acquisition_validated=bool(hardware_messages and bound.get('live_acquisition_validated')))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    write_json(output.with_suffix('.meta.json'), meta)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['source','calibration','binding','output']:
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    result = convert(args.source, args.calibration, args.binding, args.output)
    print(json.dumps(dict(rows=result['rows'], score_values_capped=result['score_values_capped'])))
