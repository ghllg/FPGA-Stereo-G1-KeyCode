"""Board-only 2D results and bounded software pairing; no neural inference."""
from collections import deque
import math
import re

SCHEMA = 'u02_board_pose/1.0'

def validate(packet):
    if packet.get('schema') != SCHEMA or packet.get('whole_network_on_board') is not True:
        raise ValueError('Require a board whole-network pose result')
    for key in ('camera_id', 'session_id', 'clock_id', 'timestamp_source', 'model_sha256'):
        if not isinstance(packet.get(key), str) or not packet[key]:
            raise ValueError('Missing ' + key)
    for key in ('frame_id', 'capture_timestamp_ns', 'inference_start_ns', 'inference_end_ns', 'pl_calls'):
        if type(packet.get(key)) is not int or packet[key] < 0:
            raise ValueError('Invalid ' + key)
    if packet['inference_end_ns'] < packet['inference_start_ns']:
        raise ValueError('Inference timestamps reversed')
    # Capture and inference clocks may differ, especially for recorded inputs.
    if not isinstance(packet.get('inference_clock_id'), str) or not packet['inference_clock_id']:
        raise ValueError('Inference clock must be explicit')
    if packet.get('input_mode') not in ('recorded', 'live'):
        raise ValueError('Input mode must distinguish recorded and live')
    if packet.get('backend') not in ('cpu', 'pl-stream'):
        raise ValueError('Only actual board backends accepted')
    if packet['backend'] == 'pl-stream' and packet['pl_calls'] <= 0:
        raise ValueError('PL inference must report actual calls')
    if packet['backend'] == 'cpu' and packet['pl_calls'] != 0:
        raise ValueError('CPU result cannot report PL calls')
    if type(packet.get('forward_ms')) not in (float, int) or not math.isfinite(packet['forward_ms']) or packet['forward_ms'] < 0:
        raise ValueError('Invalid measured forward time')
    if packet.get('pairing_status') != 'unpaired' or not isinstance(packet.get('anomalies'), list) or any(not isinstance(x, str) for x in packet['anomalies']):
        raise ValueError('Explicit pairing state and anomaly list required')
    if packet.get('coordinate_space') != 'original_image_pixels':
        raise ValueError('Require original image coordinates')
    size = packet.get('source_size', [])
    if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
        raise ValueError('Invalid source dimensions')
    roi = packet.get('roi_xywh', [])
    if len(roi) != 4 or not all(type(x) in (float, int) and math.isfinite(x) for x in roi) or min(roi[2:]) <= 0:
        raise ValueError('Invalid ROI mapping')
    points = packet.get('keypoints', [])
    if len(points) != 17 or any(len(p) != 3 or not all(type(x) in (float, int) and math.isfinite(x) for x in p) or not 0 <= p[2] <= 1 for p in points):
        raise ValueError('Require finite COCO17 coordinates and confidence')
    if packet.get('joint_order') != 'COCO17' or packet.get('exposure_timestamp_verified') is not False:
        raise ValueError('Unverified exposure timestamp must remain explicit')
    if 'pairing_timestamp_ns' in packet:
        alignment=packet.get('time_alignment',{})
        if packet['input_mode']!='recorded' or alignment.get('method')!='offline_content_offset_nearest_actual_frame':
            raise ValueError('Time correction limited to explicit recorded alignment')
        if not re.fullmatch(r'[0-9a-f]{64}',alignment.get('proof_sha256','')):
            raise ValueError('Alignment proof hash required')
        offset=alignment.get('offset_ns')
        if type(offset) is not int or abs(offset)>10_000_000_000 or type(packet['pairing_timestamp_ns']) is not int or packet['pairing_timestamp_ns']<0:
            raise ValueError('Invalid time correction')
        if packet['pairing_timestamp_ns']!=packet['capture_timestamp_ns']-offset or not packet.get('pairing_clock_id'):
            raise ValueError('Raw capture time must remain preserved')
    elif packet.get('time_alignment') or packet.get('pairing_clock_id'):
        raise ValueError('Incomplete time correction metadata')
    return packet

def pairing_time(packet):
    return packet.get('pairing_timestamp_ns',packet['capture_timestamp_ns'])

class StereoPairer:
    """Consume each frame once, reject clock ambiguity, bound queued frames."""
    def __init__(self, camera_ids, max_delta_ms=10, capacity=4, max_age_ms=200):
        if len(camera_ids) != 2 or camera_ids[0] == camera_ids[1]:
            raise ValueError('Two distinct cameras required')
        if not math.isfinite(max_delta_ms) or max_delta_ms < 0 or not math.isfinite(max_age_ms) or max_age_ms <= 0 or type(capacity) is not int or capacity < 1:
            raise ValueError('Invalid pairing bounds')
        self.ids = tuple(camera_ids)
        self.delta = round(max_delta_ms * 1e6)
        self.age = round(max_age_ms * 1e6)
        self.capacity = capacity
        self.queues = [deque(), deque()]
        self.previous = [None, None]
        self.identity = None
        self.latest = -1
        self.stats = dict(received=0, paired=0, dropped_expired=0, dropped_capacity=0, dropped_time_mismatch=0)

    def push(self, packet):
        validate(packet)
        if packet['camera_id'] not in self.ids:
            raise ValueError('Unexpected physical camera identity')
        index = self.ids.index(packet['camera_id'])
        identity = tuple(packet[k] for k in ('session_id', 'clock_id', 'timestamp_source', 'model_sha256', 'backend', 'input_mode'))
        identity += (packet.get('pairing_clock_id'),packet.get('time_alignment',{}).get('proof_sha256'))
        if self.identity is not None and identity != self.identity:
            raise ValueError('Cannot mix sessions, clocks, models or backends')
        prior = self.previous[index]
        stamp = pairing_time(packet)
        if prior is not None and (stamp <= prior[0] or packet['frame_id'] <= prior[1]):
            raise ValueError('Frames must increase; duplicates cannot be reused')
        self.identity = identity
        self.previous[index] = (stamp, packet['frame_id'])
        self.latest = max(self.latest, stamp)
        self.stats['received'] += 1
        self.queues[index].append(packet)
        for queue in self.queues:
            while queue and self.latest - pairing_time(queue[0]) > self.age:
                queue.popleft(); self.stats['dropped_expired'] += 1
            while len(queue) > self.capacity:
                queue.popleft(); self.stats['dropped_capacity'] += 1
        while all(self.queues):
            left, right = [queue[0] for queue in self.queues]
            delta = pairing_time(right) - pairing_time(left)
            if abs(delta) <= self.delta:
                for queue in self.queues: queue.popleft()
                self.stats['paired'] += 1
                return dict(left=left, right=right, pair_delta_ns=delta,
                            raw_pair_delta_ns=right['capture_timestamp_ns']-left['capture_timestamp_ns'],
                            pairing='offline_content_offset_nearest_actual_frame' if left.get('time_alignment') else 'software_capture_timestamp',
                            pairing_status='matched', hardware_sync=False)
            self.queues[0 if delta > 0 else 1].popleft()
            self.stats['dropped_time_mismatch'] += 1
        return None
