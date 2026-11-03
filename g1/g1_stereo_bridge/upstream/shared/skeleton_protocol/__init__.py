"""Cross-end skeleton contract. No camera, Windows, GPU or robot dependency."""
import json
import math

COCO17 = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle')
VERSION = 'human_skeleton/1.0'


def validate(packet):
    """Reject ambiguous units, non-finite values, reordered joints and fake zeros."""
    if packet.get('schema') != VERSION or packet.get('unit') != 'm':
        raise ValueError('骨架协议版本或单位错误。')
    if packet.get('joint_names') != list(COCO17):
        raise ValueError('关节顺序必须为 COCO17。')
    for key in ('seq', 'timestamp_ns'):
        if type(packet.get(key)) is not int or packet[key] < 0:
            raise ValueError(f'{key} 必须是非负整数。')
    for key in ('session_id', 'source', 'clock_id', 'coordinate_system', 'calibration_id', 'model_id'):
        if not isinstance(packet.get(key), str) or not packet[key]:
            raise ValueError(f'缺少 {key}。')
    joints = packet.get('joints', [])
    if len(joints) != 17:
        raise ValueError('必须提供 17 个关节。')
    for joint in joints:
        if type(joint.get('valid')) is not bool or type(joint.get('estimated')) is not bool:
            raise ValueError('valid / estimated 必须是布尔值。')
        xyz = joint.get('xyz')
        if joint['valid']:
            if not isinstance(xyz, list) or len(xyz) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in xyz):
                raise ValueError('有效关节必须有有限的 XYZ 坐标。')
        elif xyz is not None:
            raise ValueError('无效关节的 xyz 必须为 null。')
        score = joint.get('confidence')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('confidence 必须在 0 到 1 之间。')
    return packet


def encode(packet):
    return json.dumps(validate(packet), ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def decode(data):
    return validate(json.loads(data))

