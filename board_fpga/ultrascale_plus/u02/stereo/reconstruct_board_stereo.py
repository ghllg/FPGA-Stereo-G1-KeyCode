"""Reuse existing calibrated triangulation/filter and shared Genesis skeleton wire.

Input is board pose JSONL. This module never loads a 2D neural network.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'windows_vision'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mocap.calibration import load_calibration
from mocap.reconstruction import triangulate, CausalFilter
from mocap.io import frame_packet
from board_pose_protocol import StereoPairer,pairing_time

class BoardStereoReconstructor:
    def __init__(self, calibration, max_delta_ms=10, capacity=4, max_age_ms=200,
                 min_confidence=.35, max_reprojection_px=5, tau=.04):
        if not all(math.isfinite(x) for x in (min_confidence,max_reprojection_px,tau)) or not 0<=min_confidence<=1 or max_reprojection_px<=0 or tau<0:
            raise ValueError('Invalid reconstruction thresholds')
        self.calibration = calibration
        self.pairer = StereoPairer(calibration['camera_ids'], max_delta_ms, capacity, max_age_ms)
        self.filter = CausalFilter(tau)
        self.confidence = min_confidence
        self.reprojection = max_reprojection_px
        self.sequence = 0

    def push(self, packet):
        if packet.get('source_size') != self.calibration['image_size']:
            raise ValueError('Calibration image dimensions do not match source pixels')
        pair = self.pairer.push(packet)
        if pair is None: return None
        start = time.perf_counter_ns()
        views = [pair['left'], pair['right']]
        joints2d = [[dict(xy=p[:2], confidence=p[2], valid=p[2]>=self.confidence)
                     for p in view['keypoints']] for view in views]
        joints = triangulate(*joints2d, self.calibration, min_confidence=self.confidence,
                             max_reprojection_px=self.reprojection)
        timestamp = max(pairing_time(v) for v in views)
        self.filter.apply(joints, timestamp)
        result = frame_packet(self.sequence, timestamp, joints,
            session_id=views[0]['session_id'], calibration_id=self.calibration['calibration_id'],
            model_id='PP-TinyPose:' + views[0]['model_sha256'],
            coordinate_system=self.calibration.get('world_coordinate_system') or self.calibration['coordinate_system'],
            clock_id=views[0].get('pairing_clock_id',views[0]['clock_id']))
        result.update(source='u02_board_stereo', input_mode=views[0]['input_mode'],
            whole_2d_networks_on_board=True, backend=views[0]['backend'],
            camera_ids=[v['camera_id'] for v in views], source_frames=[v['frame_id'] for v in views],
            capture_timestamps_ns=[v['capture_timestamp_ns'] for v in views],
            pairing_timestamps_ns=[pairing_time(v) for v in views],
            time_alignment=[v.get('time_alignment') for v in views],
            timestamp_sources=[v['timestamp_source'] for v in views],
            source_roi_xywh=[v['roi_xywh'] for v in views],
            pair_delta_ms=pair['pair_delta_ns']/1e6, hardware_sync=False,
            raw_pair_delta_ms=pair['raw_pair_delta_ns']/1e6,pairing_method=pair['pairing'],
            exposure_timestamps_verified=False, pairing_status='matched',
            pl_calls=[v['pl_calls'] for v in views], forward_ms=[v['forward_ms'] for v in views],
            inference_clocks=[v['inference_clock_id'] for v in views],
            inference_interval_scopes=[v.get('inference_interval_scope','unspecified') for v in views],
            inference_intervals_ns=[[v['inference_start_ns'],v['inference_end_ns']] for v in views],
            anomalies=[v['anomalies'] for v in views],
            reconstruction_ms=(time.perf_counter_ns()-start)/1e6,
            calibration_physical_mount_verified=False)
        self.sequence += 1
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-delta-ms', type=float, default=10)
    args=parser.parse_args()
    assert not args.output.exists(), 'Preserve existing results'
    processor=BoardStereoReconstructor(load_calibration(args.calibration), max_delta_ms=args.max_delta_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding='utf-8') as source, args.output.open('x',encoding='utf-8') as destination:
        for line in iter(lambda: source.readline(65537), ''):
            if len(line)>65536:raise ValueError('Pose message exceeds bounded size')
            result=processor.push(json.loads(line))
            if result is not None:destination.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(processor.pairer.stats))

if __name__=='__main__':main()
