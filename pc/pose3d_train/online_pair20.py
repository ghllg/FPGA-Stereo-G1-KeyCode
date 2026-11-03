"""Explicit 20 ms candidate of the frozen causal processor; no camera/UART access."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np
import torch
from mocap.io import COCO17, frame_packet, write_json
from mocap.reconstruction import triangulate
from .board_adapter import PacketDecoder
from .geometry_reprojection import BODY, Stereo
from .infer import complete_missing, retry_wrist_conflicts
from .model import load_model


class SkeletonEngine:
    def __init__(self, calibration, checkpoint, retry_conflicting_wrists=False, stabilize_arms=False,
                 max_delta_ms=10., max_reprojection_px=5.):
        if max_delta_ms not in (10., 16.667, 20.):
            raise ValueError('Use a declared 10 ms, 16.667 ms or 20 ms reconstruction policy.')
        self.max_delta_ms = max_delta_ms
        self.stereo = Stereo(calibration, max_reprojection_px=max_reprojection_px)
        self.model, saved = load_model(checkpoint)
        self.candidate_base = saved.get('feature_policy')=='geometry_candidate_v1'
        self.single_hip_anchor = saved.get('single_hip_anchor',False)
        self.single_view_ray = saved.get('single_view_ray',False)
        self.reject_conflicting_view = saved.get('reject_conflicting_view',False)
        self.retry_conflicting_wrists = retry_conflicting_wrists or stabilize_arms
        self.forearm_lengths = {9: [], 10: []}
        self.arm_stabilizer = None
        if stabilize_arms:
            from .arm_stability import ArmStabilizer
            self.arm_stabilizer = ArmStabilizer(self.stereo)
        if saved['calibration_hash'] != self.stereo.hash:
            raise ValueError('Checkpoint/calibration mismatch.')
        self.model_id = 'pose3d_mlp:' + hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]
        self.previous = -1
        torch.set_num_threads(1)
        with torch.inference_mode():
            for _ in range(10): self.model(torch.zeros(1,147), torch.zeros(1,12,3))

    def process(self, row, clock_id, session_id):
        started = time.perf_counter_ns()
        if row['timestamp_ns'] <= self.previous:
            raise ValueError('Source timestamps must strictly increase.')
        if self.retry_conflicting_wrists and self.previous >= 0 and row['timestamp_ns']-self.previous > 250_000_000:
            self.forearm_lengths = {9: [], 10: []}
        left, right = row['views']
        delta = (right['timestamp_ns']-left['timestamp_ns'])/1e6
        if abs(delta) > self.max_delta_ms:
            joints = [dict(xyz=None,valid=False,confidence=0.,estimated=False,reason='pair_time_mismatch') for _ in COCO17]
        else:
            pixels = np.array([[[j['xy'] if j['valid'] else [0.,0.] for j in v['joints'][5:]] for v in row['views']]])
            confidence = np.array([[[j['confidence'] if j['valid'] else 0. for j in v['joints'][5:]] for v in row['views']]])
            x, base, observed, _ = self.stereo.features(pixels, confidence, candidate_base=self.candidate_base)
            with torch.inference_mode():
                prediction = self.model(torch.from_numpy(x),torch.from_numpy(base)).numpy()
            hybrid, _, completed = complete_missing(self.stereo,prediction,base,observed,pixels,confidence,
                                                    single_hip_anchor=self.single_hip_anchor,single_view_ray=self.single_view_ray,
                                                    conflict_model=self.model if self.reject_conflicting_view else None)
            joints = triangulate(left['joints'],right['joints'],self.stereo.calibration,
                                 max_reprojection_px=self.stereo.max_reprojection_px)
            for body_index,coco_index in enumerate(BODY):
                if completed[0,body_index]:
                    joints[coco_index] = dict(xyz=hybrid[0,body_index].astype(float).tolist(),valid=True,
                                             estimated=True,confidence=.2,reason='learned_body_completion')
        for joint in joints:
            if joint['valid']:joint['raw_xyz'] = list(joint['xyz'])
        if self.retry_conflicting_wrists and abs(delta) <= self.max_delta_ms:
            retry_wrist_conflicts(self.stereo, self.model, joints, observed, pixels, confidence,
                                  self.forearm_lengths, self.candidate_base, self.single_hip_anchor, self.single_view_ray)
        if self.arm_stabilizer is not None:
            if abs(delta) <= self.max_delta_ms:
                self.arm_stabilizer.update(joints, row['views'], row['timestamp_ns'])
            else:
                self.arm_stabilizer.reset()
        self.previous = row['timestamp_ns']
        result = frame_packet(row['seq'],row['timestamp_ns'],joints,session_id=session_id,
                              calibration_id=self.stereo.calibration['calibration_id'],model_id=self.model_id,
                              coordinate_system=self.stereo.calibration['world_coordinate_system'],clock_id=clock_id)
        result.update(source='pose3d_train_online',source_2d_model=row['model_id'],source_seq=row['seq'],
                      source_frames=[v['frame_index'] for v in row['views']],pair_delta_ms=delta,
                      experimental=True,causal=True,filter=dict(type='none',tau_s=0.))
        result['reconstruction_reprojection_limit_px'] = self.stereo.max_reprojection_px
        result['compute_ms'] = (time.perf_counter_ns()-started)/1e6
        if self.max_delta_ms != 10.:
            result.update(reconstruction_pair_limit_ms=self.max_delta_ms,
                          extended_pair_admission=abs(delta) > 10.,
                          exposure_sync_verified=False)
        if self.retry_conflicting_wrists:
            result['wrist_correction_policy'] = 'wrist_conflict_retry_v1'
        if self.arm_stabilizer is not None:
            result['arm_correction_policy'] = 'arm_length_fit_v3_reliability'
        return result


def draw_skeleton(packet, status='', origin=None):
    """Three orthographic views of the same world-metre skeleton."""
    import cv2
    canvas = np.full((620,1200,3),(30,24,20),np.uint8)
    cv2.putText(canvas,status[:150],(20,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1)
    if packet is None:return canvas
    joints = packet['joints']
    valid = np.array([j['valid'] for j in joints])
    xyz = np.array([j['xyz'] if j['valid'] else [np.nan]*3 for j in joints])
    if origin is not None: center = np.asarray(origin)
    elif valid[11:13].all(): center = xyz[11:13].mean(axis=0)
    elif valid.any(): center = np.nanmean(xyz,axis=0)
    else:return canvas
    edges = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for column,(axis,label) in enumerate([(1,'FRONT Y/Z'),(0,'SIDE X/Z'),(None,'OBLIQUE')]):
        h = xyz[:,axis]-center[axis] if axis is not None else .8*(xyz[:,1]-center[1])+.6*(xyz[:,0]-center[0])
        xy = np.column_stack([200+400*column-h*230,350-(xyz[:,2]-center[2])*230])
        cv2.putText(canvas,label,(400*column+90,70),cv2.FONT_HERSHEY_SIMPLEX,.6,(180,180,180),1)
        for a,b in edges:
            if valid[a] and valid[b]:cv2.line(canvas,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),(180,180,180),2)
        for i in range(17):
            if valid[i]:cv2.circle(canvas,tuple(np.rint(xy[i]).astype(int)),5,(50,180,255) if joints[i].get('estimated') else (170,220,40),-1)
    return canvas


class OnlineSession:
    """Four queued packets maximum; raw receiver logging stays independent."""
    def __init__(self, calibration, checkpoint, binding, output, *, max_delta_ms=10., queue_capacity=4, max_reprojection_px=5.):
        if type(queue_capacity) is not int or queue_capacity not in (1, 4):
            raise ValueError('Use the declared one-pair hybrid or four-packet ordinary queue.')
        self.engine = SkeletonEngine(calibration,checkpoint,max_delta_ms=max_delta_ms,
                                     max_reprojection_px=max_reprojection_px)
        self.binding = dict(binding)
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.dropped = 0
        self.count = 0
        self.preview_updates_skipped = 0
        self.timings = []
        self.latencies = []
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def submit(self, record):
        try:self.queue.put_nowait(record)
        except queue.Full:
            try:self._record_drop(self.queue.get_nowait())
            except queue.Empty:pass
            self.queue.put_nowait(record)

    def _record_drop(self, record):
        self.dropped += 1

    def run(self):
        decoder = None
        try:
            with (self.output/'skeleton3d.jsonl').open('x',encoding='utf-8') as skeleton, (self.output/'pose2d.jsonl').open('x',encoding='utf-8') as poses:
                while not self.stop.is_set() or not self.queue.empty():
                    try:record = self.queue.get(timeout=.05)
                    except queue.Empty:continue
                    if decoder is None:
                        # Boot/model/geometry come from preflight; only the fresh session id comes from the first packet.
                        self.binding.setdefault('session_id',record['message']['session_id'])
                        decoder = PacketDecoder(self.engine.stereo,self.binding)
                        write_json(self.output/'session_binding.json',self.binding)
                    row = decoder.decode(record,self.count+1)
                    if row is None:continue
                    packet = self.engine.process(row,'zynq_v4l2_monotonic:'+self.binding['boot_id'],str(self.binding['session_id']))
                    done = time.perf_counter_ns()
                    packet.update(pc_receive_ns=record['pc_receive_ns'],pc_complete_ns=done,
                                  receive_to_3d_ms=(done-record['pc_receive_ns'])/1e6,queue_dropped=self.dropped)
                    self.count += 1
                    self.timings.append(packet['compute_ms'])
                    self.latencies.append(packet['receive_to_3d_ms'])
                    poses.write(json.dumps(row,allow_nan=False)+'\n');poses.flush()
                    skeleton.write(json.dumps(packet,allow_nan=False)+'\n');skeleton.flush()
                    self.latest = packet
                    pending = self.output/'latest.tmp'
                    pending.write_text(json.dumps(packet,allow_nan=False),encoding='utf-8')
                    try:pending.replace(self.output/'latest.json')
                    except PermissionError:
                        # Windows viewers can briefly hold a file without delete sharing.
                        # Durable pose logs above remain complete; refresh on the next frame.
                        self.preview_updates_skipped += 1
        except Exception:
            self.error = traceback.format_exc()
            (self.output/'error.txt').write_text(self.error,encoding='utf-8')

    def finish(self):
        self.stop.set();self.thread.join(10)
        if self.thread.is_alive():raise RuntimeError('Online processor did not stop.')
        report = dict(frames=self.count,dropped_queue_packets=self.dropped,queue_capacity=self.queue.maxsize,error=self.error,
                      preview_updates_skipped=self.preview_updates_skipped,
                      causal=True,raw_geometry_preserved=True,
                      reconstruction_pair_limit_ms=self.engine.max_delta_ms,exposure_sync_verified=False,
                      reconstruction_reprojection_limit_px=self.engine.stereo.max_reprojection_px,
                      compute_p95_ms=float(np.percentile(self.timings,95)) if self.timings else None,
                      receive_to_3d_p95_ms=float(np.percentile(self.latencies,95)) if self.latencies else None,
                      latency_scope='PC complete network receipt to computed skeleton; excludes camera, board and display.')
        write_json(self.output/'summary.json',report)
        if self.error:raise RuntimeError(self.error)
        return report


def replay_packets(source, calibration, checkpoint, binding, output, retry_conflicting_wrists=False, stabilize_arms=False):
    """Offline equivalence check, preserving ordering without simulated clock latency."""
    engine = SkeletonEngine(calibration,checkpoint,retry_conflicting_wrists=retry_conflicting_wrists,
                            stabilize_arms=stabilize_arms)
    decoder = PacketDecoder(engine.stereo,json.loads(Path(binding).read_text()))
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    packets=[]
    with (output/'skeleton3d.jsonl').open('x',encoding='utf-8') as stream:
        for number,line in enumerate(Path(source).read_text().splitlines(),1):
            row = decoder.decode(json.loads(line),number)
            if row is None:continue
            packet = engine.process(row,'zynq_v4l2_monotonic:'+decoder.bound['boot_id'],str(decoder.bound['session_id']))
            packets.append(packet);stream.write(json.dumps(packet,allow_nan=False)+'\n')
    report=dict(frames=len(packets),compute_p95_ms=float(np.percentile([p['compute_ms'] for p in packets],95)),
                source='saved packets; no cameras accessed',causal=True)
    if retry_conflicting_wrists:
        report['wrist_correction_policy'] = 'wrist_conflict_retry_v1'
    if stabilize_arms:
        report['arm_correction_policy'] = 'arm_length_fit_v2_continuous'
        report['wrist_correction_policy'] = 'wrist_conflict_retry_v1'
    write_json(output/'summary.json',report)
    import cv2
    preview=draw_skeleton(packets[-1],f"SAVED PACKET REPLAY | {len(packets)} pairs | geometry green / learned orange")
    (output/'preview.jpg').write_bytes(cv2.imencode('.jpg',preview)[1].tobytes())
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','calibration','checkpoint','binding','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--retry-conflicting-wrists',action='store_true')
    parser.add_argument('--stabilize-arms',action='store_true',help='Experimental arm-length correction, including wrist conflict retry.')
    args=parser.parse_args()
    print(json.dumps(replay_packets(**vars(args))))
