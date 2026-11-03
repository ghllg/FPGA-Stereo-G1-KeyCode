"""Causal live processing. Acquisition, inference, UI and disk writers are separated."""
import json
import queue
import threading
import time
import uuid
from dataclasses import replace
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from windows_camera.capture import Recording, Sample
from mocap.calibration import load_calibration
from mocap.io import COCO17, frame_packet, write_json
from mocap.lafan import CORE, NAMES, PARENTS, extend_coco, offsets_from_targets, fit_rotations, export_lafan
from mocap.pose import PoseEstimator
from mocap.reconstruction import triangulate, CausalFilter
from .source import CameraSource, DemoSource, demo_calibration, demo_views
from .sync import PairBuffer, align_pulses


class Logs:
    def __init__(self, folder):
        self.folder = folder; self.queue = queue.Queue(maxsize=256)
        self.closed = threading.Event(); self.error = ''; self.rows = 0
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def submit(self, rows):
        if self.error: raise RuntimeError(self.error)
        try: self.queue.put_nowait(rows)
        except queue.Full: raise RuntimeError('骨架写入积压，停止记录；不会静默丢失记录。')

    def _run(self):
        files = {}
        try:
            files = {name: (self.folder/(name+'.jsonl')).open('x', encoding='utf-8')
                     for name in ('pose2d', 'skeleton3d', 'live_lafan')}
            while not self.closed.is_set() or not self.queue.empty():
                try: rows = self.queue.get(timeout=.05)
                except queue.Empty: continue
                for name, row in rows.items(): files[name].write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                self.rows += 1
        except Exception as exc: self.error = str(exc)
        finally:
            for f in files.values(): f.close()

    def close(self):
        self.closed.set(); self.thread.join(timeout=15)
        if self.thread.is_alive(): raise RuntimeError('骨架文件写入未结束，请保留目录并检查磁盘。')
        if self.error: raise RuntimeError(self.error)


class Engine:
    def __init__(self, config, calibration_path=None, device='auto', model=None, target_fps=30, demo=False, inference_size=640,
                 source_factory=None, replay_alignment=None, validation_record=False, refine_person=False):
        if target_fps <= 0 or target_fps > 60: raise ValueError('目标骨架帧率应在 1–60。')
        self.config, self.calibration_path = config, calibration_path
        self.device, self.model, self.target_fps, self.demo = device, model, target_fps, demo
        self.inference_size = inference_size; self.previous_quaternion = None
        self.refine_person = refine_person
        if (replay_alignment is not None or validation_record) and source_factory is None:
            raise ValueError('录像验证参数不能用于物理相机入口。')
        self.source_factory, self.replay_alignment, self.validation_record = source_factory, replay_alignment, validation_record
        self.buffer = PairBuffer(); self.lock = threading.Lock(); self.record_lock = threading.Lock()
        self.latest_frames = [None, None]; self.snapshot = dict(status='等待启动', images=[], lafan=None)
        self.commands = queue.Queue(); self.stop_event = threading.Event()
        self.source = None; self.recording = None; self.logs = None; self.folder = None
        self.error = ''; self.sync = None; self.sync_history = None; self.sync_until = 0
        self.sync_last = [-1, -1]; self.seen = [-1, -1]; self.seq = 0
        self.offsets = None; self.bone_samples = []; self.filter = CausalFilter()
        self.session_id = 'live-'+uuid.uuid4().hex; self.started = time.perf_counter()
        self.poses = 0; self.last_timestamp = -1; self.last_start = 0
        self.pose_times = deque(); self.last_pose_host = 0; self.torso_length = None

    def update(self, **values):
        with self.lock: self.snapshot = dict(self.snapshot, **values)

    def read(self):
        with self.lock: return self.snapshot.copy()

    def request(self, command, value=None): self.commands.put((command, value))

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def stop(self): self.stop_event.set()

    def fail(self, message):
        self.error = message; self.stop_event.set(); self.update(status='处理失败：'+message)

    def _receive(self, frame):
        if self.stop_event.is_set(): return
        with self.record_lock:
            if self.recording:
                frame.recording_index = self.recording.submit(Sample(frame.camera, frame.timestamp_ns, frame.receive_ns, frame.payload))
                frame.recording_id = str(self.folder)
                if self.recording.error: self.fail(self.recording.error); return
        self.buffer.add(frame)
        with self.lock: self.latest_frames[frame.camera] = frame

    def _start_recording(self, folder):
        if self.recording: raise ValueError('已经在记录。')
        if self.sync is None: raise ValueError('请先完成入口校时。')
        if not all(self.source.pixel_formats): raise ValueError('相机尚未就绪。')
        if self.offsets is None and not self.validation_record: raise ValueError('请先全身站稳，等待骨长初始化完成。')
        folder = Path(folder)
        if folder.exists(): raise FileExistsError('记录目录已存在，请使用新的输出目录。')
        with self.lock: frames = list(self.latest_frames)
        if not all(frames): raise ValueError('两路输入尚未就绪。')
        raw = Recording(folder/'raw', replace(self.config, pixel_formats=tuple(self.source.pixel_formats)),
                        min(f.timestamp_ns for f in frames))
        if not raw.ready.wait(5) or raw.error:
            raw.stop(); raise RuntimeError(raw.error or '录像写入器启动超时。')
        try: logs = Logs(folder)
        except Exception:
            raw.stop(); raise
        self.folder = folder
        self.manifest = dict(schema='windows_live_workflow/1.0', state='recording', synthetic=self.demo,
                             session_id=self.session_id, calibration_id=self.cal['calibration_id'],
                             model_id=self.model_id, camera_ids=self.cal['camera_ids'],
                             inference_image_size=self.inference_size, torch_cpu_threads=4 if not self.demo else None,
                             person_roi_refinement=self.refine_person,
                             requested_capture_fps=self.config.fps, target_skeleton_fps=self.target_fps,
                             image_size=self.cal['image_size'], synchronization=self.sync,
                             clock_id=self.session_id+':visually_corrected_source',
                             raw_video_pairing='raw/pairs.csv uses uncorrected timestamps; pose2d.jsonl has actual corrected processing pairs',
                             algorithm='YOLO COCO17 -> triangulation -> causal filter -> fixed-bone local rotations',
                             filter_tau_s=.04, bone_offsets_m=self.offsets.tolist() if self.offsets is not None else None,
                             recording_before_bone_initialization_for_validation=self.validation_record,
                             input_kind='recorded_video_replay' if self.source_factory else 'synthetic_demo' if self.demo else 'physical_camera',
                             live_uses_future_frames=False, latency_scope='host receive to completed pose; excludes pre-receive camera/USB queues',
                             lafan_joint_names=list(NAMES), lafan_parents=PARENTS.tolist(),
                             lafan_unit='cm', lafan_axes='Y_up', quaternion_order='wxyz',
                             estimated_parts='Spine, head, shoulders, toes and axial twist are inferred, not directly measured.')
        with self.record_lock: self.recording, self.logs = raw, logs
        write_json(folder/'workflow.json', self.manifest)
        write_json(folder/'calibration_snapshot.json', self.cal)
        self.buffer.clear(); self.update(recording=True, output=str(folder), status='正在记录视频与实时骨架')

    def _stop_recording(self):
        with self.record_lock:
            raw, logs = self.recording, self.logs; self.recording = self.logs = None
        if raw is None: return
        self.update(status='正在保存并导出 BVH…', recording=False)
        raw.stop(self.error)
        completed = raw.done.wait(20)
        try:
            logs.close()
            if not completed: raise RuntimeError('原始视频未完成保存。')
            if raw.error: raise RuntimeError(raw.error)
            if self.error: raise RuntimeError(self.error)
            self.manifest.update(state='complete', skeleton_rows=logs.rows, drops=list(self.buffer.dropped))
            write_json(self.folder/'skeleton3d.meta.json', dict(state='complete', **{'live_workflow': self.manifest}))
            write_json(self.folder/'workflow.json', self.manifest)
            try:
                export_lafan(self.folder/'skeleton3d.jsonl', self.folder/'lafan1', fps=self.target_fps)
                self.manifest['bvh_export'] = 'offline bounded-gap resampling after recording; live_lafan.jsonl retains actual causal poses'
                self.update(status='保存完成：视频、实时骨架与 LAFAN1 BVH', output=str(self.folder))
            except Exception as exc:
                self.manifest['bvh_export_error'] = str(exc)
                self.update(status='视频与骨架已保存，BVH 未导出：'+str(exc))
        except Exception as exc:
            self.manifest.update(state='incomplete', error=str(exc)); self.update(status='保存失败：'+str(exc))
        finally:
            write_json(self.folder/'workflow.json', self.manifest); self.buffer.clear()

    def _preview_and_sync(self):
        with self.lock: frames = list(self.latest_frames)
        images = list(self.read().get('images', []))
        if len(images) != 2: images = [None, None]
        for c, frame in enumerate(frames):
            if frame is None or frame.index == self.seen[c]: continue
            # Reduced decode belongs to processing thread, never acquisition threads.
            image = cv2.imdecode(np.frombuffer(frame.payload, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
            if image is None: raise ValueError(f'cam{c} MJPEG 帧损坏，停止以保留可审计的帧索引。')
            self.seen[c] = frame.index; images[c] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.sync_history is not None and frame.index != self.sync_last[c]:
                h, w = image.shape[:2]; roi = image[h//4:3*h//4, w//4:3*w//4]
                brightness = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
                self.sync_history[c].append((frame.timestamp_ns, brightness)); self.sync_last[c] = frame.index
        self.update(images=images, frame_indices=[f.index if f else None for f in frames],
                    source_timestamps_ns=[f.timestamp_ns if f else None for f in frames],
                    capture_counts=list(self.source.counts), drops=list(self.buffer.dropped))
        if self.sync_history is not None and time.perf_counter() >= self.sync_until:
            try:
                self.sync = align_pulses(self.sync_history)
                self.update(status=f'校时完成：右路减去 {self.sync["right_offset_ns"]/1e6:.1f} ms；全身站稳初始化骨长',
                            synchronization=self.sync)
            except ValueError as exc: self.update(status='校时未通过：'+str(exc))
            self.sync_history = None; self.buffer.clear()

    def _process(self, frames):
        begin = time.perf_counter_ns()
        offset = self.sync['right_offset_ns']
        timestamp = (frames[0].timestamp_ns+frames[1].timestamp_ns-offset)//2
        if timestamp <= self.last_timestamp: return
        self.last_timestamp = timestamp
        images = []
        for frame in frames:
            image = cv2.imdecode(np.frombuffer(frame.payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None: raise ValueError('MJPEG 解码失败。')
            if [image.shape[1], image.shape[0]] != self.cal['image_size']: raise ValueError('图像尺寸与标定不一致。')
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        decoded = time.perf_counter_ns()
        views = demo_views(timestamp, self.cal) if self.demo else self.estimator.predict(images)
        inferred = time.perf_counter_ns()
        joints = self.filter.apply(triangulate(views[0]['joints'], views[1]['joints'], self.cal), timestamp)
        reconstructed = time.perf_counter_ns()
        if self.demo:
            for joint in joints:
                if joint['valid']: joint['reason'] = 'synthetic_fixture'
        pose = frame_packet(self.seq, timestamp, joints, session_id=self.session_id,
                            calibration_id=self.cal['calibration_id'], model_id=self.model_id,
                            coordinate_system='world_right_handed_X_forward_Y_left_Z_up',
                            clock_id=self.session_id+':visually_corrected_source')
        pose['source'] = 'windows_workflow'
        delta = (frames[1].timestamp_ns-offset-frames[0].timestamp_ns)/1e6
        lafan = dict(schema='live_lafan_pose/1.0', seq=self.seq, timestamp_ns=timestamp, valid=False,
                     joint_names=list(NAMES), unit='cm', axis='Y_up', local_quaternion_wxyz=None,
                     root_position_cm=None, reason='missing_core_joints')
        xyz = np.array([j['xyz'] if j['valid'] else [np.nan]*3 for j in joints])
        fk = None
        if np.isfinite(xyz[list(CORE)]).all():
            try:
                targets, basis = extend_coco(xyz, self.torso_length)
                if self.offsets is None:
                    self.bone_samples.append(targets)
                    if len(self.bone_samples) >= 30:
                        self.offsets = offsets_from_targets(np.array(self.bone_samples))
                        self.torso_length = float(np.percentile([np.linalg.norm(q[11]-q[0]) for q in self.bone_samples], 90))
                if self.offsets is not None:
                    rotations, fk = fit_rotations(targets, self.offsets, basis)
                    quat = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
                    if self.previous_quaternion is not None:
                        quat[np.sum(quat*self.previous_quaternion, axis=1) < 0] *= -1
                    self.previous_quaternion = quat.copy()
                    lafan.update(valid=True, reason='causal_geometric_fit',
                                 local_quaternion_wxyz=quat.tolist(), root_position_cm=(targets[0]*100).tolist(),
                                 fk_positions_cm=(fk*100).tolist(),
                                 target_fit_max_m=float(np.max(np.linalg.norm(fk-targets, axis=1))))
                else: lafan['reason'] = f'bone_initialization_{len(self.bone_samples)}/30'
            except ValueError as exc: lafan['reason'] = str(exc)
        self.seq += 1; self.poses += 1
        end = time.perf_counter_ns()
        self.last_pose_host = end
        self.pose_times.append(end)
        while self.pose_times and end-self.pose_times[0] > 2_000_000_000: self.pose_times.popleft()
        measured_fps = (len(self.pose_times)-1)*1e9/(end-self.pose_times[0]) if len(self.pose_times) > 1 else 0.
        timing = dict(pair_delta_ms=delta, right_offset_ns=offset,
                      source_frames=[f.index for f in frames], raw_source_ns=[f.timestamp_ns for f in frames],
                      saved_video_frames=[f.recording_index for f in frames],
                      processing_ms=(end-begin)/1e6, host_receive_to_pose_ms=(end-min(f.receive_ns for f in frames))/1e6)
        timing.update(decode_ms=(decoded-begin)/1e6, inference_ms=(inferred-decoded)/1e6,
                      reconstruction_ms=(reconstructed-inferred)/1e6, lafan_ms=(end-reconstructed)/1e6)
        pose['timing'] = timing
        for c, view in enumerate(views):
            view.update(camera_id=self.cal['camera_ids'][c], frame_index=frames[c].index,
                        saved_video_frame_index=frames[c].recording_index,
                        raw_timestamp_ns=frames[c].timestamp_ns,
                        timestamp_ns=frames[c].timestamp_ns-(offset if c else 0))
        if self.logs and all(f.recording_id == str(self.folder) and f.recording_index is not None for f in frames):
            self.logs.submit(dict(pose2d=dict(schema='pose2d/1.0', seq=pose['seq'], timestamp_ns=timestamp,
                                             joint_names=list(COCO17), views=views, timing=timing),
                                  skeleton3d=pose, live_lafan=lafan))
        # Draw only previews, never burn annotations into native recorded frames.
        thumbs = []
        for image, view in zip(images, views):
            thumb = cv2.resize(image, (480, round(480*image.shape[0]/image.shape[1])))
            scale = 480/image.shape[1]
            for joint in view['joints']:
                if joint['valid']: cv2.circle(thumb, tuple(np.int32(np.array(joint['xy'])*scale)), 3, (0, 255, 0), -1)
            thumbs.append(thumb)
        self.update(images=thumbs, lafan=lafan, skeleton=pose, timing=timing,
                    source_timestamps_ns=[f.timestamp_ns for f in frames], frame_indices=[f.index for f in frames],
                    capture_counts=list(self.source.counts), drops=list(self.buffer.dropped),
                    initialized=self.offsets is not None, valid_joints=sum(j['valid'] for j in joints),
                    pose_fps=measured_fps,
                    status=('正在记录' if self.recording else '实时骨架运行中') if lafan['valid'] else '骨架不可用：'+lafan['reason'])

    def _run(self):
        try:
            self.cal = demo_calibration() if self.demo else load_calibration(self.calibration_path,
                     dict(devices=[vars(d) for d in self.config.devices], requested_mode=dict(width=self.config.width, height=self.config.height)))
            if self.cal.get('world_from_cam0') is None: raise ValueError('需要带世界坐标的标定文件。')
            self.update(status='正在加载并预热人体模型…')
            if not self.demo:
                import torch
                torch.set_num_threads(4)
                cv2.setNumThreads(2)
            self.estimator = None if self.demo else PoseEstimator(self.model, self.device, image_size=self.inference_size,
                                                                 refine_person=self.refine_person)
            self.model_id = 'SYNTHETIC_NOT_MEASURED' if self.demo else self.estimator.model_id
            if self.estimator: self.estimator.predict([np.zeros((360, 640, 3), np.uint8)]*2)
            factory = self.source_factory or (DemoSource if self.demo else CameraSource)
            self.source = factory(self.config, self._receive, self.fail)
            if self.replay_alignment is not None: self.sync = dict(self.replay_alignment)
            self.source.start(); self.update(status='双路预览已启动。请点击入口闪光校时。')
            while not self.stop_event.is_set():
                try: command, value = self.commands.get_nowait()
                except queue.Empty: command = None
                try:
                    if command == 'sync':
                        if self.recording: raise ValueError('请先停止记录再重新校时。')
                        self.sync = None; self.sync_history = [[], []]; self.sync_until = time.perf_counter()+10
                        self.filter = CausalFilter(); self.offsets = None; self.bone_samples = []
                        self.torso_length = None; self.pose_times.clear(); self.last_pose_host = 0
                        self.previous_quaternion = None
                        self.last_timestamp = -1; self.poses = 0; self.buffer.clear()
                        self.update(lafan=None, skeleton=None, initialized=False, synchronization={}, timing={},
                                    status='10 秒校时中：中央白纸共同闪光至少三次，间隔长短不同；此时不解算骨架')
                    elif command == 'record': self._start_recording(value)
                    elif command == 'save': self._stop_recording()
                except (ValueError, FileExistsError) as exc: self.update(status=str(exc))
                if self.sync is None:
                    self._preview_and_sync(); time.sleep(.003); continue
                now = time.perf_counter()
                if now-self.last_start < 1/self.target_fps:
                    # Short Condition/Event waits can round to the Windows timer tick.
                    # Python 3.12 sleep keeps the FPS limiter from losing an extra tick.
                    time.sleep(min(.001, 1/self.target_fps-(now-self.last_start))); continue
                frames = self.buffer.latest(self.sync['right_offset_ns'], self.config.tolerance_ms)
                if frames is None:
                    self._preview_and_sync()
                    if not self.last_pose_host or time.perf_counter_ns()-self.last_pose_host > 250_000_000:
                        self.update(lafan=None, pose_fps=0., status='等待时间匹配的双路帧；不会拿不同步帧解算')
                    time.sleep(.001); continue
                if not self.poses: self.pose_started = now
                self.last_start = now; self._process(frames)
        except Exception as exc: self.fail(str(exc))
        finally:
            if self.source: self.source.stop()
            try: self._stop_recording()
            except Exception as exc: self.update(status='保存失败：'+str(exc))
            self.update(running=False, lafan=None)
            if not self.error: self.update(status='已停止')
