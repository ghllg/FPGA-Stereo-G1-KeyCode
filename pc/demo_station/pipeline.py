from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .core import ROOT,JsonlTail,read_json,write_json,digest,read_lines
from .candidate import local_preflight,validate_live_identity


def vision_imports():
    path=str(ROOT/'windows_vision')
    if path not in sys.path:sys.path.insert(0,path)


def reconstruct_recording(path,progress,cancel):
    """Process measured keypoints without inventing a live transport packet."""
    path=Path(path);manifest=read_json(path/'session.json');assets=manifest.get('assets',{})
    for key in ('calibration','checkpoint','binding'):
        item=assets.get(key)
        if not item or digest(path/item['path'])!=item['sha256']:raise ValueError('缺少或损坏的会话关联文件：'+key)
    vision_imports()
    from pose3d_train.online import SkeletonEngine
    calibration=path/assets['calibration']['path'];checkpoint=path/assets['checkpoint']['path'];binding=read_json(path/assets['binding']['path'])
    engine=SkeletonEngine(calibration,checkpoint)
    if binding.get('calibration_hash')!=engine.stereo.hash or binding.get('camera_physical_ids')!=engine.stereo.calibration.get('camera_physical_ids'):
        raise ValueError('录制绑定与标定相机身份不一致')
    comparison=read_json(path/'comparison.json');model=comparison['input_identity']['models']['model.opt.bin'].split()[0]
    if binding.get('model_sha256')!=model:raise ValueError('录制绑定的二维模型与板端测量模型不同')
    saved_roi=manifest.get('roi_xywh');bound_roi=binding.get('roi_xywh')
    if saved_roi and bound_roi and any(saved_roi[s]!=(bound_roi[s] if isinstance(bound_roi,dict) else bound_roi) for s in ('left','right')):
        raise ValueError('录制 ROI 与标定绑定不符')
    output=path/'online_3d';output.mkdir(exist_ok=False)
    rows=read_lines(path/'pl-stream.ndjson');count=0
    with (output/'skeleton3d.jsonl').open('x',encoding='utf-8') as f:
        for i,row in enumerate(rows):
            if cancel.is_set():raise RuntimeError('已停止重建；保留已生成结果')
            views=[]
            for side,camera in zip(('left','right'),row['cameras']):
                if camera['camera_id']!=side:raise ValueError('相机顺序错误')
                joints=[dict(xy=p[:2] if p[3] else None,confidence=min(p[2],1.),valid=p[3]) for p in camera['keypoints']]
                views.append(dict(timestamp_ns=camera['capture_timestamp_us']*1000,joints=joints,frame_index=i))
            decoded=dict(timestamp_ns=views[0]['timestamp_ns'],views=views,seq=i,model_id='PP-TinyPose:'+model[:16])
            packet=engine.process(decoded,'recorded_v4l2_monotonic',path.name)
            packet.update(source='board_recorded_jpeg_comparison',replay_processing=True)
            f.write(json.dumps(packet,allow_nan=False)+'\n');count+=1
            if i%25==0:progress(f'三维重建 {i+1}/{len(rows)}')
    write_json(output/'summary.json',dict(frames=count,source='measured pl-stream keypoints; original capture timestamps',live_latency_measured=False))
    return output/'skeleton3d.jsonl'


class G1Runner:
    """Prepare locally, explicitly start input, poll independent tracks, then drain.

    start() never contacts the board. start_input() must follow the UI countdown
    and current BoardClient proof. Socket disconnect is not a completion signal.
    """
    def __init__(self,config,path,progress):
        self.config=dict(config);self.path=Path(path);self.progress=progress
        self.process=self.log=None;self.seq=0;self.input_started=False;self.owned={};self.next_owned_scan=0.
        self.tails={key:JsonlTail(self.path/name,persistent=True) for key,name in (
            ('paired','pc_receiver/received.ndjson'),('skeleton','online_3d/skeleton3d.jsonl'),('robot','g1/poses.ndjson'))}
        if self.config.get('hybrid_refinement'):self.tails['refined']=JsonlTail(self.path/'online_3d/pose2d.jsonl',persistent=True)
        self.latest=dict.fromkeys(self.tails);self.freshness=dict.fromkeys(self.tails)
    def _command(self,action,**fields):
        self.seq+=1;write_json(self.path/'candidate_control.json',dict(seq=self.seq,action=action,**fields))
    def _collect_owned(self,force=False):
        if not self.process or self.process.poll() is not None:return
        now=time.monotonic()
        if not force and now<self.next_owned_scan:return
        self.next_owned_scan=now+1.
        import psutil
        try:
            parent=psutil.Process(self.process.pid)
            for child in [parent,*parent.children(recursive=True)]:self.owned[(child.pid,child.create_time())]=child
        except psutil.Error:pass
    def start(self,cancel):
        if self.process:raise RuntimeError('候选已启动；请创建新会话')
        proof=local_preflight(self.config);self.path.mkdir(parents=True,exist_ok=True)
        for name in ('candidate_status.json','candidate_control.json','candidate_runtime_sources','g1'):
            if (self.path/name).exists():raise ValueError('已有候选运行记录，不覆盖：'+name)
        write_json(self.path/'candidate_launch_config.json',self.config)
        self.log=(self.path/'candidate_worker.log').open('xb')
        env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.process=subprocess.Popen([proof['python'],'-B','-X','utf8','-m','demo_station.candidate_worker',
            '--config',str(self.path/'candidate_launch_config.json'),'--output',str(self.path)],
            cwd=ROOT,env=env,stdout=self.log,stderr=self.log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.progress('准备推荐候选的本机模型与仿真，尚未连接板端…')
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if cancel.is_set():raise RuntimeError('已取消本机预启动')
            snapshot=self.poll()
            if snapshot.get('error'):raise RuntimeError(snapshot['error'])
            if snapshot['state']=='ready':return
            cancel.wait(.1)
        raise TimeoutError('本机候选准备超时')
    def start_input(self,binding=None,preflight=None):
        if self.poll()['state']!='ready' or self.input_started:raise RuntimeError('候选尚未准备或输入已开始')
        if not self.config.get('recording_source'):
            template=read_json(self.path/'assets/binding.json')
            validate_live_identity(binding,preflight,template,self.config)
        self._command('start',binding=binding,preflight=preflight);self.input_started=True
    @staticmethod
    def _read_status(path):
        # Windows can briefly deny an open while the producer atomically replaces this JSON.
        for attempt in range(3):
            try:return read_json(path)
            except PermissionError:
                if attempt==2:raise
                time.sleep(.005)
    def poll(self):
        self._collect_owned()
        for key,tail in self.tails.items():
            rows=tail.poll()
            if rows:self.latest[key]=rows[-1];self.freshness[key]=time.monotonic()
        status_path=self.path/'candidate_status.json'
        status=self._read_status(status_path) if status_path.exists() else dict(state='preparing' if self.process else 'idle')
        tracking_path=self.path/'tracking_status.json'
        tracking=self._read_status(tracking_path) if status.get('state')!='preparing' and tracking_path.exists() else {}
        if self.process and self.process.poll() is not None:
            if self.process.returncode or status.get('state') not in ('finished','stopped'):
                status.update(state='failed',error=status.get('error') or '候选进程异常退出；见 candidate_worker.log')
        return {**status,**self.latest,'freshness':dict(self.freshness),'tracking':tracking}
    def finish(self,cancel=None):
        if not self.process:return
        if self.process.poll() is None:
            self._collect_owned(force=True);self._command('stop')
        deadline=time.monotonic()+30
        while self.process.poll() is None and time.monotonic()<deadline:
            self.poll();time.sleep(.05)
        if self.process.poll() is None:raise TimeoutError('候选收尾超过 30 秒，保留记录并仅清理本次进程')
        result=self.poll()
        if result.get('error') or result['state']=='failed':raise RuntimeError(result.get('error') or '候选收尾失败')
        if self.input_started and (result['state']!='finished' or result.get('reference_drained') is not True):
            raise RuntimeError('输入已开始但最后参考未完整排空')
    def close(self):
        try:
            if self.process and self.process.poll() is None:self.finish()
        finally:
            try:
                self._collect_owned(force=True)
                import psutil
                active=[]
                for (pid,created),process in reversed(list(self.owned.items())):
                    try:
                        if process.is_running() and process.create_time()==created:process.terminate();active.append(process)
                    except psutil.Error:pass
                _,remaining=psutil.wait_procs(active,timeout=3)
                for process in remaining:
                    try:process.kill()
                    except psutil.Error:pass
                psutil.wait_procs(remaining,timeout=3)
            finally:
                for tail in self.tails.values():tail.close()
                if self.log:self.log.close();self.log=None


def simulate_recording(config,path,progress,cancel):
    path=Path(path);source=path/'online_3d/skeleton3d.jsonl'
    if not source.exists():source=reconstruct_recording(path,progress,cancel)
    manifest=read_json(path/'session.json');assets=manifest.get('assets',{})
    for key in ('calibration','checkpoint','binding'):
        item=assets.get(key)
        if not item or digest(path/item['path'])!=item['sha256']:raise ValueError('缺少或损坏的会话关联文件：'+key)
    replay_config={**config,'recording_source':str(source.resolve())}
    for key,item in assets.items():
        if key in ('calibration','checkpoint','binding'):replay_config[key]=str((path/item['path']).resolve())
    output=path/'candidate_replay'
    if output.exists():raise ValueError('已有候选生成记录，不覆盖')
    runner=G1Runner(replay_config,output,progress)
    try:
        runner.start(cancel);runner.start_input()
        while True:
            snapshot=runner.poll()
            if snapshot.get('error'):raise RuntimeError(snapshot['error'])
            if snapshot['state'] in ('finished','stopped','failed'):break
            if cancel.wait(.05):break
        runner.finish()
        if cancel.is_set():raise RuntimeError('已停止生成，保留部分结果但不标为完整回放')
        manifest.setdefault('streams',{}).update(robot='candidate_replay/g1/poses.ndjson',robot_report='candidate_replay/g1/report.json')
        manifest['simulation_scope']='Saved reconstructed 3D at source time; new PC receipts; no new board measurement'
        write_json(path/'session.json',manifest)
    finally:runner.close()
    return str(path)
