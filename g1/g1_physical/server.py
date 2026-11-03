"""Streaming BVH -> GMR -> released policy -> free-running MuJoCo physics."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import uuid
import threading
import mujoco
import numpy as np
from aiohttp import web, WSMsgType
from physical import PhysicsTracker, ReferenceBuffer
from reference_trajectory import SmoothReferenceBuffer
from position_control import PositionFeedback
from height_control import HeightFeedback
from stream_client import StreamClient
from windows_timing import enable_timer

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'g1_realtime' if (ROOT/'g1_realtime').exists() else ROOT.parent/'g1_realtime'
sys.path.insert(0,str(BASE))
from tracker import Tracker, model_description


class Stream:
    def __init__(self, message):
        if message.get('engine','protomotions')!='protomotions':
            raise ValueError('对照策略未通过稳定性验收，请刷新页面使用原版 ProtoMotions')
        self.ik = Tracker(message)
        self.render_data = mujoco.MjData(self.ik.model)
        self.physics = PhysicsTracker()
        self.position_feedback = PositionFeedback() if self.ik.metadata.get('recovered_skeleton') else None
        self.height_feedback = (HeightFeedback(self.physics, float(message['source_leg_length_m']))
                                if self.position_feedback and 'source_leg_length_m' in message else None)
        self.heights = []
        self.conservative=message.get('reference_mode','original')=='conservative'
        self.buffer = (SmoothReferenceBuffer(self.physics.model) if self.ik.metadata.get('recovered_skeleton')
                       else ReferenceBuffer(self.physics.model))
        self.order = [self.ik.metadata['joint_names'].index(n) for n in self.physics.joints]
        self.reverse = np.argsort(self.order)
        self.humans = []
        self.source_origin = self.wall_origin = self.client_origin = None
        self.t = 0.
        self.started = self.ended = self.finished = False
        self.received = self.processed = 0
        self.last_seq = -1
        self.lag = .38
        self.session = uuid.uuid4().hex
        self.frame_lock = threading.RLock()
        self.metadata = dict(self.ik.metadata,session=self.session,engine='ProtoMotions + MuJoCo physics',lag_ms=380)
        self.metadata['reference_mode']='conservative' if self.conservative else 'original'
        self.metadata['request_id']=message.get('request_id')

    def close(self):
        pass

    def receive(self,msg):
        # The official adapter can solve BVH in a separate executor. Keep the
        # expensive solve outside the shared reference-buffer critical section.
        with self.frame_lock:self.validate_frame(msg)
        result=self.ik.solve(msg['values'])
        with self.frame_lock:return self.accept_frame(msg,result)

    def validate_frame(self,msg):
        stamp=float(msg['source_time'])
        if not np.isfinite(stamp) or not isinstance(msg['seq'],int) or msg['seq']<=self.last_seq:
            raise ValueError('时间戳须有限，帧号须递增')
        if self.source_origin is not None and stamp-self.source_origin<=self.buffer.times[-1]:
            raise ValueError('输入时间必须严格递增；跳转或重播请重新初始化')
        if self.height_feedback and not np.isfinite(float(msg['source_hip_height_m'])):
            raise ValueError('髋部相对脚踝高度必须是有限数值')

    def accept_frame(self,msg,result):
        stamp=float(msg['source_time'])
        if self.source_origin is None:
            self.source_origin=stamp
            self.wall_origin=time.perf_counter()
            self.client_origin=msg.get('client_time')
        q=np.array(result['qpos']);q[7:]=q[7:][self.order]
        if self.conservative:
            from reference_conditioning import limit_pelvis_tilt
            q=limit_pelvis_tilt(q)
        self.buffer.push(stamp-self.source_origin,q)
        self.humans.append(np.array(result['human_pos']))
        self.humans=self.humans[-32:]
        if self.height_feedback:
            self.heights.append(float(msg['source_hip_height_m']))
            self.heights=self.heights[-32:]
        self.last_seq=msg['seq'];self.received+=1
        return dict(type='ack',seq=msg['seq'],ik_ms=result['solve_ms'])

    def tick(self):
        with self.frame_lock:return self.tick_locked()

    def tick_locked(self):
        if self.wall_origin is None or self.finished:
            return None
        start=time.perf_counter()
        if not self.started:
            if self.buffer.times[-1]<max(self.physics.offsets) and not self.ended:return None
            self.physics.reset(self.buffer.sample(0)[0]);self.started=True
        # Initialize within the reference buffer instead of adding reset/DDS
        # handshake time after it. The first physics step still waits for lag.
        if time.perf_counter()<self.wall_origin+self.lag:return None
        last=self.buffer.times[-1]
        # Follow the timestamped source clock. A missing reference or a delayed
        # control tick must not permanently shift every later reference frame.
        source_now=max(0.,time.perf_counter()-self.wall_origin-self.lag)
        waiting=bool(source_now+max(self.physics.offsets)>last+1e-8 and not self.ended)
        target_t=min(source_now,last)
        stamps=np.minimum(target_t+self.physics.offsets,last)
        refs=self.buffer.references(stamps)
        reference=self.buffer.sample(target_t)[0]
        if self.position_feedback:
            refs=self.position_feedback.apply(refs,reference,self.physics.data.qpos,self.physics.data.qvel,self.physics.dt)
        height_target=None
        if self.height_feedback:
            height=float(np.interp(target_t,self.buffer.times,self.heights))
            refs,height_target=self.height_feedback.apply(refs,height,self.physics.data.qpos,self.physics.data.qvel,self.physics.dt)
        self.physics.step(refs)
        self.t=target_t+self.physics.dt
        self.processed+=1
        if self.ended and self.t>=last:self.finished=True
        q=self.physics.data.qpos.copy()
        fallen=self.physics.fallen()
        if fallen:self.finished=True
        tilt=float(np.degrees(np.arccos(np.clip(self.physics.data.xmat[1,8],-1,1))))
        # Render the original G1 meshes using the actual simulated state, never the reference.
        render_q=q.copy();render_q[7:]=q[7:][self.reverse]
        self.render_data.qpos[:]=render_q
        mujoco.mj_forward(self.ik.model,self.render_data)
        i=min(max(1,int(np.searchsorted(self.buffer.times,target_t))),len(self.humans)-1)
        lo=max(0,i-1);span=self.buffer.times[i]-self.buffer.times[lo]
        w=np.clip((target_t-self.buffer.times[lo])/span,0,1) if span else 0
        human=self.humans[lo]*(1-w)+self.humans[i]*w
        elapsed=(time.perf_counter()-start)*1000
        return dict(type='pose',session=self.session,seq=self.processed,source_time=target_t+self.source_origin,
            client_time=self.client_origin+target_t*1000 if self.client_origin is not None else None,
            qpos=render_q.tolist(),root_pos=q[:3].tolist(),root_quat=q[3:7].tolist(),joint_pos=render_q[7:].tolist(),
            body_pos=self.render_data.xpos.tolist(),body_quat=self.render_data.xquat.tolist(),human_pos=human.tolist(),
            geom_pos=self.render_data.geom_xpos.tolist(),geom_mat=self.render_data.geom_xmat.tolist(),
            solve_ms=elapsed,process_ms=(time.perf_counter()-self.wall_origin-target_t)*1000,
            received=self.received,processed=self.processed,dropped=0,waiting=waiting,fallen=fallen,finished=self.finished,
            physics=True,simulation_time=self.physics.data.time,initial_floor_offset=self.physics.initial_floor_offset,
            pelvis_tilt_deg=tilt,reference_root_pos=reference[:3].tolist(),
            position_feedback=bool(self.position_feedback),
            height_target_m=height_target,
            knee_reference_correction_rad=self.height_feedback.filtered if self.height_feedback else None,
            horizontal_error_m=float(np.linalg.norm(reference[:2]+self.position_feedback.origin-q[:2])) if self.position_feedback else None,
            reference_tilt_correction_deg=np.rad2deg(self.position_feedback.filtered).tolist() if self.position_feedback else None)


def create_app(frame_executor=None):
    stop_timer=enable_timer()
    app=web.Application(client_max_size=512*1024)
    clients={};executor=ThreadPoolExecutor(max_workers=1)
    state=dict(stream=None,owner=None)
    async def execute(fn,*args):
        return await asyncio.get_running_loop().run_in_executor(executor,fn,*args)
    async def receive_frame(stream,msg):
        return await asyncio.get_running_loop().run_in_executor(frame_executor or executor,stream.receive,msg)
    async def broadcast(p):
        payload=json.dumps(p,separators=(',',':'))
        for client in list(clients.values()):client.offer(p,payload)
    async def reset_input():
        stream=state['stream']
        state.update(owner=None,stream=None)
        if stream:await execute(stream.close)
        await broadcast(dict(type='reset'))
    app['reset_input']=reset_input
    async def worker():
        deadline=time.perf_counter()
        while True:
            deadline+=.02
            stream=state['stream']
            if stream:
                try:
                    p=await execute(stream.tick)
                    if p and stream is state['stream']:await broadcast(p)
                except Exception as e:
                    stream.finished=True
                    await execute(stream.close)
                    await broadcast(dict(type='error',message=str(e)))
            await asyncio.sleep(max(0,deadline-time.perf_counter()))
            if time.perf_counter()-deadline>.1:deadline=time.perf_counter()
    async def socket(request):
        if request.headers.get('Origin') not in (None,f'http://{request.host}',f'https://{request.host}'):
            raise web.HTTPForbidden()
        ws=web.WebSocketResponse(heartbeat=20,max_msg_size=512*1024)
        await ws.prepare(request)
        client=StreamClient(ws);clients[ws]=client
        def reply(p):client.offer(p,json.dumps(p,separators=(',',':')))
        reply(dict(type='hello',busy=state['owner'] is not None))
        if state['stream']:reply(state['stream'].metadata)
        try:
            async for packet in ws:
                if packet.type!=WSMsgType.TEXT:continue
                kind=None
                try:
                    msg=json.loads(packet.data);kind=msg.get('type')
                    if kind=='init':
                        if state['owner'] not in (None,ws):raise ValueError('已有输入源')
                        if state['stream']:await execute(state['stream'].close)
                        state['owner']=ws;state['stream']=None
                        state['stream']=await execute(Stream,msg)
                        await broadcast(state['stream'].metadata)
                    elif kind in ('frame','end','release','tracking_state'):
                        if kind=='release' and state['owner'] is not ws:
                            reply(dict(type='idle'))
                            continue
                        if state['owner'] is not ws or not state['stream']:raise ValueError('请先初始化')
                        if kind=='frame':reply(await receive_frame(state['stream'],msg))
                        elif kind=='end':state['stream'].ended=True
                        elif kind=='tracking_state':
                            handler=getattr(state['stream'],'set_tracking_state',None)
                            if handler is None:raise ValueError('This input does not support quality recovery')
                            reply(await execute(handler,msg))
                        else:
                            await execute(state['stream'].close)
                            state.update(owner=None,stream=None)
                            await broadcast(dict(type='idle'))
                    else:raise ValueError('无效消息类型')
                except (ValueError,KeyError,TypeError) as e:
                    if kind=='init' and state['owner'] is ws and state['stream'] is None:
                        state.update(owner=None,stream=None)
                    reply(dict(type='error',message=str(e)))
        finally:
            clients.pop(ws,None)
            await client.close()
            if state['owner'] is ws:
                if state['stream']:await execute(state['stream'].close)
                state.update(owner=None,stream=None)
        return ws
    async def index(request):return web.FileResponse(ROOT/'static/index.html')
    async def static(request):
        name=request.match_info['name']
        path=ROOT/'static'/name if name in ('app.js','index.html','smoothing.js','timed-display.js','frame-socket.js','frame-sender.js') else BASE/'static'/name
        if not path.resolve().is_relative_to((ROOT/'static').resolve()) and not path.resolve().is_relative_to((BASE/'static').resolve()):raise web.HTTPForbidden()
        return web.FileResponse(path)
    async def model(request):
        response=web.Response(body=app['model_json'],content_type='application/json',zlib_executor_size=1024)
        response.enable_compression();return response
    async def samples(request):return web.json_response(json.loads((BASE/'samples/catalog.json').read_text(encoding='utf-8')))
    async def health(request):return web.json_response(dict(status='ok',portable_root=str(ROOT.parent),engine='ProtoMotions + MuJoCo physics',version='20261014',control_hz=50,lag_ms=380))
    async def start(app):
        app['model']=await execute(model_description)
        app['model_json']=await execute(lambda:json.dumps(app['model']).encode('utf-8'))
        app['worker']=asyncio.create_task(worker())
    async def stop(app):
        app['worker'].cancel()
        await asyncio.gather(app['worker'],return_exceptions=True)
        for ws in list(clients):await ws.close()
        executor.shutdown()
        stop_timer()
    app.on_startup.append(start);app.on_cleanup.append(stop)
    app.router.add_get('/',index);app.router.add_get('/api/model',model);app.router.add_get('/api/samples',samples)
    app.router.add_get('/api/health',health);app.router.add_get('/ws',socket)
    app.router.add_get('/static/{name:.*}',static);app.router.add_static('/samples/',BASE/'samples')
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770)
    web.run_app(create_app(),host='127.0.0.1',port=p.parse_args().port)
