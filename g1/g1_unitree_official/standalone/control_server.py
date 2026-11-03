"""Input-only web service; the official simulator is an independent process."""
import asyncio
import json
import os
import time
import subprocess
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import mujoco
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import dds_controller as dds
import server
from bvh_worker import RemoteTracker
from aiohttp import web

async def main():
    from build_policy_scene import build
    build()
    manager = dds.ControllerManager(address=r'\\.\pipe\g1-controller-'+uuid.uuid4().hex, ctx=dds.multiprocessing.get_context('spawn'))
    manager.start()
    dds.CONTROLLER = manager.DDSController()
    dds.FK_META = dds.CONTROLLER.model_spec()
    c, t = dds.FK_META['control'], dds.FK_META['timing']
    from physical import load_mujoco_model
    dds.FK_MODEL, _ = load_mujoco_model(dds.FK_META['robot']['mjcf_path'],c['stiffness'],c['damping'],t['physics_dt'])
    server.PhysicsTracker = dds.ExternalTracker
    server.Tracker = RemoteTracker
    generation = 0
    class ExternalStream(server.Stream):
        def __init__(self, message):
            dds.CONTROLLER.read_state()
            super().__init__(message)
            # This local model only reconstructs DDS geometry for the webpage.
            self.ik.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            self.generation = generation
            self.metadata['engine'] = '官方 unitree_mujoco · ProtoMotions 兼容物理配置 · DDS 控制'
        def tick(self):
            if self.generation != generation:
                self.finished = True
                return None
            result = super().tick()
            if result and result['finished']:
                self.close()
            return result
        def close(self):
            dds.CONTROLLER.stop_reference()
    server.Stream = ExternalStream
    frame_executor = ThreadPoolExecutor(max_workers=1)
    app = server.create_app(frame_executor=frame_executor)
    simulator_process = None
    async def launch_simulator(request):
        nonlocal simulator_process
        if request.headers.get('Origin') != 'http://'+request.host:
            raise web.HTTPForbidden()
        try:
            dds.CONTROLLER.read_state()
            return web.json_response(dict(message='官方仿真已运行，请切换到 MuJoCo 窗口'))
        except ValueError:
            pass
        if simulator_process is not None and simulator_process.poll() is None:
            return web.json_response(dict(message='官方程序正在启动，请等待 DDS 连接'))
        root = Path(__file__).resolve().parent
        entry = root/'simulate_python/unitree_mujoco.py'
        original = root.parent/'unitree_mujoco-main/simulate_python/unitree_mujoco.py'
        if hashlib.sha256(entry.read_bytes()).digest()!=hashlib.sha256(original.read_bytes()).digest():
            raise web.HTTPConflict(text='官方入口校验失败')
        dds.CONTROLLER.reset_controller()
        with (root/'simulator.log').open('a') as out, (root/'simulator_error.log').open('a') as err:
            simulator_process = subprocess.Popen([str(root.parents[1]/'runtime/pythonw.exe'), '-X','utf8','-u',str(entry)], cwd=entry.parent, stdout=out, stderr=err)
        return web.json_response(dict(message='已启动官方 MuJoCo 窗口，等待 DDS 连接'))
    app.router.add_post('/api/official/start', launch_simulator)
    async def restart_simulator(request):
        nonlocal simulator_process, generation
        if request.headers.get('Origin') != 'http://'+request.host:
            raise web.HTTPForbidden()
        entry = str(Path(__file__).resolve().parent/'simulate_python/unitree_mujoco.py')
        command = "$entry='"+entry.replace("'","''")+"'; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and $_.CommandLine -and $_.CommandLine.Contains($entry) } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }"
        generation += 1
        await app['reset_input']()
        process = await asyncio.create_subprocess_exec('powershell.exe','-NoProfile','-Command',command, creationflags=subprocess.CREATE_NO_WINDOW)
        await process.wait()
        simulator_process = None
        dds.CONTROLLER.reset_controller()
        return await launch_simulator(request)
    app.router.add_post('/api/official/restart', restart_simulator)
    @web.middleware
    async def label(request, handler):
        if request.path == '/api/health':
            c = dds.CONTROLLER
            try:
                q, v = c.read_state()
                height = float(q[2])
            except ValueError: height = None
            return web.json_response(dict(status='ok', engine='Independent original Unitree entry + external DDS controller',
                controller_pid=os.getpid(), compatibility='released policy MJCF + implicit PD + synchronized 20 substeps', **c.metrics(),
                root_height=height, simulator_connected=height is not None, domain=int(os.environ.get('G1_DDS_DOMAIN','74')), simulation_in_this_process=False, portable_root=str(Path(__file__).resolve().parents[2])))
        if request.path == '/':
            html = (server.ROOT/'static/index.html').read_text(encoding='utf-8')
            html = html.replace('<head>', '<head><style>#official-entry{display:none!important}</style>')
            html = html.replace('</body>', '''<script>
document.addEventListener('DOMContentLoaded',()=>{
 const box=document.createElement('div');box.style='padding:16px;background:#103a43;color:white;position:sticky;top:0;z-index:1000';
 box.innerHTML='<button id="launch-official">打开宇树仿真原生窗口</button> <button id="restart-official">重新站立 / 重启仿真</button> <strong id="official-status">检查官方连接…</strong>';
 document.body.prepend(box);
 const status=document.getElementById('official-status'), button=document.getElementById('launch-official');
 button.onclick=async()=>{button.disabled=true;try{const r=await fetch('/api/official/start',{method:'POST'});if(!r.ok)throw Error(await r.text());status.textContent=(await r.json()).message;}catch(e){status.textContent=e.message;}finally{button.disabled=false;}};
 document.getElementById('restart-official').onclick=async()=>{window.dispatchEvent(new Event('official-reset'));try{const r=await fetch('/api/official/restart',{method:'POST'});if(!r.ok)throw Error(await r.text());status.textContent=(await r.json()).message;}catch(e){status.textContent=e.message;}};
 window.prepareOfficialClip=async()=>{
  window.officialPreparing=true;
  try{const r=await fetch('/api/official/start',{method:'POST'});if(!r.ok)throw Error(await r.text());
   for(let i=0;i<30;i++){await new Promise(resolve=>setTimeout(resolve,500));const h=await(await fetch('/api/health')).json();if(h.simulator_connected)return;}
   throw Error('官方仿真未恢复站立，请检查窗口或重新启动服务');
  }finally{window.officialPreparing=false;}
 };
 let blocked=false;
 async function check(){try{const h=await(await fetch('/api/health')).json();const down=h.simulator_connected&&h.fallen;status.textContent=down?'官方 G1 已倒地；本轮失败保留，仿真暂停':h.simulator_connected?(h.simulation_paused?'● 官方仿真已连接 · 仿真暂停，等待输入':'● 官方仿真正在物理跟踪 · 在 MuJoCo 窗口查看'):'○ 官方仿真未连接，请先点击启动窗口';const play=document.getElementById('play');if(!h.simulator_connected||window.officialPreparing){play.disabled=true;blocked=true;}else if(blocked){play.disabled=document.getElementById('clip-name').textContent==='尚未选择动作';blocked=false;}}catch(e){status.textContent='控制服务断开，重启后将自动重连';}}
 check();setInterval(check,1500);
});
</script></body>''')
            html = html.replace('<body>', '<body><div style="padding:16px;background:#103a43;color:white">自定义 BVH 接入面板（非宇树官方网页） → 宇树开源 unitree_mujoco 的 MuJoCo 原生窗口。动作跟踪采用第三方 ProtoMotions 策略及本项目的兼容适配。使用策略对应的 G1 模型和隐式 PD，按控制周期同步推进物理。每次开始片段仅初始化一次；倒地即记录失败，跟踪中不重置。暂停/结束输入时仿真也暂停，与原网页一致；不代表停下后自主平衡。 <a href="http://127.0.0.1:8770/" style="color:white">返回原网页模式</a></div>')
            html = html.replace('AMD 实例物理计算 · 加密 SSH 链路','本机独立官方程序 · SDK2 DDS')
            html=html.replace('http://127.0.0.1:8770/', 'http://127.0.0.1:'+os.environ.get('G1_FALLBACK_PORT','8770')+'/')
            return web.Response(text=html,content_type='text/html')
        return await handler(request)
    app.middlewares.append(label)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner,'127.0.0.1',int(os.environ.get('G1_WEB_PORT','8771'))).start()
    print('External controller ready; start original unitree_mujoco.py separately.',flush=True)
    try: await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        frame_executor.shutdown()
        dds.CONTROLLER.close()
        manager.shutdown()

if __name__ == '__main__': asyncio.run(main())
