"""Portable Windows CPU launcher. Only manages processes using this runtime."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT/'runtime/python.exe'
PYTHONW = ROOT/'runtime/pythonw.exe'
STATE = ROOT/'logs/processes.json'


def settings():
    value = json.loads((ROOT/'ports.json').read_text(encoding='utf8'))
    for key in ('web_port','official_web_port','board_port'):
        if type(value[key]) is not int or not 1024 <= value[key] <= 65535:
            raise ValueError('Invalid port: '+key)
    if value['web_port']==value['official_web_port']:
        raise ValueError('Web-only and native web ports must differ')
    if type(value['dds_domain']) is not int or not 0 <= value['dds_domain'] <= 100:
        raise ValueError('Invalid isolated simulation DDS domain')
    return value


def environment():
    cfg = settings()
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        env[name] = '1'
    env.update(PYTHONUTF8='1', G1_WEB_PORT=str(cfg['official_web_port']),
               G1_FALLBACK_PORT=str(cfg['web_port']), G1_DDS_DOMAIN=str(cfg['dds_domain']))
    return env


def health(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as r:
            return json.load(r)
    except OSError:
        return None


def spawn(script, gui=False, arguments=()):
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
    with (logs/(script.stem+'.log')).open('ab') as out, (logs/(script.stem+'_error.log')).open('ab') as err:
        process = subprocess.Popen([str(PYTHONW if gui else PYTHON),'-X','utf8','-u',str(script),*arguments],
            cwd=script.parent,env=environment(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    records = json.loads(STATE.read_text()) if STATE.exists() else []
    records.append(process.pid)
    STATE.write_text(json.dumps(records))
    return process


def start(native=True, browser=True):
    cfg = settings()
    port = cfg['official_web_port'] if native else cfg['web_port']
    status = health(port)
    if status and status.get('portable_root') != str(ROOT):
        raise RuntimeError(f'Port {port} belongs to another project. Edit ports.json or close that project yourself.')
    if not status:
        script = ROOT/'g1_unitree_official/standalone/control_server.py' if native else ROOT/'g1_physical/server.py'
        process = spawn(script,arguments=() if native else ('--port',str(port)))
        deadline = time.monotonic()+60
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError('Service exited. See logs/'+script.stem+'_error.log')
            status = health(port)
            if status: break
            time.sleep(.2)
        else: raise RuntimeError('Service startup timed out; see logs')
    if native and not status.get('simulator_connected'):
        process = spawn(ROOT/'g1_unitree_official/standalone/simulate_python/unitree_mujoco.py',gui=True)
        deadline = time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError('Native viewer exited. See logs/unitree_mujoco_error.log')
            status = health(port)
            if status and status.get('simulator_connected'): break
            time.sleep(.3)
        else: raise RuntimeError('No native DDS connection; see logs')
    url = f'http://127.0.0.1:{port}/'
    if browser: webbrowser.open(url)
    print(url,flush=True)
    return url


def stop():
    """Check executable ownership, stop controller processes, then remaining children."""
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    # Query process inventory through PowerShell, without changing any UI or filesystem.
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',
        '[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress'],
        capture_output=True,text=True,encoding='utf-8-sig',creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    inventory = json.loads(result.stdout)
    own = {str(PYTHON).lower(),str(PYTHONW).lower()}
    for record in inventory:
        if str(record.get('ExecutablePath','')).lower() not in own: continue
        pid = record['ProcessId']
        if pid == os.getpid(): continue
        handle = kernel.OpenProcess(0x1000|1,False,pid)
        if not handle: continue
        try:
            size=wintypes.DWORD(32768); path=ctypes.create_unicode_buffer(size.value)
            if kernel.QueryFullProcessImageNameW(handle,0,path,ctypes.byref(size)) and path.value.lower() in own:
                kernel.TerminateProcess(handle,0)
        finally: kernel.CloseHandle(handle)
    if STATE.exists(): STATE.write_text('[]')
    print('Stopped only processes using this package runtime.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['start','web','stop','check','bridge','receive'])
    parser.add_argument('--no-browser',action='store_true')
    args,extra=parser.parse_known_args()
    if args.mode!='stop' and not str(ROOT).isascii():
        raise RuntimeError('MuJoCo needs an ASCII folder path. Extract the full ZIP to D:\\G1 or C:\\G1, without Chinese characters.')
    cfg=settings()
    if args.mode in ('start','web'): start(args.mode=='start',not args.no_browser)
    elif args.mode=='stop': stop()
    elif args.mode=='check':
        subprocess.run([str(PYTHON),'-X','utf8',str(ROOT/'check_package.py')],env=environment(),check=True)
    elif args.mode=='bridge':
        if any(arg == '--calibration' or arg.startswith('--calibration=') for arg in extra):
            raise ValueError('ABI9 input must pass pose3d_train.abi9_bridge first; use its skeleton3d.jsonl here.')
        start(native=True,browser=False)
        subprocess.run([str(PYTHON),'-X','utf8',str(ROOT/'g1_stereo_bridge/stream.py'),
            '--url',f"http://127.0.0.1:{cfg['official_web_port']}/ws",
            '--fps','0','--warmup-seconds','1.4',*extra],env=environment(),check=True)
    elif args.mode=='receive':
        subprocess.run([str(PYTHON),'-X','utf8',str(ROOT/'g1_stereo_bridge/upstream/handoff_1080p_14p24_20261019/pc/receive_paired_pose.py'),
            '--source-mode','1080p','--pair-limit-us','16667',
            '--host',cfg['board_host'],'--port',str(cfg['board_port']),*extra],env=environment(),check=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        raise SystemExit(1)
