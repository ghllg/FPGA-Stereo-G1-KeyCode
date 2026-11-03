from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import struct
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

from .core import HOME, ROOT, TRAIN, JsonlTail, digest, read_json, write_json, summarize_comparison
from .pairing import mutual_nearest_pairs
from .hybrid_candidate import latency_settings

DEFAULTS = dict(host='192.168.137.2', ssh_port=22, user='root', key_file='', board_port=18084,
                remote_root='/home/root/mocap_station', seconds=10, max_mib=512,
                left_bus='usb-xhci-hcd.0.auto-1.4', right_bus='usb-xhci-hcd.0.auto-1.2',
                model_dir='/home/root/mocap_station/models', pair_limit_us=10000,
                left_roi=[555,0,810,1080], right_roi=[555,0,810,1080],
                calibration='', checkpoint='', binding='', g1_dir='', overview_camera=0,
                vision_python='', g1_url='http://127.0.0.1:8771/ws',
                board_runtime_record=str(HOME/'data/board_runtime_current.json'))


def load_config():
    path=HOME/'config.json'
    return {**DEFAULTS, **(read_json(path) if path.exists() else {})}


def motion_config(config, motion_mode):
    selected = dict(config)
    binding = config.get('standing_upper_body_binding') if motion_mode == 'standing_upper_body' else None
    scope = 'standing_upper_body'
    if motion_mode == 'full_body' and config.get('full_body_check'):
        binding = config.get('full_body_check_binding')
        scope = 'full_body_standing_check'
        if not binding: raise ValueError('缺少本次全身短测的动作框绑定')
    if binding:
        profile = read_json(binding)
        if profile.get('motion_scope') != scope:
            raise ValueError('测试动作框与所选跟随范围不一致')
        selected.update(binding=binding, **{side+'_roi':list(profile['roi_xywh'][side])
                                         for side in ('left','right')})
    return selected


def pairing20_host(archived, boot=None):
    path=HOME/'profiles/pairing20_board.json';candidate=read_json(path)
    verification=Path(candidate['verification']['path'])
    if digest(verification)!=candidate['verification']['sha256']:
        raise ValueError('20 ms 板端候选验证记录已改变')
    proof=read_json(verification)
    binary=Path(candidate['binary']['path']);sha=candidate['binary']['sha256']
    if (candidate.get('schema')!='pairing20_board/1' or candidate.get('pair_limit_us')!=20000
            or candidate.get('pairing_policy')!='nearest_raw_20ms'
            or candidate.get('baseline_host_sha256')!=archived['stereo_host']
            or digest(binary)!=sha or proof.get('host_sha256')!=sha or not proof.get('complete')
            or proof.get('pair_limit_us')!=20000 or proof.get('pairing_policy')!='nearest_raw_20ms'
            or proof.get('camera_opened') is not False or not proof.get('saved_input_heatmaps_exact')
            or proof.get('actual_PL_calls',0)<=0 or not proof.get('listener_no_camera_passed')
            or not proof.get('cleanup_verified') or (boot is not None and proof.get('boot_id')!=boot)):
        raise ValueError('20 ms 板端程序、同 boot 实测或来源验证不完整')
    remote=candidate['remote']
    if not re.fullmatch(r'/tmp/[A-Za-z0-9_/-]+',remote) or '..' in PurePosixPath(remote).parts:
        raise ValueError('20 ms 候选必须位于独立临时目录')
    return dict(remote=remote,files={**archived,'stereo_host':sha},boot_id=proof['boot_id'],
                profile=dict(path=str(path),sha256=digest(path)),verification=candidate['verification'])


def new_session(name,*,root=None):
    root=Path(root) if root is not None else HOME/'data/sessions'
    path=root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'session.json',dict(schema='motion_session/1',name=name,state='preparing',created=time.strftime('%Y-%m-%d %H:%M:%S')))
    return path


def snapshot_assets(path, config):
    assets={}
    for key in ('calibration','checkpoint','binding'):
        if config.get(key):
            src=Path(config[key])
            if not src.is_file():raise ValueError(f'{key} 文件不存在: {src}')
            dest=path/'assets'/(key+src.suffix);dest.parent.mkdir(exist_ok=True)
            shutil.copy2(src,dest)
            assets[key]=dict(path=dest.relative_to(path).as_posix(),sha256=digest(dest))
    return assets


def safe_extract(archive, destination):
    destination=Path(destination).resolve()
    with tarfile.open(archive,'r:*') as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>10*1024**3:raise ValueError('归档超出大小限制')
        for m in members:
            name=m.name.replace('\\','/')
            if m.issym() or m.islnk() or not (m.isfile() or m.isdir()):raise ValueError('归档含不支持的文件类型')
            target=(destination/name).resolve()
            if not target.is_relative_to(destination):raise ValueError('归档路径越界')
        for m in members:
            target=(destination/m.name).resolve()
            if m.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src, target.open('xb') as out:shutil.copyfileobj(src,out)


def import_capture(path, limit_us=10000):
    path=Path(path);raw=path/'raw'
    sides={}
    for side in ('left','right'):
        with (raw/(side+'.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        parsed=[];last=-1
        for row in rows:
            stamp=int(row['source_timestamp_us']);i=int(row['capture_index'])
            if stamp<=last:raise ValueError('采集时间戳未递增')
            image=raw/f'{side}_{i}.jpg'
            if not image.is_file() or image.stat().st_size!=int(row['jpeg_bytes']):raise ValueError('原图缺失或大小不符')
            parsed.append(dict(capture_index=i,timestamp_us=stamp,sequence=int(row['sequence']),path=image.relative_to(path).as_posix()))
            last=stamp
        if not parsed:raise ValueError('相机没有有效帧')
        sides[side]=parsed
    origin=min(sides['left'][0]['timestamp_us'],sides['right'][0]['timestamp_us'])
    media={}
    for side,rows in sides.items():
        with (path/(side+'_index.jsonl')).open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps({**row,'time_s':(row['timestamp_us']-origin)/1e6})+'\n')
        media[side]=dict(index=side+'_index.jsonl',kind='jpeg_sequence',source_size=[1920,1080])
    pairs=[];left,right=sides['left'],sides['right'];i=j=0
    if limit_us==20000:
        matches=mutual_nearest_pairs([r['timestamp_us'] for r in left],[r['timestamp_us'] for r in right],limit_us)
        pairs=[dict(left=left[i],right=right[j],delta_us=delta) for i,j,delta in matches]
    else:
        while i<len(left) and j<len(right):
            delta=right[j]['timestamp_us']-left[i]['timestamp_us']
            if abs(delta)<=limit_us:
                pairs.append(dict(left=left[i],right=right[j],delta_us=delta));i+=1;j+=1
            elif delta>0:i+=1
            else:j+=1
    write_json(path/'pairs.json',dict(pair_limit_us=limit_us,pairs=pairs,exposure_sync_verified=False))
    state=read_json(path/'session.json')
    state.update(origin_ns=origin*1000,media=media,duration_s=(max(left[-1]['timestamp_us'],right[-1]['timestamp_us'])-origin)/1e6,
                 state='recorded',pair_count=len(pairs),source_size=[1920,1080])
    write_json(path/'session.json',state)
    return pairs


class BoardClient:
    """OpenSSH transport. Each operation owns a unique board directory."""
    def __init__(self, config, progress=lambda s:None, cancel=None):
        self.config=config;self.progress=progress;self.cancel=cancel or threading.Event();self.remote_job=None
        self.prepared_live=None;self.owned_live=None;self.pending_live=None
        for key in ('host','user'):
            if not re.fullmatch(r'[A-Za-z0-9_.:-]+',str(config[key])) or str(config[key]).startswith('-'):
                raise ValueError('SSH 地址或用户格式错误')
        root=config['remote_root']
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+',root) or '..' in PurePosixPath(root).parts or root in ('/','/tmp','/home','/home/root'):
            raise ValueError('板端目录必须是独立的绝对路径')
        self.root=root.rstrip('/')
        (HOME/'data').mkdir(exist_ok=True)

    def args(self):
        c=self.config
        args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=4','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=3',
              '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(HOME/'data/known_hosts'),'-p',str(int(c['ssh_port']))]
        if c.get('key_file'):args+=['-i',c['key_file']]
        return args+[c['user']+'@'+c['host']]

    def command(self, command, timeout=30, stdin=None, output=None):
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        err=HOME/'output/ssh_error.log';err.parent.mkdir(exist_ok=True)
        with err.open('wb') as error, tempfile.TemporaryFile() as captured:
            process=subprocess.Popen(self.args()+[command],stdin=stdin or subprocess.DEVNULL,
                stdout=output if output is not None else captured,stderr=error,creationflags=flags)
            deadline=time.monotonic()+timeout;sent_cancel=False;cancel_deadline=None
            # Even status/proc queries can exceed a pipe buffer before exit.
            while process.poll() is None:
                if (self.cancel.is_set() or time.monotonic()>deadline) and not sent_cancel:
                    if self.remote_job:
                        try:
                            subprocess.run(self.args()+['touch '+shlex.quote(self.remote_job+'/cancel')],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8,creationflags=flags)
                        except subprocess.TimeoutExpired:pass
                    sent_cancel=True;cancel_deadline=time.monotonic()+20
                if cancel_deadline and time.monotonic()>cancel_deadline:
                    process.terminate();process.wait(timeout=5)
                    raise TimeoutError('板端任务响应超时；远端目录已保留，请检查任务状态')
                time.sleep(.05)
            captured.seek(0)
            data=captured.read() if output is None else b''
        if sent_cancel:raise RuntimeError('已取消，保留已录数据')
        if process.returncode:raise RuntimeError(err.read_text(encoding='utf-8',errors='replace')[-2000:] or f'板端退出码 {process.returncode}')
        return data.decode('utf-8',errors='replace')

    def upload(self, local, remote):
        with Path(local).open('rb') as source:self.command('cat > '+shlex.quote(remote),timeout=180,stdin=source)

    def download(self, remote, local, timeout=180):
        with Path(local).open('wb') as target:self.command('cat '+shlex.quote(remote),timeout=timeout,output=target)

    def probe(self):
        result=self.command('uname -m; cat /proc/sys/kernel/random/boot_id; df -Pk '+shlex.quote('/home/root')+'; test -c /dev/u02_stream_pipeline && echo ABI9_NODE_PRESENT || true')
        if 'aarch64' not in result:raise ValueError('目标不是 AArch64 板端')
        return result

    def prepare_live_service(self, local_output):
        """Read-only qualification before the user's START/countdown; no service is started."""
        output=Path(local_output);output.mkdir(parents=True,exist_ok=True)
        proof=dict(schema='board_service_preparation/1',ok=False,host=self.config['host'],
                   checked_at_utc=datetime.now(timezone.utc).isoformat(),errors=[],camera_opened=False)
        self.prepared_live=None
        try:
            from pose3d_train.geometry import Stereo
            cfg=self.config
            stereo=Stereo(cfg['calibration']);binding=read_json(cfg['binding'])
            rois={side:list(cfg[side+'_roi']) for side in ('left','right')}
            if any(len(v)!=4 or any(type(x) is not int for x in v) or v[0]<0 or v[1]<0 or v[2]<=0 or v[3]<=0
                   or v[0]+v[2]>1920 or v[1]+v[3]>1080 for v in rois.values()):raise ValueError('ROI 必须位于 1920×1080 原图内')
            if (not Path(cfg['checkpoint']).is_file() or binding.get('schema')!='board_session_binding/1.0'
                    or binding.get('calibration_hash')!=stereo.hash or stereo.calibration.get('image_size')!=[1920,1080]
                    or binding.get('camera_physical_ids')!=stereo.calibration.get('camera_physical_ids')
                    or binding.get('camera_physical_ids')!=['USB1-1.4:1.0','USB1-1.2:1.0']
                    or binding.get('roi_xywh')!=rois or int(cfg['board_port'])!=18084
                    or cfg['left_bus']!='usb-xhci-hcd.0.auto-1.4' or cfg['right_bus']!='usb-xhci-hcd.0.auto-1.2'):
                raise ValueError('本地模型/标定/物理相机/ROI/正式端口不一致')
            record=Path(self.config['board_runtime_record'])
            if not record.is_file():raise ValueError('板端现场资格待检查：board_runtime_current.json 尚不存在')
            index=read_json(record)
            if index.get('schema')!='board_runtime_current/1':raise ValueError('现场资格索引 schema 不符')
            evidence={}
            for name in ('live_stage','run','preflight','verification'):
                item=index['evidence'][name];path=Path(item['path'])
                if not path.is_absolute() or digest(path)!=item['sha256']:raise ValueError('现场资格证据损坏：'+name)
                evidence[name]=read_json(path)
            stage,run,preflight,verification=(evidence[n] for n in ('live_stage','run','preflight','verification'))
            boot=stage['boot_id'];remote=stage['remote'];files=stage['files']
            if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote) or '..' in PurePosixPath(remote).parts:
                raise ValueError('现场制品目录不合法')
            if (not all(v.get('complete') is True and v.get('boot_id')==boot for v in evidence.values())
                    or run.get('remote')!=remote or preflight.get('remote')!=remote
                    or run.get('dma_ambiguous') is not False or run.get('dma_abi')!=9
                    or not run.get('all_full_heatmaps_exact') or run.get('actual_PL_calls',0)<=0
                    or not run.get('cleanup_verified') or not preflight.get('cleanup_verified')
                    or not verification.get('CHW_programmed_and_idle_verified')
                    or not verification.get('TCP_listener_no_camera_preflight_passed')
                    or verification.get('board_ip')!=self.config['host']):
                raise ValueError('真实 PL、DMA、位流或无相机预检资格不完整，或证据 boot 不一致')
            archived=read_json(TRAIN.parent/'board_resume_20261023_112300/live_stage.json')['files']
            for name in ('stereo_host','model.stream.param','model.opt.bin','u02_stream_pipeline.ko','u02_uvc1080.ko'):
                if files.get(name)!=archived[name]:raise ValueError('不是已核验正式板端制品：'+name)
            candidate20=None
            if cfg.get('pair_limit_us')==20000:
                if (cfg.get('pairing_policy')!='nearest_raw_20ms' or binding.get('pair_limit_us')!=20000
                        or binding.get('pairing_policy')!='nearest_raw_20ms'):
                    raise ValueError('20 ms 配置与会话绑定不一致')
                candidate20=pairing20_host(archived,boot)
                if cfg.get('hybrid_refinement'):
                    from .hybrid_candidate import board_qualification
                    candidate20=board_qualification(candidate20,boot)
                remote=candidate20['remote'];files={**files,**candidate20['files']}
            if binding.get('model_sha256')!=files['model.opt.bin']:raise ValueError('本地绑定与板端 2D 模型不一致')
            script="printf 'BOOT:%s\\n' \"$(cat /proc/sys/kernel/random/boot_id)\"; uname -m; "
            script+="printf 'BOARD_MODEL:%s\\n' \"$(tr -d '\\000' < /proc/device-tree/model 2>/dev/null)\"; "
            script+='test -c /dev/u02_stream_pipeline && test -d /sys/module/u02_stream_pipeline && echo ABI9_PRESENT; '
            for name in ('stereo_host','model.stream.param','model.opt.bin','u02_stream_pipeline.ko','u02_uvc1080.ko'):
                script+='sha256sum '+shlex.quote(remote+'/'+name)+'; '
            script+='for d in 1-1.4:1.0 1-1.2:1.0; do printf "DRIVER:%s:%s\\n" "$d" "$(readlink -f /sys/bus/usb/devices/$d/driver)"; done; '
            script+='for v in /sys/class/video4linux/video*; do test "$(cat "$v/index")" = 0 || continue; printf "NODE:%s:%s\\n" "${v##*/}" "$(readlink -f "$v/device")"; done; '
            script+='for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f" 2>/dev/null); case "$t" in /dev/video*|/dev/u02*) printf "BUSY:%s\\n" "$t";; esac; done; '
            script+="awk '$2 ~ /:"+format(int(self.config['board_port']),'04X')+"$/ && $4 == \"0A\" {print \"LISTENER_BUSY\"}' /proc/net/tcp; true"
            text=self.command(script,timeout=30);proof['remote_evidence']=text
            if not re.search(r'(?m)^BOOT:'+re.escape(boot)+r'\r?$',text):raise ValueError('板已重启：须重新完成现场资格，不能复用旧 boot')
            if not re.search(r'(?m)^aarch64\r?$',text) or 'ABI9_PRESENT' not in text:raise ValueError('AArch64 或已加载 ABI9 节点缺失')
            if re.search(r'(?m)^(BUSY:|LISTENER_BUSY)',text):raise ValueError('已有相机、DMA 所有者或监听服务；保留原进程，不自动停止')
            for name in ('stereo_host','model.stream.param','model.opt.bin','u02_stream_pipeline.ko','u02_uvc1080.ko'):
                if not re.search(r'(?m)^'+files[name]+r'\s+'+re.escape(remote+'/'+name)+r'\r?$',text):raise ValueError('现场制品 SHA 不一致：'+name)
            nodes=[]
            for port in ('1-1.4','1-1.2'):
                if not re.search(r'(?m)^DRIVER:'+re.escape(port+':1.0')+r':[^\r\n]*/u02_uvc1080\r?$',text):
                    raise ValueError('1080p 驱动未就绪；不会自动 bind 或加载模块')
                matches=re.findall(r'(?m)^NODE:(video[0-9]+):[^\r\n]*/'+re.escape(port+':1.0')+r'\r?$',text)
                if len(matches)!=1:raise ValueError('相机物理口对应的图像节点不唯一：'+port)
                nodes.append('/dev/'+matches[0])
            if set(nodes)!={'/dev/video0','/dev/video2'}:raise ValueError('当前正式 stereo_host 只核验过 video0/video2 图像节点')
            binding=dict(binding);binding.pop('session_id',None);binding['boot_id']=boot
            proof.update(ok=True,boot_id=boot,remote=remote,files=files,camera_nodes=nodes,
                         board_model=next((v.removeprefix('BOARD_MODEL:') for v in text.splitlines() if v.startswith('BOARD_MODEL:')),None),
                         binding=binding,model_sha256=files['model.opt.bin'],camera_physical_ids=binding['camera_physical_ids'],roi_xywh=rois,
                         qualification_index=dict(path=str(record.resolve()),sha256=digest(record)),
                         evidence=index['evidence'],local_output=str(output.resolve()),
                         limitations=['Current boot is matched to prior physical qualification evidence; this read-only check does not itself run PL or prove no intervening reprogramming.'])
            if candidate20:
                proof.update(pair_limit_us=20000,pairing_policy='nearest_raw_20ms',host_candidate=candidate20)
            self.prepared_live=proof
        except (KeyError,ValueError,TypeError,OSError,RuntimeError,TimeoutError) as e:proof['errors'].append(str(e))
        write_json(output/'board_service_preparation.json',proof)
        return proof

    def start_live_service(self, local_output, seconds):
        """Caller owns explicit START and 3-second countdown; only launch our qualified host."""
        prepared=self.prepared_live;output=Path(local_output)
        if not prepared or not prepared['ok'] or prepared['local_output']!=str(output.resolve()):
            raise ValueError('请先完成本次输出目录的只读板端资格检查')
        if self.owned_live or self.pending_live:raise ValueError('本对象已有启动记录，请先核清旧服务状态')
        seconds=int(seconds)
        if not 1<=seconds<=300:raise ValueError('实时检查时长必须在 1–300 秒内')
        cfg=self.config;remote=prepared['remote'];job=remote+'/ui_'+uuid.uuid4().hex
        input_ready_budget=20 if cfg.get('full_body_check') else 10 if cfg.get('motion_mode')=='standing_upper_body' else 0
        board_seconds=seconds+10+input_ready_budget
        if self.cancel.is_set():raise RuntimeError('已取消，未启动板端服务')
        if cfg['host']!=prepared['host'] or {s:list(cfg[s+'_roi']) for s in ('left','right')}!=prepared['roi_xywh']:
            raise ValueError('配置在准备后改变，请重新检查')
        args=[remote+'/stereo_host','pl-stream',remote+'/model.stream.param',remote+'/model.opt.bin',
              ','.join(prepared['camera_nodes']),'result.json','1920','1080','555','0','810','1080']
        env=dict(POSE_BIND_IP=cfg['host'],POSE_LEFT_ROI=','.join(map(str,cfg['left_roi'])),
                 POSE_RIGHT_ROI=','.join(map(str,cfg['right_roi'])),POSE_DEBUG_SNAPSHOTS='1',POSE_RUN_SECONDS=str(board_seconds),
                 POSE_INFLIGHT_PAIRS=str(latency_settings(cfg)['board_inflight_pairs']),POSE_THREADS='1',POSE_PAIR_NETWORK='1',POSE_PAIR_WAIT_CLIENT_MS='10000',
                 POSE_QUANT_VECTOR='1',GOMP_SPINCOUNT='300000',OMP_WAIT_POLICY='PASSIVE')
        if cfg.get('hybrid_refinement'):env['POSE_HYBRID_JPEG']='1'
        launch=shlex.join(['env']+[k+'='+v for k,v in env.items()]+args)
        # Recheck boot/ownership immediately before the sole mutation (our unique job/start).
        script='test "$(cat /proc/sys/kernel/random/boot_id)" = '+shlex.quote(prepared['boot_id'])+' || exit 41; '
        script+="! awk '$2 ~ /:"+format(int(cfg['board_port']),'04X')+"$/ && $4 == \"0A\" {found=1} END {exit !found}' /proc/net/tcp || exit 42; "
        script+='for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f" 2>/dev/null); case "$t" in /dev/video*|/dev/u02*) exit 43;; esac; done; '
        script+='test -c /dev/u02_stream_pipeline && test -d /sys/module/u02_stream_pipeline || exit 46; '
        script+='for d in 1-1.4:1.0 1-1.2:1.0; do test "$(readlink -f /sys/bus/usb/devices/$d/driver)" = /sys/bus/usb/drivers/u02_uvc1080 || exit 47; done; '
        for name in ('stereo_host','model.stream.param','model.opt.bin'):
            script+='test "$(sha256sum '+shlex.quote(remote+'/'+name)+' | cut -d \' \' -f 1)" = '+shlex.quote(prepared['files'][name])+' || exit 45; '
        script+='mkdir '+shlex.quote(job)+' && cd '+shlex.quote(job)+' || exit 44; '
        script+="( trap '' HUP; "+launch+' > run.log 2>&1 < /dev/null & pid=$!; printf "%s\\n" "$pid" > pid; '
        script+='awk \'{print $22}\' /proc/$pid/stat > start_ticks; wait "$pid"; printf "%s\\n" "$?" > exit ) < /dev/null > wrapper.log 2>&1 & '
        script+='ready=0; n=0; while test "$n" -lt 160; do n=$((n+1)); '
        script+='if test -s '+shlex.quote(job+'/start_ticks')+'; then pid=$(cat '+shlex.quote(job+'/pid')+'); '
        script+="inodes=$(awk '$2 ~ /:46A4$/ && $4 == \"0A\" {print $10}' /proc/net/tcp); "
        script+='for inode in $inodes; do for f in /proc/$pid/fd/*; do test "$(readlink "$f" 2>/dev/null)" != "socket:[$inode]" || ready=1; done; done; fi; '
        script+='test "$ready" = 0 || break; test ! -f '+shlex.quote(job+'/exit')+' || break; sleep 0.05; done; '
        script+='printf "BOOT\\t%s\\nPID\\t%s\\nSTART_TICKS\\t%s\\nREADY\\t%s\\n" "$(cat /proc/sys/kernel/random/boot_id)" "$(cat '+shlex.quote(job+'/pid')+')" "$(cat '+shlex.quote(job+'/start_ticks')+')" "$ready"'
        write_json(output/'board_service_start_request.json',dict(remote_job=job,boot_id=prepared['boot_id'],
                   seconds=seconds,board_auto_stop_seconds=board_seconds,input_ready_budget_seconds=input_ready_budget,
                   command=launch,preparation='board_service_preparation.json',
                   capture_protocol=cfg.get('capture_protocol'),caller_START_and_countdown_required=True))
        self.remote_job=job
        self.pending_live=dict(boot_id=prepared['boot_id'],remote_job=job,exe=remote+'/stereo_host',
                               sha256=prepared['files']['stereo_host'],local_output=str(output.resolve()))
        answer=dict(line.split('\t',1) for line in self.command(script,timeout=20).splitlines() if '\t' in line)
        if not answer.get('PID','').isdigit() or not answer.get('START_TICKS','').isdigit():
            raise RuntimeError('启动回执不完整，请检查保留的远端目录：'+job)
        self.owned_live=dict(self.pending_live,pid=int(answer['PID']),start_ticks=answer['START_TICKS'])
        write_json(output/'board_service_owned.json',self.owned_live)
        ok=answer.get('BOOT')==prepared['boot_id'] and answer.get('READY')=='1'
        proof=dict(schema='board_live_preflight/1',ok=ok,checked_at_utc=datetime.now(timezone.utc).isoformat(),
                   checked_pc_ns=time.perf_counter_ns(),host=cfg['host'],port=int(cfg['board_port']),boot_id=answer.get('BOOT'),
                   board_model=prepared['board_model'],
                   model_sha256=prepared['model_sha256'],camera_physical_ids=prepared['camera_physical_ids'],roi_xywh=prepared['roi_xywh'],
                   binding=dict(prepared['binding']) if ok else None,owned=dict(self.owned_live),
                   checks=dict(current_boot=dict(passed=answer.get('BOOT')==prepared['boot_id']),owned_listener_ready=dict(passed=answer.get('READY')=='1')),
                   camera_geometry=dict(verified=False,reason='Check camera position and orientation on site; USB identity is not calibration geometry.'),
                   warnings=['当前机位需现场检查；诊断标定的地面尺度和腕点精度尚未验收。'],
                   limitations=prepared['limitations']+['Preparation hashes are rechecked immediately before exec; no model load or full proc scan occurs in the 10-second client gate.'],
                   preparation='board_service_preparation.json',remote_response=answer)
        if prepared.get('pair_limit_us')==20000:
            proof.update(pair_limit_us=20000,pairing_policy='nearest_raw_20ms',host_candidate=prepared['host_candidate'])
        write_json(output/'board_live_preflight.json',proof)
        return proof

    def stop_live_service(self):
        """SIGTERM only the exact process created here; never kill an existing or reused PID."""
        owned=self.owned_live
        cleanup=BoardClient(self.config,progress=self.progress,cancel=threading.Event())
        if not owned and self.pending_live:
            pending=self.pending_live;job=pending['remote_job']
            try:
                script='test "$(cat /proc/sys/kernel/random/boot_id)" = '+shlex.quote(pending['boot_id'])+' || exit 51; '
                script+='cat '+shlex.quote(job+'/pid')+' '+shlex.quote(job+'/start_ticks')
                reply=cleanup.command(script,timeout=8).splitlines()
                if len(reply)!=2 or not all(v.isdigit() for v in reply):raise ValueError('本任务 PID 启动回执尚不完整')
                owned=dict(pending,pid=int(reply[0]),start_ticks=reply[1]);self.owned_live=owned
                write_json(Path(pending['local_output'])/'board_service_owned.json',dict(owned,recovered_after_lost_start_reply=True))
            except (OSError,RuntimeError,TimeoutError,ValueError) as e:
                result=dict(stopped=False,remote_job=job,error='启动/退出状态未确认：'+str(e),logs_retained=True)
                write_json(Path(pending['local_output'])/'board_service_stop.json',result);return result
        if not owned:return dict(stopped=False,reason='No service owned by this client')
        pid=str(owned['pid']);job=owned['remote_job']
        script='test "$(cat /proc/sys/kernel/random/boot_id)" = '+shlex.quote(owned['boot_id'])+' || exit 51; '
        script+='if test ! -d /proc/'+pid+'; then echo ALREADY_EXITED; else '
        script+='test "$(awk \'{print $22}\' /proc/'+pid+'/stat)" = '+shlex.quote(owned['start_ticks'])+' || exit 52; '
        script+='test "$(readlink -f /proc/'+pid+'/exe)" = '+shlex.quote(owned['exe'])+' || exit 53; '
        script+='test "$(readlink -f /proc/'+pid+'/cwd)" = '+shlex.quote(job)+' || exit 54; '
        script+='test "$(sha256sum /proc/'+pid+'/exe | cut -d \' \' -f 1)" = '+shlex.quote(owned['sha256'])+' || exit 55; '
        script+='kill -TERM '+pid+' && echo TERM_REQUESTED; fi; '
        script+='n=0; while test "$n" -lt 100 && test ! -f '+shlex.quote(job+'/exit')+'; do n=$((n+1)); sleep 0.05; done; '
        script+='if test -f '+shlex.quote(job+'/exit')+'; then printf "EXIT:%s\\n" "$(cat '+shlex.quote(job+'/exit')+')"; else echo EXIT_UNCONFIRMED; fi'
        result=dict(stopped=False,signal_requested=False,remote_job=job,logs_retained=True)
        try:
            text=cleanup.command(script,timeout=8)
            exit_match=re.search(r'(?m)^EXIT:(\d+)\r?$',text)
            result.update(response=text,stopped=bool(exit_match),signal_requested='TERM_REQUESTED' in text,
                          exit_code=int(exit_match[1]) if exit_match else None)
        except (OSError,RuntimeError,TimeoutError) as e:result['error']=str(e)
        artifacts=Path(owned['local_output'])/('board_artifacts_'+uuid.uuid4().hex[:8]);artifacts.mkdir()
        try:
            if result['stopped']:
                archive=artifacts/'board.tar'
                with archive.open('wb') as out:cleanup.command('cd '+shlex.quote(job)+' && tar -cf - .',timeout=180,output=out)
                raw=artifacts/'raw';raw.mkdir();safe_extract(archive,raw)
                result['artifact_files']=[dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(raw.rglob('*')) if p.is_file()]
                result['artifacts_complete']=True
            else:
                cleanup.download(job+'/run.log',artifacts/'partial_run.log',timeout=15)
                result['artifacts_complete']=False
            result['artifact_directory']=str(artifacts.resolve())
        except (OSError,RuntimeError,TimeoutError,ValueError,tarfile.TarError) as e:
            result.update(artifact_error=str(e),artifacts_complete=False,error='板端证据未完整取回：'+str(e))
        write_json(Path(owned['local_output'])/'board_service_stop.json',result)
        return result

    def preflight_live(self):
        """Inspect an already running qualified service; never start cameras or drivers."""
        from pose3d_train.geometry import Stereo
        cfg=self.config
        proof=dict(schema='board_live_preflight/1',ok=False,checked_at_utc=datetime.now(timezone.utc).isoformat(),
                   checked_pc_ns=time.perf_counter_ns(),host=cfg['host'],port=int(cfg['board_port']),
                   boot_id=None,model_sha256=None,camera_physical_ids=None,roi_xywh=None,binding=None,
                   camera_geometry=dict(verified=False,reason='Camera position and orientation must be checked on site; matching USB ports does not validate the calibration.'),
                   checks={},warnings=[],limitations=[
                       'Read-only proc/sysfs identity check; no camera open, stream start, driver load or FPGA write.',
                       'ABI9 node/module presence is not an ioctl ABI, loaded bitstream or DMA health test.',
                       'USB physical ports and sysfs index are checked; image format, exposure and synchronization require a separate capture.',
                       'On-disk hashes and current process identity do not prove immutable model bytes throughout a later run.'])
        def check(name,passed,detail):
            proof['checks'][name]=dict(passed=bool(passed),detail=detail)
            return bool(passed)
        try:
            calibration=Path(cfg['calibration']);checkpoint=Path(cfg['checkpoint']);template=Path(cfg['binding'])
            if not all(p.is_file() for p in (calibration,checkpoint,template)):
                raise ValueError('需要本地标定、3D 模型和会话绑定模板文件')
            stereo=Stereo(calibration);bound=read_json(template);cal=stereo.calibration
            rois={side:list(cfg[side+'_roi']) for side in ('left','right')}
            valid_roi=all(len(v)==4 and all(type(x) is int for x in v) and v[0]>=0 and v[1]>=0
                          and v[2]>0 and v[3]>0 and v[0]+v[2]<=1920 and v[1]+v[3]<=1080 for v in rois.values())
            if (bound.get('schema')!='board_session_binding/1.0' or bound.get('calibration_hash')!=stereo.hash
                    or bound.get('camera_physical_ids')!=cal.get('camera_physical_ids')
                    or cal.get('camera_ids')!=['left','right'] or cal.get('image_size')!=[1920,1080]
                    or bound.get('roi_xywh')!=rois or not valid_roi
                    or not re.fullmatch('[0-9a-f]{64}',bound.get('model_sha256',''))):
                raise ValueError('标定、物理左右相机、1080p、ROI 或 2D 模型模板不一致')
            physical=bound['camera_physical_ids']
            if len(physical)!=2 or len(set(physical))!=2 or any(not re.fullmatch(r'USB[0-9]+-[0-9.]+:[0-9.]+',p) for p in physical):
                raise ValueError('物理相机模板必须含两个不同的 USB 接口')
            stage=read_json(TRAIN.parent/'board_resume_20261023_112300/live_stage.json')
            expected=stage['files']
            if cfg.get('pair_limit_us')==20000 and (cfg.get('pairing_policy')!='nearest_raw_20ms'
                    or bound.get('pair_limit_us')!=20000 or bound.get('pairing_policy')!='nearest_raw_20ms'):
                raise ValueError('20 ms 配置与会话绑定不一致')
            candidate20=pairing20_host(expected) if cfg.get('pair_limit_us')==20000 else None
            if candidate20:
                expected=candidate20['files']
                proof.update(pair_limit_us=20000,pairing_policy='nearest_raw_20ms',host_candidate=candidate20)
            if expected['model.opt.bin']!=bound['model_sha256']:
                raise ValueError('绑定模型不属于归档的正式板端制品')
            proof.update(camera_physical_ids=physical,roi_xywh=rois,
                         qualified_manifest=dict(path=str(TRAIN.parent/'board_resume_20261023_112300/live_stage.json'),
                                                 sha256=digest(TRAIN.parent/'board_resume_20261023_112300/live_stage.json')),
                         local_assets={name:dict(path=str(p.resolve()),sha256=digest(p)) for name,p in
                                       [('calibration',calibration),('checkpoint',checkpoint),('binding_template',template)]})
            check('local_template',True,'Calibration hash, physical order, source size, ROI and qualified 2D model agree.')
            proof['warnings'].append('机位是否移动需上板时现场检查；USB 左右身份匹配不代表标定几何仍有效。')
            if cal.get('accepted_for_reconstruction') is not True:
                proof['warnings'].append('当前标定仍为诊断候选；绝对地面高度、腕点精度未验收。')
        except (KeyError,ValueError,TypeError,OSError) as e:
            check('local_template',False,str(e));return proof
        script=r'''set -f
printf 'ARCH\t%s\n' "$(uname -m)"
printf 'BOOT\t%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
printf 'BOARD_MODEL\t%s\n' "$(tr -d '\000' < /proc/device-tree/model 2>/dev/null)"
test ! -c /dev/u02_stream_pipeline || printf 'ABI_NODE\tcharacter\n'
for m in u02_stream_pipeline u02_uvc1080; do
 test ! -d /sys/module/$m || printf 'MODULE\t%s\n' "$m"
done
for v in /sys/class/video4linux/video*; do
 test -d "$v" || continue
 printf 'VIDEO\t%s\t%s\t%s\t%s\t%s\n' "${v##*/}" "$(cat "$v/index")" "$(readlink -f "$v/device")" "$(readlink -f "$v/device/driver")" "$(cat "$v/name")"
done
inodes=$(awk '$2 ~ /:PORTHEX$/ && $4 == "0A" {print $10}' /proc/net/tcp)
for inode in $inodes; do
 printf 'LISTENER\t%s\n' "$inode"
 for p in /proc/[0-9]*; do
  owns=0
  for f in "$p"/fd/*; do test "$(readlink "$f" 2>/dev/null)" != "socket:[$inode]" || owns=1; done
  test "$owns" = 1 || continue
  pid=${p##*/}; exe=$(readlink -f "$p/exe"); cwd=$(readlink -f "$p/cwd")
  printf 'PROC\t%s\t%s\t%s\n' "$pid" "$exe" "$cwd"
  test "${exe##*/}" = stereo_host || continue
  tr '\000' '\n' < "$p/cmdline" | while IFS= read -r a; do printf 'ARG\t%s\t%s\n' "$pid" "$a"; done
  tr '\000' '\n' < "$p/environ" | while IFS= read -r a; do
   case "$a" in POSE_LEFT_ROI=*|POSE_RIGHT_ROI=*|POSE_BIND_IP=*) printf 'ENV\t%s\t%s\n' "$pid" "$a";; esac
  done
  oldifs=$IFS; IFS='
'
  set -- $(tr '\000' '\n' < "$p/cmdline"); IFS=$oldifs
  printf 'HASH\t%s\tstereo_host\t%s\n' "$pid" "$(sha256sum "$p/exe" | cut -d ' ' -f 1)"
  param=$3; model=$4
  case "$param" in /*) ;; *) param=$cwd/$param;; esac
  case "$model" in /*) ;; *) model=$cwd/$model;; esac
  printf 'HASH\t%s\tmodel.stream.param\t%s\n' "$pid" "$(sha256sum "$param" | cut -d ' ' -f 1)"
  printf 'HASH\t%s\tmodel.opt.bin\t%s\n' "$pid" "$(sha256sum "$model" | cut -d ' ' -f 1)"
  printf 'PROC_END\t%s\t%s\n' "$pid" "$(readlink -f "$p/exe")"
 done
done
printf 'BOOT_END\t%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
'''.replace('PORTHEX',format(proof['port'],'04X'))
        try:
            raw=self.command('sh -c '+shlex.quote(script),timeout=30)
        except (OSError,RuntimeError,TimeoutError) as e:
            check('ssh_readonly',False,str(e));return proof
        rows=[line.split('\t') for line in raw.splitlines() if line]
        def values(tag):return [row[1:] for row in rows if row[0]==tag]
        def single(tag):
            found=values(tag)
            return found[0][0] if len(found)==1 and found[0] else None
        proof['remote_evidence']=dict(records=rows)
        boot=single('BOOT');proof['boot_id']=boot;proof['board_model']=single('BOARD_MODEL')
        check('ssh_readonly',True,'Only proc/sysfs reads and sha256sum; no network pose client opened.')
        check('architecture',single('ARCH')=='aarch64',single('ARCH'))
        check('boot_identity',bool(boot and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',boot)) and boot==single('BOOT_END'),boot)
        if candidate20:check('candidate_same_boot',boot==candidate20['boot_id'],candidate20['boot_id'])
        check('board_model_reported',bool(proof['board_model']),proof['board_model'])
        modules=[v[0] for v in values('MODULE')]
        check('abi9_node_and_module',single('ABI_NODE')=='character' and 'u02_stream_pipeline' in modules,modules)
        processes=values('PROC');check('unique_listener',len(values('LISTENER'))==1 and len(processes)==1,processes)
        if len(processes)==1 and len(processes[0])==3:
            pid,exe,cwd=processes[0]
            args=[v[1] for v in values('ARG') if len(v)==2 and v[0]==pid]
            env=dict(v[1].split('=',1) for v in values('ENV') if len(v)==2 and v[0]==pid and '=' in v[1])
            hashes={v[1]:v[2] for v in values('HASH') if len(v)==3 and v[0]==pid}
            proof['model_sha256']=hashes.get('model.opt.bin')
            check('qualified_artifacts',all(hashes.get(n)==expected[n] for n in ('stereo_host','model.stream.param','model.opt.bin')),hashes)
            check('process_still_present',values('PROC_END')==[[pid,exe]],dict(pid=pid,exe=exe,cwd=cwd))
            check('live_arguments',len(args)==12 and args[1]=='pl-stream' and args[6:8]==['1920','1080'],args)
            check('live_bind_address',env.get('POSE_BIND_IP')==str(cfg['host']),env.get('POSE_BIND_IP'))
            check('live_roi',all(env.get('POSE_'+s.upper()+'_ROI')==','.join(map(str,rois[s])) for s in rois),env)
            cameras=[]
            for side,physical_id in zip(('left','right'),physical):
                interface=physical_id[3:]
                matches=[v for v in values('VIDEO') if len(v)==5 and v[1]=='0' and v[2].rstrip('/').endswith('/'+interface)]
                if len(matches)==1:
                    node,index,device,driver,name=matches[0]
                    cameras.append('/dev/'+node)
                    bus='usb-xhci-hcd.0.auto-'+interface.split(':')[0].split('-',1)[1]
                    check(side+'_camera',driver.endswith('/u02_uvc1080') and cfg[side+'_bus']==bus,
                          dict(node=node,physical_id=physical_id,sysfs_path=device,driver=driver,name=name,configured_bus=cfg[side+'_bus']))
                else:check(side+'_camera',False,dict(physical_id=physical_id,matches=matches))
            check('live_camera_order',len(args)>4 and len(cameras)==2 and args[4].split(',')==cameras,cameras)
        proof['ok']=all(v['passed'] for v in proof['checks'].values())
        if proof['ok']:
            fresh=dict(bound);fresh['boot_id']=boot;fresh.pop('session_id',None)
            proof['binding']=fresh
        proof['completed_pc_ns']=time.perf_counter_ns()
        return proof

    def install(self):
        binaries=HOME/'board/build'
        manifest=read_json(binaries/'build.json')
        self.command('mkdir -p '+shlex.quote(self.root+'/bin'))
        for name in ('record_stereo','compare_sequence'):
            p=binaries/name
            if digest(p)!=manifest['binaries'][name]['sha256']:raise ValueError('板端工具哈希不符')
            self.progress('上传 '+name)
            remote=self.root+'/bin/'+name
            self.upload(p,remote)
            found=self.command('sha256sum '+shlex.quote(remote)).split()[0]
            if found!=digest(p):raise ValueError('板端文件校验失败')
            self.command('chmod 700 '+shlex.quote(remote))
        models=ROOT.parent/'handoff_1080p_14p24_20261019'
        matches={name:list(models.rglob(name)) for name in ('model.cpu.param','model.stream.param','model.opt.bin')}
        if all(matches.values()):
            self.command('mkdir -p '+shlex.quote(self.config['model_dir']))
            for name,candidates in matches.items():
                self.progress('部署模型 '+name);remote=self.config['model_dir']+'/'+name
                # Refuse replacing a different existing model in a user-selected directory.
                existing=self.command('if test -f '+shlex.quote(remote)+'; then sha256sum '+shlex.quote(remote)+'; fi').split()
                sha=digest(candidates[0])
                if existing and existing[0]!=sha:raise ValueError('目标模型不同，保留原文件：'+remote)
                if not existing:self.upload(candidates[0],remote)
                if self.command('sha256sum '+shlex.quote(remote)).split()[0]!=sha:raise ValueError('模型传输校验失败')
        return '板端录制与对比工具已部署；位流和内核驱动由原运行环境提供'

    def capture(self):
        self.probe();cfg=self.config
        seconds=int(cfg['seconds']);limit=int(cfg['max_mib'])
        if not 1<=seconds<=300 or not 32<=limit<=8192:raise ValueError('录制时长或空间上限不合法')
        interfaces=[]
        for side in ('left','right'):
            match=re.fullmatch(r'usb-xhci-hcd\.0\.auto-([0-9.]+)',cfg[side+'_bus'])
            if not match:raise ValueError('录制仅支持已核验的 xHCI 物理端口；不会自动更换相机驱动')
            interfaces.append('1-'+match[1]+':1.0')
        if len(set(interfaces))!=2:raise ValueError('左右相机必须使用不同物理端口')
        driver_check=self.command('for d in '+shlex.join(interfaces)+'; do printf "DRIVER:%s:%s\\n" "$d" "$(readlink -f /sys/bus/usb/devices/$d/driver)"; done; '
            'for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f" 2>/dev/null); case "$t" in /dev/video*) printf "BUSY:%s\\n" "$t";; esac; done; true')
        if any(not re.search(r'(?m)^DRIVER:'+re.escape(d)+r':[^\r\n]*/u02_uvc1080\r?$',driver_check) for d in interfaces):
            raise ValueError('1080p 录制尚未具备已核验的 u02_uvc1080 驱动绑定；请按正式板端准备流程检查，不会自动换驱动')
        if re.search(r'(?m)^BUSY:',driver_check):raise ValueError('相机正被其他进程占用，请先按原服务流程结束；不会自动停止进程')
        path=new_session('现场双目录制');assets=snapshot_assets(path,cfg)
        self.remote_job=self.root+'/sessions/'+path.name
        state=read_json(path/'session.json');state.update(assets=assets,board_host=cfg['host'],remote_path=self.remote_job,
            camera_bus=dict(left=cfg['left_bus'],right=cfg['right_bus']),roi_xywh=dict(left=cfg['left_roi'],right=cfg['right_roi']),
            capture_protocol=cfg.get('capture_protocol'),capture_driver_preflight=driver_check)
        write_json(path/'session.json',state)
        self.command('mkdir -p '+shlex.quote(self.root+'/sessions')+' && mkdir '+shlex.quote(self.remote_job))
        args=[self.root+'/bin/record_stereo',str(seconds),str(limit),cfg['left_bus'],cfg['right_bus']]
        self.progress(f'板端录制约 {seconds} 秒 · 完成后取回原始数据')
        try:
            with (path/'capture_console.txt').open('wb') as out:
                self.command('cd '+shlex.quote(self.remote_job)+' && '+shlex.join(args),timeout=seconds+45,output=out)
            self.progress('取回原始图像与时间戳…')
            with (path/'capture.tar').open('wb') as out:
                self.command('cd '+shlex.quote(self.remote_job)+' && tar -cf - .',timeout=600,output=out)
            raw=path/'raw';raw.mkdir();safe_extract(path/'capture.tar',raw)
            pairs=import_capture(path,int(cfg['pair_limit_us']))
            self.progress(f'录制已保存，合格配对 {len(pairs)} 对')
            return str(path)
        except Exception as e:
            state.update(state='failed',error=str(e));write_json(path/'session.json',state);raise

    def compare(self, path):
        path=Path(path);cfg=self.config
        state=read_json(path/'session.json')
        rois=state.get('roi_xywh',dict(left=cfg['left_roi'],right=cfg['right_roi']))
        pairs=read_json(path/'pairs.json')['pairs']
        if len(pairs)<2:raise ValueError('至少需要两对通过时间检查的原始图像')
        if (path/'comparison.json').exists():raise ValueError('该会话已有对比结果；请复制会话后重测，避免覆盖')
        self.remote_job=self.root+'/comparisons/'+uuid.uuid4().hex
        self.command('mkdir -p '+shlex.quote(self.remote_job))
        staging=path/'comparison_input';staging.mkdir(exist_ok=False)
        # Hash the exact input bytes, order, source times and ROI configuration.
        identity=[]
        with (staging/'pairs.tsv').open('w',encoding='ascii') as f:
            for i,pair in enumerate(pairs):
                for side in ('left','right'):
                    src=(path/pair[side]['path']).resolve()
                    if not src.is_relative_to(path.resolve()):raise ValueError('图像路径越界')
                    dest=staging/f'{side}_{i}.jpg';shutil.copy2(src,dest)
                    identity.append([dest.name,digest(dest),pair[side]['timestamp_us']])
                f.write(f"left_{i}.jpg\tright_{i}.jpg\t{pair['left']['timestamp_us']}\t{pair['right']['timestamp_us']}\n")
        write_json(staging/'identity.json',dict(images=identity,left_roi=rois['left'],right_roi=rois['right']))
        bundle=path/'comparison_input.tar'
        with tarfile.open(bundle,'w') as tar:
            for p in staging.iterdir():tar.add(p,arcname=p.name)
        self.progress('上传同一组原始双目图像…');self.upload(bundle,self.remote_job+'/input.tar')
        self.command('cd '+shlex.quote(self.remote_job)+' && tar -xf input.tar',timeout=120)
        results={};model_dir=cfg['model_dir']
        model_names=['model.cpu.param','model.stream.param','model.opt.bin']
        model_hashes={name:self.command('sha256sum '+shlex.quote(model_dir+'/'+name)).strip() for name in model_names}
        runtime=self.command('uname -a; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
        for backend in ('cpu','pl-stream'):
            self.progress('测量板载 ARM CPU…' if backend=='cpu' else '测量 ARM + FPGA…')
            param='model.cpu.param' if backend=='cpu' else 'model.stream.param'
            args=[self.root+'/bin/compare_sequence',backend,model_dir+'/'+param,model_dir+'/model.opt.bin','pairs.tsv',backend]
            args += [str(x) for x in rois['left']+rois['right']]
            with (path/(backend+'_console.txt')).open('wb') as out:
                self.command('cd '+shlex.quote(self.remote_job)+' && '+shlex.join(args),timeout=max(120,len(pairs)*8),output=out)
            self.download(self.remote_job+'/'+backend+'.json',path/(backend+'.json'))
            self.download(self.remote_job+'/'+backend+'.ndjson',path/(backend+'.ndjson'))
            results[backend]=read_json(path/(backend+'.json'))
            if any(self.command('sha256sum '+shlex.quote(model_dir+'/'+name)).strip()!=model_hashes[name] for name in model_names):
                raise ValueError('对比期间模型文件变化')
        summary=summarize_comparison(results['cpu'],results['pl-stream'],dict(sha256=digest(staging/'identity.json'),models=model_hashes,runtime=runtime))
        write_json(path/'comparison.json',summary)
        return str(path)


class U02Decoder:
    def __init__(self):
        self.buffer=bytearray();self.assembly=bytearray();self.identity=None;self.next_chunk=0;self.previous=None

    def feed(self, data):
        self.buffer.extend(data);rows=[]
        if len(self.buffer)>65536:raise ValueError('接收缓存超限')
        while len(self.buffer)>=4:
            length,=struct.unpack_from('<I',self.buffer)
            if not 24<length<=1200:raise ValueError('传输包长度错误')
            if len(self.buffer)<length+4:break
            packet=bytes(self.buffer[4:4+length]);del self.buffer[:4+length]
            magic,session,frame,total,index,count=struct.unpack_from('<4sQIIHH',packet)
            if magic!=b'U02P' or not 4<total<=16388 or count!=(total+1175)//1176 or index>=count or len(packet)-24!=min(1176,total-index*1176):
                raise ValueError('传输分块错误')
            if index==0:
                if self.identity is not None:raise ValueError('上一条消息不完整')
                self.identity=(session,frame,total,count);self.next_chunk=0;self.assembly.clear()
            if self.identity!=(session,frame,total,count) or index!=self.next_chunk:raise ValueError('分块顺序错误')
            self.assembly.extend(packet[24:]);self.next_chunk+=1
            if self.next_chunk==count:
                n,=struct.unpack_from('<I',self.assembly)
                if n!=total-4 or len(self.assembly)!=total:raise ValueError('消息长度错误')
                m=json.loads(self.assembly[4:]);self.validate(m,session)
                rows.append(dict(message=m,pc_receive_ns=time.perf_counter_ns(),transport_frame_id=frame))
                self.identity=None
        return rows

    def validate(self,m,session):
        if m.get('schema')!=2 or m.get('message_type')!='paired_coco17' or m.get('session_id')!=session:
            raise ValueError('不支持的骨架消息')
        if m.get('keypoint_order')!='COCO17' or len(m.get('cameras',[]))!=2:raise ValueError('骨架拓扑错误')
        if self.previous:
            old=self.previous
            if (old['boot_id'],old['session_id'])!=(m['boot_id'],m['session_id']):raise ValueError('板端会话变化，请重新开始')
            if m['pair_id']<=old['pair_id']:raise ValueError('重复或乱序帧')
        cams=m['cameras']
        for side,c in zip(('left','right'),cams):
            if c['camera_id']!=side or c['coordinate_space']!='source_image_pixels' or len(c['keypoints'])!=17:
                raise ValueError('相机身份或关键点不符')
            for point in c['keypoints']:
                if len(point)!=4 or type(point[3]) is not bool or not all(math.isfinite(v) for v in point[:3]):raise ValueError('关键点数据错误')
            if self.previous:
                old=self.previous['cameras'][0 if side=='left' else 1]
                if c['capture_timestamp_us']<=old['capture_timestamp_us']:raise ValueError('源时间戳倒退')
        if cams[1]['capture_timestamp_us']-cams[0]['capture_timestamp_us']!=m['pair_delta_us']:raise ValueError('配对时间差错误')
        self.previous=m


class ClockBounds:
    def __init__(self):self.nonce=1;self.pending=None
    def request(self,now):
        if self.pending:return None
        self.pending=(self.nonce,now)
        return struct.pack('<4sIQ',b'U02C',self.nonce,now)
    def accept(self,row):
        probe=row['message'].get('clock_probe',{})
        if not probe.get('nonce'):return
        if self.pending!=(probe['nonce'],probe['pc_send_ns']):raise ValueError('板端时钟探测回应不匹配')
        lower=probe['board_send_ns']-row['pc_receive_ns'];upper=probe['board_receive_ns']-probe['pc_send_ns']
        if lower>upper:raise ValueError('板端时钟区间无效')
        row['latency']=dict(offset_min_ns=lower,offset_max_ns=upper,uncertainty_ms=(upper-lower)/2e6)
        self.pending=None;self.nonce+=1


def run_live(config, output, stop, on_row, status):
    output=Path(output);(output/'pc_receiver').mkdir(exist_ok=True)
    with socket.create_connection((config['host'],int(config['board_port'])),timeout=3) as sock:
        sock.settimeout(.3);decoder=U02Decoder();last=time.monotonic();clock=ClockBounds()
        sock.sendall(clock.request(time.perf_counter_ns()))
        status('已连接板端 · 等待骨架')
        with (output/'pc_receiver/received.ndjson').open('x',encoding='utf-8') as f:
            while not stop.is_set():
                try:data=sock.recv(32768)
                except socket.timeout:
                    if time.monotonic()-last>3:status('骨架断流 · 等待板端')
                    continue
                if not data:raise ConnectionError('板端连接已关闭')
                for row in decoder.feed(data):
                    clock.accept(row);last=time.monotonic();f.write(json.dumps(row)+'\n');f.flush();on_row(row)
                    probe=clock.request(time.perf_counter_ns())
                    if probe:sock.sendall(probe)
