"""User-gated real-board acceptance; prepare never accesses a camera."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from .geometry import Stereo
from .board_adapter import convert
from .pipeline import replay
from .online import OnlineSession

HERE=Path(__file__).resolve().parent


def upload(demo,c,args,local,remote):
    data=Path(local).read_bytes();digest=hashlib.sha256(data).hexdigest()
    token=uuid.uuid4().hex;ready='RX_'+token;done='DONE_'+token
    c.send((f'timeout 90 nc -l -p 18088 > {remote} < /dev/null & p=$!; sleep 1; echo {ready}; wait $p; echo {done}:$?\n').encode())
    c.wait(ready.encode()+rb'\r?\n',10)
    with demo.connection(args,18088) as sock:
        sock.sendall(data);sock.shutdown(socket.SHUT_WR)
    _,m=c.wait(done.encode()+rb':([0-9]+)\r?\n',95)
    if int(m[1]) or digest not in demo.checked(c,'sha256sum '+remote):
        raise ValueError('Uploaded executable hash mismatch.')
    return digest


def download_archive(demo,c,args,remote,local):
    expected=re.search(r'(?m)^([a-f0-9]{64})\s',demo.checked(c,'sha256sum '+remote))[1]
    token=uuid.uuid4().hex;ready='TX_'+token;done='DONE_'+token
    c.send((f'timeout 90 nc -l -p 18083 < {remote} > /dev/null & p=$!; sleep 1; echo {ready}; wait $p; echo {done}:$?\n').encode())
    c.wait(ready.encode()+rb'\r*\n',10)
    with demo.connection(args,18083) as sock,Path(local).open('xb') as stream:
        sock.shutdown(socket.SHUT_WR)
        while True:
            block=sock.recv(1024*1024)
            if not block:break
            stream.write(block)
    _,match=c.wait(done.encode()+rb':([0-9]+)\r*\n',95)
    if int(match[1]) or hashlib.sha256(Path(local).read_bytes()).hexdigest()!=expected:
        raise ValueError('Saved camera archive hash mismatch.')
    return expected


def release_saved_images(demo,c,remote,name,archive_path,expected_sha,pair_ids):
    """Remove only this run's redundant RAM copies after checking its local archive."""
    import cv2
    import numpy as np
    if not re.fullmatch(r'/tmp/u02_demo_[a-f0-9]{32}',remote) or not re.fullmatch(r'v4_[0-9]{8}_[0-9]{6}',name):
        raise ValueError('Unexpected cleanup path.')
    if hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('Local archive changed; preserve board copies.')
    images={};total_bytes=0;unreceived_pair_ids=set()
    with tarfile.open(archive_path) as archive:
        result=json.loads(archive.extractfile(name+'.json').read())
        for member in archive:
            if not member.isfile() or not member.name.startswith(name+'.json.frames/'):continue
            match=re.fullmatch(re.escape(name)+r'\.json\.frames/(left|right)_([0-9]+)\.jpg',member.name)
            if not match or member.name in images:
                raise ValueError('Unexpected saved image; preserve board copies.')
            # JPEG logging precedes network delivery, which can drop complete pairs.
            if int(match[2]) not in pair_ids:unreceived_pair_ids.add(int(match[2]))
            raw=archive.extractfile(member).read()
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
            if image is None or image.shape!=(720,1280,3):raise ValueError('Invalid saved JPEG.')
            images[member.name]=hashlib.sha256(raw).hexdigest();total_bytes+=len(raw)
    if not images or len(images)!=result['image_log_frames'] or total_bytes!=result['image_log_bytes']:
        raise ValueError('Saved image counts differ; preserve board copies.')
    record=dict(archive_sha256=expected_sha,verified_images=len(images),image_bytes=total_bytes,
                partial_image_log=len(images)<result['completed_pairs']*2,images=images,
                saved_but_unreceived_pair_ids=sorted(unreceived_pair_ids),board_copies_removed=False)
    record_path=Path(archive_path).parent/'image_archive_verification.json'
    record_path.write_text(json.dumps(record,indent=2))
    frames=remote+'/'+name+'.json.frames';archive_remote=remote+'/'+name+'.tar'
    resolved=demo.checked(c,'readlink -f '+frames)
    if not re.search(r'(?m)^'+re.escape(frames)+r'\r*$',resolved):raise ValueError('Cleanup target resolved elsewhere.')
    if expected_sha not in demo.checked(c,'sha256sum '+archive_remote):raise ValueError('Remote archive changed.')
    demo.checked(c,'rm -r '+frames)
    demo.checked(c,'rm '+archive_remote)
    record['board_copies_removed']=True
    record_path.write_text(json.dumps(record,indent=2))
    return record


def run(args):
    ipaddress.IPv4Address(args.board)
    if not 5<=args.seconds<=300:raise ValueError('Duration must be 5..300 seconds.')
    package=Path(args.package).resolve()
    sys.path.insert(0,str(package/'pc'))
    import demo
    from uart_console import Console
    from receive_paired_pose import Receiver
    demo.verify()
    stereo=Stereo(args.calibration)
    ready_path=HERE/'runs/20261017_realtime_v4/live_ready.json'
    output=HERE/'runs'/('live_v4_'+time.strftime('%Y%m%d_%H%M%S')+'_'+args.action)
    output.mkdir(parents=True)
    if args.mode=='run':
        state=json.loads(ready_path.read_text())
        if state['calibration_hash']!=stereo.hash or not state['complete']:
            raise ValueError('Prepare against the current calibration first.')
        print('这是最终实机验收，不是重新采集训练集。无需棋盘。',flush=True)
        print('站在之前能看见鞋的靠后中间位置；身体朝向两摄中间，双脚原地不动。',flush=True)
        actions=dict(arms='站稳3秒；双臂缓慢抬到胸前再放下，重复，保持双脚原地。',
                     range='站稳3秒；双臂缓慢向左右伸展到肩高以下，再收回，身体小幅左右转动。',
                     squat='站稳3秒；轻蹲、站起，缓慢重复，再左右轻微移动重心。',
                     stability='保持在共同视野内，前后各做一次缓慢抬臂和轻蹲；中间正常站立，无需持续动作。')
        print('开始声后：'+actions[args.action]+f' 本段约{args.seconds}秒。',flush=True)
        print('持续到结束声。输入START后有10秒就位；其他输入不拍摄。',flush=True)
        if input('准备好请输入 START：').strip()!='START':
            print('没有开始拍摄。');return
        import winsound
        for remaining in range(10,0,-1):
            print('就位倒计时',remaining,flush=True);time.sleep(1)
        winsound.Beep(900,250)
    with (output/'uart.log').open('wb') as log:
        c=Console(args.serial,log)
        receiver=None
        online=None
        viewer=None
        try:
            c.sync();boot=demo.preflight(c)
            deployment=json.loads((package/'logs/deployment.json').read_text())
            if deployment['boot_id']!=boot:
                raise ValueError('Board was rebooted; restore and deploy before acceptance.')
            remote=deployment['remote']
            if not re.fullmatch(r'/tmp/u02_demo_[a-f0-9]{32}',remote):
                raise ValueError('Unexpected deployment directory.')
            if args.mode=='prepare':
                if not args.binary:raise ValueError('Provide tested candidate binary.')
                demo.program(args,output)
                if demo.preflight(c)!=boot:raise ValueError('Boot changed during prepare.')
                digest=upload(demo,c,args,args.binary,remote+'/pose3d_live_dark')
                demo.checked(c,'chmod 700 '+remote+'/pose3d_live_dark')
                state=dict(complete=True,boot_id=boot,remote=remote,binary_sha256=digest,
                    calibration_hash=stereo.hash,model_sha256=hashlib.sha256((package/'board/model.opt.bin').read_bytes()).hexdigest(),
                    physical_geometry_user_confirmed_unchanged=True,heatmap_decode='dark',roi_xywh=[435,180,405,540],
                    adaptive_roi=args.adaptive_roi,threads=args.threads,
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
                ready_path.parent.mkdir(parents=True,exist_ok=True)
                ready_path.write_text(json.dumps(state,indent=2))
                print('READY_NO_CAMERA_OPENED',ready_path,flush=True);return
            if state['boot_id']!=boot or state['remote']!=remote:
                raise ValueError('Prepared board session no longer matches.')
            if state['checkpoint_sha256']!=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest():
                raise ValueError('Prepared checkpoint changed.')
            disk=demo.checked(c,'df -k /tmp')
            disk_rows=[line.split() for line in disk.splitlines() if line.startswith('tmpfs')]
            required_mib=576 if args.seconds>30 else 128
            if not disk_rows or int(disk_rows[-1][3])<required_mib*1024:
                raise ValueError(f'Need {required_mib} MiB free /tmp for this diagnostic; preserve other projects.')
            if state['binary_sha256'] not in demo.checked(c,'sha256sum '+remote+'/pose3d_live_dark'):
                raise ValueError('Candidate binary changed.')
            demo.staged(c)
            opened=demo.checked(c,'for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*) echo VIDEO_OPEN;; esac; done; true')
            if re.search(r'(?m)^VIDEO_OPEN\r*$',opened):raise ValueError('Close another camera application first.')
            for device in demo.DEVICES:
                if '/drivers/uvcvideo' not in demo.checked(c,'readlink /sys/bus/usb/devices/'+device+'/driver'):
                    raise ValueError('Unexpected camera driver binding.')
            demo.checked(c,'insmod '+remote+'/u02_uvcvideo.ko payload_limit=2048')
            # Free both devices before binding left first; reboot can reverse node indices.
            for device in demo.DEVICES:
                demo.checked(c,"printf '%s' "+device+' > /sys/bus/usb/drivers/uvcvideo/unbind')
            for device in demo.DEVICES:
                demo.checked(c,"printf '%s' "+device+' > /sys/bus/usb/drivers/u02_uvcvideo/bind')
            for node,physical in [('video0','1-1.4'),('video2','1-1.2')]:
                if '/'+physical+'/' not in demo.checked(c,'readlink -f /sys/class/video4linux/'+node+'/device'):
                    raise ValueError('Camera order mismatch; do not run inference.')
            demo.checked(c,'insmod '+remote+'/u02_stream_fp32.ko bridge_checked=1 cached_dma=1')
            binding=dict(schema='board_session_binding/1.0',calibration_hash=stereo.hash,
                camera_physical_ids=stereo.calibration['camera_physical_ids'],model_sha256=state['model_sha256'],
                boot_id=boot,heatmap_decode='dark',host_binary_sha256=state['binary_sha256'],
                physical_geometry_user_confirmed_unchanged=True,live_acquisition_validated=True,
                roi_xywh=state['roi_xywh'],adaptive_roi=state['adaptive_roi'])
            online=OnlineSession(args.calibration,args.checkpoint,binding,output/'online')
            class ObservedRows(list):
                def append(self,record):
                    super().append(record)
                    online.submit(record)
            receiver=Receiver(output/'receiver',args.board,18084)
            receiver.rows=ObservedRows();receiver.start()
            if args.show:
                viewer=subprocess.Popen([sys.executable,'-X','utf8','-m','pose3d_train.online_view','--source',str(output/'online/latest.json')],
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            name='v4_'+time.strftime('%Y%m%d_%H%M%S')
            command=(f'cd {remote} && OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 POSE_THREADS={state["threads"]} POSE_WARMUP=0 '
                     f'POSE_IMAGE_INPUT=1 POSE_QUANT_VECTOR=1 POSE_DARK_DECODE=1 POSE_ADAPTIVE_ROI={int(state["adaptive_roi"])} POSE_SAVE_FRAMES=1 POSE_BIND_IP={args.board} POSE_RUN_SECONDS={args.seconds} '
                     './pose3d_live_dark pl-stream model.stream.param model.opt.bin /dev/video0,/dev/video2 '
                     f'{name}.json 1280 720 435 180 405 540 >{name}.log 2>&1')
            try:
                demo.checked(c,command,args.seconds+50)
            except RuntimeError:
                c.download(remote+'/'+name+'.log',output/'board_error.log')
                failure=(output/'board_error.log').read_text(errors='replace')
                print(failure,flush=True)
                if 'STEREO_START_ALIGNMENT_FAILED before inference' in failure:
                    # This specific failure occurs before any image forward or DMA submission.
                    opened=demo.checked(c,'for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*|/dev/u02*) echo OPEN;; esac; done; true')
                    if not re.search(r'(?m)^OPEN\r*$',opened):
                        demo.checked(c,'rmmod u02_stream_fp32');demo.restore(c)
                raise
            summary=receiver.finish();receiver=None
            demo.checked(c,'rmmod u02_stream_fp32');demo.restore(c)
            finished_online=online;online=None
            online_report=finished_online.finish()
            import winsound
            winsound.Beep(1200,300);winsound.Beep(1200,300)
            print('本段已结束，可回到电脑前。正在检查三维骨架与BVH。',flush=True)
            # JPEGs are already compressed; avoid a slow gzip pass on the board.
            demo.checked(c,f'cd {remote} && tar -cf {name}.tar {name}.json.frames {name}.json {name}.json.jpg {name}.log',60)
            archive_sha=download_archive(demo,c,args,remote+'/'+name+'.tar',output/'selected_camera_frames.tar')
            with tarfile.open(output/'selected_camera_frames.tar') as archive:
                for remote_name,local_name in [(name+'.json','board_result.json'),(name+'.json.jpg','last_view.jpg'),(name+'.log','board_log.txt')]:
                    member=archive.getmember(remote_name)
                    if not member.isfile():raise ValueError('Invalid board result archive.')
                    (output/local_name).write_bytes(archive.extractfile(member).read())
            rows=[json.loads(s) for s in (output/'receiver/received.ndjson').read_text().splitlines()]
            if not rows:raise ValueError('No paired poses received.')
            release_saved_images(demo,c,remote,name,output/'selected_camera_frames.tar',archive_sha,
                                 {row['message']['pair_id'] for row in rows})
            message=rows[0]['message']
            binding.update(session_id=message['session_id'],selected_camera_archive_sha256=archive_sha)
            bound=output/'session_binding.json';bound.write_text(json.dumps(binding,indent=2))
            pose=output/'pose2d.jsonl'
            convert(output/'receiver/received.ndjson',args.calibration,bound,pose)
            report=replay(pose,args.calibration,args.checkpoint,output/'replay',max_sample_interval_s=.25,single_frame_gap_s=.35)
            duration=(rows[-1]['message']['cameras'][0]['capture_timestamp_us']-rows[0]['message']['cameras'][0]['capture_timestamp_us'])/1e6
            longest=max((f['frames']/30/duration for f in report.get('bvh_files',[])),default=0.)
            checks=dict(body_coverage=report['body_coverage_after']>=.98,bvh_coverage=report.get('bvh_coverage',0)>=.95,
                        longest_continuous_segment=longest>=.90,rate=summary['receive_pairs_per_s']>=6.0,
                        no_online_queue_drop=online_report['dropped_queue_packets']==0)
            (output/'acceptance.json').write_text(json.dumps(dict(action=args.action,checks=checks,quantitative_pass=all(checks.values()),
                longest_segment_ratio=longest,visual_review_required=True,geometry_accuracy_independently_accepted=False),indent=2))
            print('LIVE_ACCEPTANCE_SAVED',output,flush=True)
            print('接收帧对：',summary['messages'],'；BVH时间覆盖率：',report.get('bvh_coverage',0),flush=True)
        finally:
            try:
                if receiver:receiver.finish()
            finally:
                try:
                    if online:online.finish()
                finally:c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--package',default=r'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000')
    parser.add_argument('--serial',default='COM22');parser.add_argument('--board',default='192.168.137.2')
    parser.add_argument('--vivado',default=r'E:\FPGAchina\Xilinx\Vivado\2020.2\bin\vivado.bat')
    parser.add_argument('--binary')
    parser.add_argument('--seconds',type=int,default=25)
    parser.add_argument('--action',choices=['arms','range','squat','stability'],default='arms')
    parser.add_argument('--adaptive-roi',action='store_true')
    parser.add_argument('--threads',type=int,choices=[1,2,3,4],default=3)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--calibration',default=str(HERE/'runs/20261017_current_geometry/calibration_printed_plane.json'))
    parser.add_argument('--checkpoint',default=str(HERE/'runs/20261017_mlp_current_v4_conflict/checkpoint.pt'))
    run(parser.parse_args())
