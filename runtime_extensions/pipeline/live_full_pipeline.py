"""START-gated 20-second camera -> FPGA -> PC diagnostic; never formal acceptance."""
import argparse
import hashlib
import json
import re
import socket
import sys
import time
import uuid
import winsound
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs/full_pipeline_exec_20261022/live_runs"
DELIVERY = Path(r"E:\FPGAchina\handoff_1080p_14p24_20261019\handoff_1080p_14p24_20261019")
sys.path.insert(0, str(DELIVERY / "pc"))
from live_receiver import LiveReceiver as Receiver
from backend import Backend
sys.path.insert(0, r"E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000\pc")
from uart_console import Console

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, choices=(30,60,300), default=30)
    args = parser.parse_args()
    seconds = args.seconds
    print(f"{seconds} 秒真实全流程性能测试：双摄→FPGA→在线三维→G1→原生窗口。摄像头不动，不用棋盘。")
    print("站在原位置；短音后先静立 5 秒，再缓慢抬手至肩高并放下，循环到末尾 5 秒静立；长音结束。")
    print("本次保留原始 FPGA 输出；只将时间差不超过 10 毫秒的帧列为合格配对。")
    print("本次实时运行在线三维和 G1，记录延时及帧率；手腕精度与地面尺度仍保留诊断标记。")
    print("本轮 G1 输入加入轻度去抖，会增加少量动作响应滞后；先观察静立时脚部是否稳定。")
    print("START 和倒计时后先检查相机同步，最多三次；短音确认有效三维已驱动 G1，再开始动作。同步不合格会退出，不放宽 10 毫秒门限。")
    if input("准备好后输入 START；其他输入取消：").strip() != "START":
        print("未开启摄像头。")
        return
    stage = json.loads(Path('C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\outputs\\board_resume_20261023_112300\\live_stage.json').read_text())
    network_proof = json.loads((ROOT.parents[1] / "outputs/full_chain_20261022/board_verification.json").read_text())
    assert network_proof["passed"] and network_proof["source_build_host_preflight_passed"]
    launch_proof = json.loads(Path('C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\outputs\\board_resume_20261023_112300\\preflight.json').read_text())
    assert launch_proof["complete"] and launch_proof["cleanup_verified"] and launch_proof["boot_id"] == stage["boot_id"]
    tested_stage = json.loads((ROOT.parents[1] / "outputs/full_chain_20261022/board/stage.json").read_text())
    storage = json.loads(Path('C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\outputs\\board_resume_20261023_112300\\snapshot_storage_verification.json').read_text())
    assert storage["passed"] and stage["synthetic_storage_test_passed"]
    for name in ("model.opt.bin", "model.stream.param", "u02_stream_pipeline.ko"):
        assert tested_stage["files"][name] == stage["files"][name]
    OUT.mkdir(exist_ok=True)
    assert stage["complete"]
    proof = json.loads(Path('C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\outputs\\board_resume_20261023_112300\\run.json').read_text())
    assert proof["complete"] and not proof["dma_ambiguous"] and proof["boot_id"] == stage["boot_id"]
    remote = stage["remote"]
    run_id = "live_" + uuid.uuid4().hex
    output = OUT / run_id
    output.mkdir()
    state = dict(complete=False, run_id=run_id, boot_id=stage["boot_id"], remote=remote,
                 explicit_START=True, camera_opened=False, module_loaded=False, camera_module_loaded=False,
                 module_uncertain=False, process_uncertain=False, bindings_changed=[],
                 wire_pair_limit_us=16667, qualification_pair_limit_us=10000,
                 input_profile="1080p_per_view_roi_20261022",
                 roi_xywh=dict(left=[780,280,510,680],right=[620,260,510,680]),
                 source="live FPGA ABI9, explicit per-view ROI candidate", full_3D_G1_chain_validated=False,
                 host_sha256=stage["files"]["stereo_host"], host_change="bounded camera-only 1080p startup qualification; model arithmetic unchanged")
    def save():
        (output / "session.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    save()
    receiver = None
    backend = None
    with (output / "uart.log").open("wb") as log:
        c = Console("COM22", log)
        original_send = c.send
        def paced(data):
            for i in range(0, len(data), 24):
                original_send(data[i:i+24]); time.sleep(.004)
        c.send = paced
        def checked(command, timeout=30):
            assert len(command.encode()) < 700
            rc, text = c.run(command, timeout)
            if rc:
                raise RuntimeError(text)
            return text
        def idle():
            text = checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*|/dev/u02*) printf "OPEN:%s\\n" "$t";; esac; done; true')
            assert not re.search(r"(?m)^OPEN:", text)
        try:
            c.sync()
            assert re.search(r"(?m)^" + stage['boot_id'] + r"\r?$", checked("cat /proc/sys/kernel/random/boot_id")), "Board rebooted: prepare again"
            assert not re.search(r"(?m)^u02_", checked("cat /proc/modules"))
            idle()
            for name, digest in stage["files"].items():
                assert re.search(r"(?m)^" + digest + r"\s", checked("sha256sum " + remote + "/" + name))
            for dev in ("1-1.4:1.0", "1-1.2:1.0"):
                assert "/drivers/uvcvideo" in checked("readlink /sys/bus/usb/devices/" + dev + "/driver")
            checked("! awk '$2 ~ /:46A4$/ && $4 == \"0A\" {found=1} END {exit !found}' /proc/net/tcp")
            selected = json.loads((OUT.parent/'selected_backend.json').read_text())
            assert selected['prepared_for_live_run']
            for entry in selected['runtime_files']:
                assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'], 'Live candidate code changed; rerun preflight'
            binding = json.loads((ROOT.parents[1]/'outputs/completion_20261022/live_runs/live_6d871440685641b1be52dc54814ac527/audit/binding.json').read_text())
            binding.pop('session_id', None)
            binding['boot_id'] = stage['boot_id']
            binding['scope'] = 'Live performance diagnosis; wrist and absolute floor accuracy unaccepted'
            backend = Backend(output, binding, selected['lag_ms'],selected.get('smooth_ms',40),selected.get('quality_guard',True))
            scope = dict(mode='live', pc_receipts_reissued=False, lag_ms=selected['lag_ms'],
                         cameras_opened_after_explicit_START=True, source_timestamps_preserved=True,
                         source_alignment='causal warmup minimum receipt offset, then frozen',
                         future_reference='observed samples only; bounded source progression',
                         partial_body_support='up to six of twelve body joints held for at most 100ms; labeled separately',
                         quality_guard=selected.get('quality_guard',True),
                         recovery_anchor='actual paused physical pose; no state reset',
                         G1_input_conditioning=dict(type='causal_exponential',tau_s=selected.get('smooth_ms',40)/1000,raw_3d_observations_changed=False,
                             scope='G1 reference only; motion-response lag is additional to source-reference timestamp latency.'))
            (output/'scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
            print(f"电脑端已准备好，基准缓冲 {selected['lag_ms']} 毫秒，尚未正式验收；接下来倒计时。",flush=True)
            for second in range(10, 0, -1):
                print("倒计时", second, flush=True); time.sleep(1)
            state["module_uncertain"] = True; save()
            checked(f"insmod {remote}/u02_stream_pipeline.ko bridge_checked=1 cached_dma=1 poll_spin_us=1000")
            state["module_loaded"] = True
            checked(f"insmod {remote}/u02_uvc1080.ko payload_limit=2048 hwtimestamps=1", 90)
            state["camera_module_loaded"] = True
            state["module_uncertain"] = False; save()
            for dev in ("1-1.4:1.0", "1-1.2:1.0"):
                checked(f"printf '%s' {dev} > /sys/bus/usb/drivers/uvcvideo/unbind", 90)
                state["bindings_changed"].append(dev); save()
                checked(f"printf '%s' {dev} > /sys/bus/usb/drivers/u02_uvc1080/bind", 90)
            text = checked('for v in /sys/class/video4linux/video*; do test "$(cat "$v/index")" = 0 || continue; printf "NODE:%s:" "${v##*/}"; readlink -f "$v/device"; done')
            nodes = []
            for port in ("1-1.4", "1-1.2"):
                found = re.findall(r"(?m)^NODE:(video[0-9]+):[^\r\n]*/" + re.escape(port) + r"/", text)
                assert len(found) == 1, text
                nodes.append("/dev/" + found[0])
            assert set(nodes) == {"/dev/video0", "/dev/video2"}, nodes
            state["nodes"] = nodes
            checked(f"mkdir {remote}/{run_id}")
            receiver = Receiver(output / "pc_receiver", host="192.168.137.2", expected_source_size=(1920,1080), expected_pair_limit_us=16667, backend=backend)
            receiver.start()
            command = f"cd {remote}/{run_id} && (POSE_BIND_IP=192.168.137.2 POSE_LEFT_ROI=780,280,510,680 POSE_RIGHT_ROI=620,260,510,680 POSE_DEBUG_SNAPSHOTS=1 POSE_RUN_SECONDS={seconds} POSE_INFLIGHT_PAIRS=2 POSE_THREADS=1 POSE_PAIR_NETWORK=1 POSE_PAIR_WAIT_CLIENT_MS=10000 POSE_QUANT_VECTOR=1 GOMP_SPINCOUNT=300000 OMP_WAIT_POLICY=PASSIVE ../stereo_host pl-stream ../model.stream.param ../model.opt.bin {','.join(nodes)} result.json 1920 1080 555 0 810 1080; echo $? > exit) > run.log 2>&1 < /dev/null &"
            state.update(camera_opened=True, process_uncertain=True); save()
            checked(command)
            begun = time.monotonic()
            sounded = False
            while time.monotonic() - begun < seconds+90:
                if (output/'states.ndjson').exists() and (output/'states.ndjson').stat().st_size and not sounded:
                    state["motion_cue_pc_ns"] = time.perf_counter_ns(); save(); winsound.Beep(900,250); print("短音：有效三维已驱动 G1。先静立 5 秒，再缓慢抬放手。", flush=True); sounded = True
                rc, _ = c.run(f"test -s {remote}/{run_id}/exit", 5)
                if rc == 0:
                    status = checked(f"cat {remote}/{run_id}/exit")
                    state["exit_code"] = int(re.search(r"(?m)^(\d+)\r?$", status).group(1))
                    idle(); state["process_uncertain"] = False; save()
                    break
                time.sleep(.5)
            assert not state["process_uncertain"], "Host ownership unresolved; no forced cleanup"
            state["board_log"] = checked(f"cat {remote}/{run_id}/run.log")
            assert state["exit_code"] == 0, state["board_log"]
            winsound.Beep(1200,800); print("长音：动作结束，正在回传。", flush=True)
            receiver.finish()
            state['forwarded_pairs'] = receiver.forwarded
            state['relay_error'] = receiver.relay_error
            receiver = None
            try:
                backend.finish()
                if state['relay_error']:
                    raise RuntimeError('Input relay stopped before capture ended: '+state['relay_error'])
                state['backend_complete'] = True
                state['live_3D_G1_path_exercised'] = True
            except Exception as error:
                state['backend_complete'] = False
                state['live_3D_G1_path_exercised'] = False
                state['backend_error'] = repr(error)
            finally:
                backend.close()
                backend = None
            checked(f"cd {remote} && tar -czf {run_id}.tgz {run_id}")
            expected = re.search(r"(?m)^([a-f0-9]{64})\s", checked(f"sha256sum {remote}/{run_id}.tgz")).group(1)
            checked(f"timeout 30 nc -l -p 18082 < {remote}/{run_id}.tgz > {remote}/download.log 2>&1 &"); time.sleep(.5)
            archive = output / "board_results.tgz"
            with socket.create_connection(("192.168.137.2",18082), timeout=30, source_address=("192.168.137.1",0)) as sock, archive.open("xb") as stream:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk: break
                    stream.write(chunk)
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
            state["board_results_sha256"] = expected
            state["complete"] = True
        except Exception as error:
            state["error"] = repr(error)
            raise
        finally:
            if receiver:
                try:
                    receiver.finish()
                except Exception as error:
                    state.update(complete=False, receiver_error=repr(error))
            try:
                if not state["process_uncertain"] and not state["module_uncertain"]:
                    idle()
                    for dev in reversed(state["bindings_changed"]):
                        rc, binding = c.run("readlink /sys/bus/usb/devices/" + dev + "/driver",30)
                        assert rc == 1 or "/drivers/u02_uvc1080" in binding, "Unexpected camera driver; leave untouched"
                        if "/drivers/u02_uvc1080" in binding:
                            checked(f"printf '%s' {dev} > /sys/bus/usb/drivers/u02_uvc1080/unbind",90)
                        checked(f"printf '%s' {dev} > /sys/bus/usb/drivers/uvcvideo/bind",90)
                    if state["camera_module_loaded"]: checked("rmmod u02_uvc1080",90)
                    if state["module_loaded"]: checked("rmmod u02_stream_pipeline",90)
                    for dev in state["bindings_changed"]:
                        assert "/drivers/uvcvideo" in checked("readlink /sys/bus/usb/devices/" + dev + "/driver")
                    idle()
                    assert not re.search(r"(?m)^u02_", checked("cat /proc/modules"))
                    state["cleanup_verified"] = True
                else:
                    state["complete"] = False
                    state["cleanup_deferred"] = "Uncertain ownership; no forced unload"
            except Exception as error:
                state.update(complete=False, cleanup_error=repr(error))
            if backend:
                try: backend.close()
                except Exception as error: state['backend_cleanup_error'] = repr(error)
            save(); c.close()
    if state["complete"]:
        print("原始数据已采集并回传；全流程是否有效以下方核验为准。", output, flush=True)
        sys.path.insert(0,str(ROOT.parents[1]/'work/completion_20261022'))
        from audit_snapshots import audit
        audit(output)
        if (output/'backend.json').exists():
            from metrics import analyze
            report = analyze(output)
            state['performance_checks_passed'] = report['headline']['live_full_chain_passed']
            save()
            print(json.dumps(report['headline'],ensure_ascii=False,indent=2),flush=True)
        print("全流程证据和核验已保存；未达目标的项目将继续优化。", flush=True)

if __name__ == "__main__":
    main()
