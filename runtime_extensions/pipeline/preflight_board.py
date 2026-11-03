"""Verify the patched listener using nonexistent camera paths; never capture."""
import json
import re
import socket
import sys
import time
from pathlib import Path
sys.path.insert(0, r"E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000\pc")
from uart_console import Console
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/full_pipeline_exec_20261022"
stage = json.loads((ROOT / "outputs/completion_20261022/live_stage.json").read_text())
proof = json.loads((ROOT / "outputs/full_chain_20261022/board/run.json").read_text())
assert stage["complete"] and proof["complete"] and not proof["dma_ambiguous"] and stage["boot_id"] == proof["boot_id"]
remote = stage["remote"]
state = dict(complete=False, camera_opened=False, boot_id=stage["boot_id"], remote=remote, host_active=False, driver_loaded=False)
assert not (OUT / "preflight.json").exists()
def save():
    (OUT / "preflight.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
with (OUT / "preflight_uart.log").open("wb") as log:
    c = Console("COM22", log)
    original_send = c.send
    def paced(data):
        for i in range(0, len(data), 24):
            original_send(data[i:i+24]); time.sleep(.004)
    c.send = paced
    def checked(command):
        rc, output = c.run(command, 30)
        if rc: raise RuntimeError(output)
        return output
    def idle():
        output = checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*|/dev/u02*) printf "OPEN:%s\\n" "$t";; esac; done; true')
        assert not re.search(r"(?m)^OPEN:", output)
    try:
        c.sync()
        assert re.search(r"(?m)^" + stage["boot_id"] + r"\r?$", checked("cat /proc/sys/kernel/random/boot_id"))
        assert not re.search(r"(?m)^u02_", checked("cat /proc/modules"))
        idle()
        for name, digest in stage["files"].items():
            assert re.search(r"(?m)^" + digest + r"\s", checked("sha256sum " + remote + "/" + name))
        checked("! awk '$2 ~ /:46A4$/ && $4 == \"0A\" {found=1} END {exit !found}' /proc/net/tcp")
        checked(f"test ! -e {remote}/NO_CAMERA_LEFT && test ! -e {remote}/NO_CAMERA_RIGHT")
        checked(f"insmod {remote}/u02_stream_pipeline.ko bridge_checked=1 cached_dma=1 poll_spin_us=1000")
        state["driver_loaded"] = True; save()
        command = f"cd {remote} && (POSE_BIND_IP=192.168.137.2 POSE_LEFT_ROI=780,280,510,680 POSE_RIGHT_ROI=620,260,510,680 POSE_DEBUG_SNAPSHOTS=1 POSE_RUN_SECONDS=2 POSE_PAIR_NETWORK=1 POSE_PAIR_WAIT_CLIENT_MS=10000 ./stereo_host pl-stream model.stream.param model.opt.bin {remote}/NO_CAMERA_LEFT,{remote}/NO_CAMERA_RIGHT fullflowcheck_20261022.json 1920 1080 555 0 810 1080; echo $? > fullflowcheck_20261022.exit) > fullflowcheck_20261022.log 2>&1 < /dev/null &"
        assert len(command.encode()) < 700
        state["host_active"] = True; save()
        checked(command)
        for attempt in range(30):
            try:
                with socket.create_connection(("192.168.137.2",18084), timeout=.5, source_address=("192.168.137.1",0)) as sock:
                    state["tcp_connected"] = True
                    sock.settimeout(5)
                    state["socket_closed_by_host"] = sock.recv(16) == b""
                break
            except (ConnectionRefusedError, TimeoutError):
                time.sleep(.1)
        for attempt in range(30):
            rc, _ = c.run(f"test -s {remote}/fullflowcheck_20261022.exit",5)
            if rc == 0:
                state["host_active"] = False
                break
            time.sleep(.2)
        assert not state["host_active"]
        state["exit"] = checked(f"cat {remote}/fullflowcheck_20261022.exit")
        state["log"] = checked(f"cat {remote}/fullflowcheck_20261022.log")
        assert state.get("tcp_connected") and re.search(r"(?m)^1\r?$",state["exit"])
        assert "Cannot listen on board pose display endpoint" not in state["log"]
        state["expected_missing_camera_error"] = "1080p candidate requires two resolved capture nodes and 1920x1080 MJPEG" in state["log"]
        assert state["expected_missing_camera_error"], state["log"]
        state["complete"] = True
    except Exception as error:
        state["error"] = repr(error)
        raise
    finally:
        try:
            if state["driver_loaded"] and not state["host_active"]:
                idle()
                assert re.search(r"(?m)^u02_stream_pipeline\s+\d+\s+0\s", checked("cat /proc/modules"))
                checked("rmmod u02_stream_pipeline")
                state["driver_loaded"] = False
                state["cleanup_verified"] = not re.search(r"(?m)^u02_", checked("cat /proc/modules"))
        finally:
            save(); c.close()
print(json.dumps(state, indent=2))
