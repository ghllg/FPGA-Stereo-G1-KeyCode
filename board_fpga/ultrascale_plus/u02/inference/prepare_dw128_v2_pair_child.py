"""Prepare the existing paired-network child for one externally owned bit.

The child never programs PL or writes clock settings. An outer owner must
establish /9 and safe ownership, then restore the protected original runtime.
"""
import argparse
import ast
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from prepare_dw128_v2_board_ab import U02, sha, verify_frozen

PAIR_HOST = "milestones/matched_cpu1_and_1080_entry_verified_20261016/cpu_current/tested_pose_pair_network"
HOST_SHA = "94806cbd5c4c8d15db26f3392e33e28073d2c75852d082cadcb551b126a3f65c"


def owner_is_live(pid):
    # Query only. Never terminate, suspend or send commands to the owner.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def verify_owner_handoff(record, bit_sha, child, restored_record):
    assert record is not None, "Outer hardware owner record is required"
    record = Path(record).resolve()
    state = json.loads((record / "state.json").read_text())
    assert state["schema"] == "u02_dw128_v2_ab_owner_v1"
    assert state["terminal"] is False and state["phase"] == "child_running"
    assert state["active_bit_sha256"] == bit_sha and state["active_pl_clock_divisor"] == 9
    assert state["child_script_sha256"] == sha(Path(child))
    assert state["restored_record_sha256"] == sha(Path(restored_record) / "state.json")
    assert state["startup_files_modified"] is False
    assert not state["command_uncertain"] and not state["gate_pending"]
    assert state["dma_ambiguous"] is True  # Parent has delegated DMA ownership.
    assert owner_is_live(state["owner_pid"]), "Outer owner is no longer running"
    return state


def main_helpers(text):
    main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return {n.name: n for n in main.body if isinstance(n, ast.FunctionDef)}


def generate(source, project, reports, bit_sha):
    original = source

    def once(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    once("from retest_original_after_region import",
         "import sys\nsys.path.insert(0," + repr(str(U02 / "inference")) + ")\nfrom retest_original_after_region import")
    once("    args=ap.parse_args()",
         "    ap.add_argument('--owner-record',type=Path)\n"
         "    args=ap.parse_args()\n"
         "    assert not args.range_cache_host, 'Fixed same-Host hardware A/B only'\n"
         "    if not args.prepare_only:\n"
         "        from prepare_dw128_v2_pair_child import verify_owner_handoff\n"
         "        verify_owner_handoff(args.owner_record," + repr(bit_sha) +
         ",Path(__file__),args.restored_record)")
    once("    host=U02/'milestones/dual_context_host_cpu_verified_20261015/prepared_pose_pair_network'",
         "    host=U02/" + repr(PAIR_HOST))
    once("assert sha(host)=='a424bd85185e84e554e169c5b69d6a5c442abe4cd1a60b98844bf372da354713'",
         "assert sha(host)==" + repr(HOST_SHA))
    once("U02/'inference/stream_ingress_project',U02/'reports/stream_ingress_implementation'",
         "Path(" + repr(str(project)) + "),Path(" + repr(str(reports)) + ")")
    assert source.count("^0x01010[Bb]02") == 2
    source = source.replace("^0x01010[Bb]02", "^0x01010902")
    once("formal_restored=False", "driver_restored=False")
    assert source.count("['formal_restored']") == 2
    source = source.replace("['formal_restored']", "['driver_restored']")
    once("original restored bit and /11 clock", "externally owned selected bit and /9 clock; child restores driver only")
    once("cases=[('pl2_serial',2,'serial','pl-stream'),('pl1_parallel',1,'parallel','pl-stream'),\n"
         "               ('pl2_parallel',2,'parallel','pl-stream'),('cpu2_parallel',2,'parallel','cpu')]",
         "cases=[('pl1_parallel',1,'parallel','pl-stream'),('cpu1_parallel',1,'parallel','cpu')]")
    once("U02/f'reports/post_reboot_dw_ab_recovered_1792104845038960500/arm_before_{sample}.json'",
         "U02/f'milestones/matched_cpu1_and_1080_entry_verified_20261016/cpu_current/cpu1_parallel_{index}.{side}.json'")
    once("    def save():",
         "    state.update(hardware_bit_sha256=" + repr(bit_sha) +
         ",pl_clock_divisor=9,hardware_project=" + repr(str(project)) +
         ",external_hardware_owner=True)\n    def save():")
    # These operations retain the existing timeout/DMA ownership behaviour.
    a, b = main_helpers(original), main_helpers(source)
    for name in ("connect", "checked", "no_fds", "module_idle", "quiescent", "run_program", "original_run"):
        assert ast.dump(a[name], include_attributes=False) == ast.dump(b[name], include_attributes=False), name
    assert "program_hw_devices" not in source
    assert not any(word in source for word in ("taskkill", "force_unload"))
    compile(source, "<dw128-pair-child>", "exec")
    return source


def prepare(output):
    hardware = U02 / "milestones/dw128_v2_hardware_prepared_20261017"
    proof = verify_frozen(hardware)
    descriptor = json.loads((hardware / "descriptor.json").read_text())
    assert descriptor["offline_prepared"] and not descriptor["board_accessed"]
    verify_frozen(U02 / "milestones/matched_cpu1_and_1080_entry_verified_20261016")
    assert sha(U02 / PAIR_HOST) == HOST_SHA
    base = U02 / "inference/validate_pair_network_dual.py"
    source = base.read_text()
    output.mkdir(parents=True, exist_ok=False)
    children = {}
    for label in ("dw118", "candidate"):
        item = descriptor["hardware"][label]
        path = output / (label + "_pair_child.py")
        path.write_text(generate(source, hardware / item["project"], hardware / item["reports"], item["bit_sha256"]))
        help_result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, check=True)
        (output / (label + "_help.log")).write_text(help_result.stdout + help_result.stderr)
        children[label] = {"script": str(path), "script_sha256": sha(path), "bit_sha256": item["bit_sha256"]}
    result = {
        "offline_prepared": True, "board_accessed": False, "hardware_executed": False,
        "outer_hardware_owner_required": True, "source_sha256": sha(base),
        "hardware_evidence": proof, "host_sha256": HOST_SHA,
        "threads_per_network": 1, "warmup_pairs": 3, "measured_pairs_per_input": 10,
        "input_pairs": 10, "children": children,
        "scope": "Ready child scripts/help checks only; not safe to run without current-boot outer owner",
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.output.resolve())


if __name__ == "__main__":
    main()
