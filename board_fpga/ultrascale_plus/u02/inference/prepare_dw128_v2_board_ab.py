"""Offline ABI/identity preflight and matched board A/B contract. No board I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

U02 = Path(__file__).resolve().parents[1]
BASE_HLS = Path("D:/u02_formal_dw_address_lut_1792070172713158000/zu5ev/impl/ip")
NEW_HLS = Path("D:/u02_dw128_write_v2_20261017/zu5ev/impl/ip")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registers(text):
    values = {}
    for name, value in re.findall(
        r"(?m)^#define\s+\w+_CONTROL_((?:ADDR|BITS)_\w+)\s+(0x[0-9a-fA-F]+|\d+)\s*$", text
    ):
        name = name.replace("C_V_DATA", "C_DATA")
        assert name not in values, "Duplicate normalized register"
        values[name] = int(value, 0)
    assert values, "No HLS registers"
    return values


def check_registers(original, candidate, driver):
    a, b = registers(original), registers(candidate)
    assert a == b, "HLS register map or pointer width changed"
    for name, address in (
        ("RETURN", 0x10), ("A", 0x18), ("B", 0x24), ("C", 0x30),
        ("M", 0x3c), ("N", 0x44), ("K", 0x4c),
    ):
        macro = re.search(r"(?m)^#define U02_STREAM_REG_" + name + r"\s+(0x[0-9a-fA-F]+)\s*$", driver)
        assert macro and int(macro[1], 0) == address, name
    assert all(b["BITS_" + name + "_DATA"] == 64 for name in ("A", "B", "C"))
    return b


def verify_frozen(path):
    manifest = json.loads((path / "manifest.json").read_text())
    entries = manifest.get("files_sha256", manifest.get("files", manifest))
    assert entries
    for name, digest in entries.items():
        source = (path / name).resolve()
        assert source.is_relative_to(path.resolve())
        assert sha(source) == digest, name
    return {"manifest_sha256": sha(path / "manifest.json"), "verified_files": len(entries)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # Imports are local validators; none of these functions opens the board.
    from run_1080p_acceptance import verify_runtime
    from pair_host_elf import verify_pair_host_elf
    candidate = U02 / "milestones/dw128_write_v2_routed_118m_20261017"
    proof = verify_frozen(candidate)
    state = json.loads((candidate / "summary.json").read_text())
    assert state["route_checks_passed"] and state["effective_clock_guards_passed"]
    assert not state["board_verified"] and not state["formal_promoted"]
    runtime, _ = verify_runtime(U02 / "inference/stereo1080_sparse_rows_300s_verified_runtime_20261016.json")
    pair_archive = U02 / "milestones/matched_cpu1_and_1080_entry_verified_20261016"
    pair_proof = verify_frozen(pair_archive)
    host = pair_archive / "cpu_current/tested_pose_pair_network"
    assert sha(host) == "94806cbd5c4c8d15db26f3392e33e28073d2c75852d082cadcb551b126a3f65c"
    elf = verify_pair_host_elf(host)
    old_headers = list((BASE_HLS / "drivers").rglob("*_hw.h"))
    new_headers = list((NEW_HLS / "drivers").rglob("*_hw.h"))
    assert len(old_headers) == len(new_headers) == 1
    api = U02 / "board/linux/u02_stream_api.h"
    mapping = check_registers(old_headers[0].read_text(), new_headers[0].read_text(), api.read_text())
    sources = [old_headers[0], new_headers[0], api,
               U02 / "board/linux/u02_stream_resident.c",
               U02 / "board/linux/u02_stream_resident_validation.h",
               U02 / "board/linux/u02_stream_dual_api.h",
               U02 / "inference/hls/store_final_128_v2.h"]
    result = {
        "schema": "u02_dw128_v2_offline_ab_preflight_v1",
        "offline_preflight_passed": True, "board_accessed": False,
        "board_test_executed": False, "executable_hardware_owner_prepared": False,
        "candidate": proof, "candidate_bit_sha256": state["bit_sha256"],
        "baseline_bit_sha256": runtime["bit_sha256"],
        "pair_host_sha256": sha(host), "pair_host_elf": elf, "pair_evidence": pair_proof,
        "same_host_for_both_bits": True, "driver_sha256": runtime["driver_sha256"],
        "control_registers": mapping,
        "source_sha256": {str(p): sha(p) for p in sources},
        "matched_contract": {
            "threads_per_network": 1, "execution": "parallel",
            "warmup_pairs": 3, "measured_pairs_per_input": 10, "input_pairs": 10,
            "pl_clock_divisor_both_bits": 9, "cached_dma": True, "poll_spin_us": 1000,
            "pl_calls_per_image": 110, "expected_pl_calls_per_measured_pair": 220,
            "numerics": "All heatmap bytes, COCO17 coordinates and confidence match original INT8 backend",
            "timing_scope": "Prepared full-network stereo input; excludes JPEG, camera and network",
            "order": "DW118 before, candidate, DW118 after; same boot/governor/model/input",
        },
        "required_live_checks": [
            "New current-boot restoration/staging proof; historical boot record is insufficient",
            "No teammate program or camera/accelerator owner remains active",
            "Independent JTAG reset/AXI and actual clock observation before DMA",
            "128-bit partial-boundary writes and sentinels; cache ownership; consecutive submissions",
            "Full-network A/B then rollback and exact output recheck",
            "Only after numeric and timing gain: real 1080p 20s then 300s stereo/PC acceptance",
        ],
        "limits": "Source/ABI compatibility is not proof of board DDR/cache correctness or speedup",
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, source in (("baseline_hw.h", old_headers[0]), ("candidate_hw.h", new_headers[0]),
                         ("executed_script.py", Path(__file__))):
        shutil.copyfile(source, out / name)
    (out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"files_sha256": manifest}, indent=2) + "\n")
    print(json.dumps({"output": str(out), "board_accessed": False, "offline_preflight_passed": True}))


if __name__ == "__main__":
    main()
