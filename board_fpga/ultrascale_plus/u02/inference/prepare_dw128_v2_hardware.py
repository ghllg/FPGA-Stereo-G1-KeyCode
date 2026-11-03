"""Create explicit candidate/baseline/rollback paths from frozen evidence only.

This never invokes Vivado, UART or a network. Loading scripts are helper inputs
for a current-boot, quiescent hardware owner, not a standalone deployment flow.
"""
import argparse
import json
from pathlib import Path
import shutil
from prepare_dw128_v2_board_ab import U02, sha, verify_frozen
from prepare_portable_hardware import load_prepared
from bounded_pl_clock_check import instrument_capture


def render(source, project, reports, kind):
    assert kind in ("program", "verify", "idle_readonly")
    paths = (project, reports, U02 / "board")
    assert all(not any(c in str(p) for c in "{}\r\n") for p in paths)
    lookup = "source [file join [file dirname [file normalize [info script]]] stream_artifacts.tcl]"
    root = "set root [file normalize [file join [file dirname [file normalize [info script]]] ../board]]"
    assert source.count(lookup) == source.count(root) == 1
    source = source.replace(lookup, "set u02_project {" + project.as_posix() +
                            "}\nset u02_implementation {" + reports.as_posix() + "}")
    source = source.replace(root, "set root {" + paths[2].as_posix() + "}")
    if kind == "idle_readonly":
        marker = "create_hw_axi_txn u02_write_count"
        assert source.count(marker) == 1
        source = source[:source.index(marker)]
        source += "close_hw_target\ndisconnect_hw_server\nclose_hw\nputs U02_READONLY_CORE_IDLE_VERIFIED\n"
        assert "-type WRITE" not in source and "program_hw_devices" not in source
    else:
        assert ("program_hw_devices" in source) == (kind == "program")
        assert "U02_STREAM_RESET_AND_AXI_INDEPENDENT_VERIFICATION_PASSED" in source
    source = instrument_capture(source, 13)
    assert "stream_artifacts.tcl" not in source and "D:/u02_" not in source
    return source


def prepare(destination):
    candidate = U02 / "milestones/dw128_write_v2_routed_118m_20261017"
    proof = verify_frozen(candidate)
    candidate_state = json.loads((candidate / "summary.json").read_text())
    assert candidate_state["route_checks_passed"] and candidate_state["effective_clock_guards_passed"]
    assert not candidate_state["board_verified"] and not candidate_state["formal_promoted"]
    portable = load_prepared(U02 / "milestones/portable_hardware_prepared_20261015/runtime_tree")
    artifacts = {
        label: (item["bit"], item["reports"] / "compute.ltx")
        for label, item in portable.items() if label in ("original", "dw118")
    }
    artifacts["candidate"] = (candidate / "compute_wrapper.bit", candidate / "implementation/compute.ltx")
    assert sha(artifacts["candidate"][0]) == candidate_state["bit_sha256"]
    assert len({sha(bit) for bit, ltx in artifacts.values()}) == 3
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "u02_dw128_v2_hardware_tree_v1", "offline_prepared": True,
        "board_accessed": False, "board_verified": False, "formal_promoted": False,
        "candidate_evidence": proof,
        "baseline_descriptor_sha256": portable["descriptor_sha256"],
        "preconditions": "Current-boot restoration proof; no live users; known idle and DMA ownership; driver unloaded before programming",
        "clock_policy": {"original_rollback_divisor": 11, "dw118_divisor": 9, "candidate_divisor": 9},
        "warning": "Helper Tcl does not enforce ARM/DMA ownership; invoke only inside verified owner",
        "hardware": {},
    }
    for label, (bit, ltx) in artifacts.items():
        project, reports = destination / label / "project", destination / label / "reports"
        target = project / "u02_stream.runs/impl_1/compute_wrapper.bit"
        target.parent.mkdir(parents=True)
        reports.mkdir()
        shutil.copyfile(bit, target)
        shutil.copyfile(ltx, reports / "compute.ltx")
        assert sha(target) == sha(bit) and sha(reports / "compute.ltx") == sha(ltx)
        result["hardware"][label] = {
            "project": project.relative_to(destination).as_posix(),
            "reports": reports.relative_to(destination).as_posix(),
            "bit_sha256": sha(target), "ltx_sha256": sha(ltx),
        }
        for kind in ("program", "verify", "idle_readonly"):
            filename = "program_stream.tcl" if kind == "program" else "verify_resident_interface.tcl"
            (destination / label / (kind + ".tcl")).write_text(
                render((U02 / "inference" / filename).read_text(), project, reports, kind)
            )
    (destination / "descriptor.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(Path(__file__), destination / "executed_script.py")
    (destination / ".gitattributes").write_text("* -text\n")
    (destination / ".gitignore").write_text("!**/\n!**/*\n")
    (destination / "README.md").write_text(
        "# DW128 v2 isolated hardware identities\n\n"
        "Offline preparation only. Three separate bit/LTX pairs: original rollback, "
        "DW118 A/B baseline and DW128-v2 candidate. No board access or promotion.\n\n"
        "Tcl helpers retain the existing reset/AXI verification and use 13 bounded "
        "clock captures. idle_readonly contains neither programming nor AXI writes. "
        "These helpers do not enforce ARM/DMA ownership: use only inside an owning "
        "test process after current-boot proof and quiescence. Never run program.tcl "
        "against an active teammate program or ambiguous DMA state.\n"
    )
    manifest = {p.relative_to(destination).as_posix(): sha(p)
                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "manifest.json").write_text(json.dumps({"files_sha256": manifest}, indent=2) + "\n")
    verify_frozen(destination)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = prepare(args.output.resolve())
    print(json.dumps({"board_accessed": False, "offline_prepared": True,
                      "hardware": result["hardware"]}, indent=2))


if __name__ == "__main__":
    main()
