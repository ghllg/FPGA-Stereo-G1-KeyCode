"""Keep the proven START-gated board lifecycle and attach the monitored PC backend."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs/full_pipeline_exec_20261022'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--board-record', type=Path, required=True)
board_record = parser.parse_args().board_record.resolve()
assert board_record.is_relative_to(ROOT/'outputs')
stage = json.loads((board_record/'live_stage.json').read_text())
proof = json.loads((board_record/'run.json').read_text())
preflight = json.loads((board_record/'preflight.json').read_text())
storage = json.loads((board_record/'snapshot_storage_verification.json').read_text())
assert stage['complete'] and proof['complete'] and proof['cleanup_verified'] and not proof['dma_ambiguous']
assert preflight['complete'] and preflight['cleanup_verified'] and storage['passed']
assert all(record['boot_id']==stage['boot_id'] for record in (proof,preflight,storage))
source = (ROOT/'work/completion_20261022/live_snapshots.py').read_text(encoding='utf-8')


def replace(old, new, count=1):
    global source
    if source.count(old) != count:
        raise ValueError('Unexpected capture baseline: '+old[:80])
    source = source.replace(old, new)


replace('import hashlib', 'import argparse\nimport hashlib')
replace('from receive_paired_pose import Receiver', 'from live_receiver import LiveReceiver as Receiver\nfrom backend import Backend')
replace('outputs/completion_20261022/live_runs', 'outputs/full_pipeline_exec_20261022/live_runs')
for name in ('live_stage.json', 'preflight.json', 'snapshot_storage_verification.json'):
    replace(f'(OUT.parent / "{name}")', f'Path({str(board_record/name)!r})')
replace('(ROOT.parents[1] / "outputs/full_chain_20261022/board/run.json")',
        f'Path({str(board_record/"run.json")!r})')
replace('def main():', '''def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, choices=(30,60,300), default=30)
    args = parser.parse_args()
    seconds = args.seconds''')
replace('print("20 秒手腕动作诊断，约每秒保留一对图像及热图。摄像头不动，不用棋盘。")',
        'print(f"{seconds} 秒真实全流程性能测试：双摄→FPGA→在线三维→G1→原生窗口。摄像头不动，不用棋盘。")')
replace('print("站在原采集位置；短提示音后静立 5 秒，再缓慢抬手、放下 10 秒，最后双手放下静立至长音。")',
        'print("站在原位置；短音后先静立 5 秒，再缓慢抬手至肩高并放下，循环到末尾 5 秒静立；长音结束。")')
replace('print("本次是双摄到 FPGA 再到电脑的诊断，结束后离线检查三维结果；尚未接通实时 G1。")',
        'print("本次实时运行在线三维和 G1，记录延时及帧率；手腕精度与地面尺度仍保留诊断标记。")\n    print("本轮 G1 输入加入轻度去抖，会增加少量动作响应滞后；先观察静立时脚部是否稳定。")\n    print("START 和倒计时后先检查相机同步，最多三次；短音确认有效三维已驱动 G1，再开始动作。同步不合格会退出，不放宽 10 毫秒门限。")')
replace('    receiver = None', '    receiver = None\n    backend = None', count=1)
replace('            for second in range(10, 0, -1):', '''            selected = json.loads((OUT.parent/'selected_backend.json').read_text())
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
            for second in range(10, 0, -1):''')
replace('expected_pair_limit_us=16667)', 'expected_pair_limit_us=16667, backend=backend)')
replace('POSE_RUN_SECONDS=20', 'POSE_RUN_SECONDS={seconds}')
replace('host_change="bounded snapshot diagnostic candidate; model arithmetic unchanged"',
        'host_change="bounded camera-only 1080p startup qualification; model arithmetic unchanged"')
replace('while time.monotonic() - begun < 120:', 'while time.monotonic() - begun < seconds+90:')
replace('if receiver.rows and not sounded:', "if (output/'states.ndjson').exists() and (output/'states.ndjson').stat().st_size and not sounded:")
replace('winsound.Beep(900,250); print("短音：开始动作。", flush=True); sounded = True',
        'state["motion_cue_pc_ns"] = time.perf_counter_ns(); save(); winsound.Beep(900,250); print("短音：有效三维已驱动 G1。先静立 5 秒，再缓慢抬放手。", flush=True); sounded = True')
replace('            receiver.finish(); receiver = None', '''            receiver.finish()
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
                backend = None''')
replace('            save(); c.close()', '''            if backend:
                try: backend.close()
                except Exception as error: state['backend_cleanup_error'] = repr(error)
            save(); c.close()''')
replace('print("本段已完成并回传。", output, flush=True)',
        'print("原始数据已采集并回传；全流程是否有效以下方核验为准。", output, flush=True)')
replace('        from audit_snapshots import audit', '''        sys.path.insert(0,str(ROOT.parents[1]/'work/completion_20261022'))
        from audit_snapshots import audit''')
replace('        print("自动核验已保存；是否合格仍需结合画面检查。", flush=True)', '''        if (output/'backend.json').exists():
            from metrics import analyze
            report = analyze(output)
            state['performance_checks_passed'] = report['headline']['live_full_chain_passed']
            save()
            print(json.dumps(report['headline'],ensure_ascii=False,indent=2),flush=True)
        print("全流程证据和核验已保存；未达目标的项目将继续优化。", flush=True)''')
ast.parse(source)
(HERE/'live_full_pipeline.py').write_text(source, encoding='utf-8')
entry = OUT/'运行30秒全流程基线.ps1'
entry.write_text("$ErrorActionPreference = 'Stop'\n& 'E:\\FPGAchina\\FPGA2026\\windows_vision\\.venv\\Scripts\\python.exe' -X utf8 '"+
    str(HERE/'live_full_pipeline.py')+"' --seconds 30\nif ($LASTEXITCODE -ne 0) { throw '全流程测试中断，请保留窗口和日志。' }\n", encoding='utf-8-sig')
print('PREPARED', entry)
