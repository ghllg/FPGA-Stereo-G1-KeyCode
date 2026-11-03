import argparse
import time
from pathlib import Path
from windows_camera.capture import Config, Device
from .engine import Engine


def main():
    parser = argparse.ArgumentParser(description='Windows 双摄实时骨架流程')
    parser.add_argument('--demo', action='store_true', help='模拟输入，不打开物理相机')
    parser.add_argument('--headless', action='store_true', help='仅用于模拟流程自动验证')
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.headless:
        from .gui import launch
        return launch(args.demo)
    if not args.demo or not args.output: parser.error('--headless 需要 --demo 和 --output 新目录')
    if args.output.exists(): parser.error('输出目录已存在，请使用新目录，避免误读旧验证结果')
    if args.seconds < 13: parser.error('模拟校时需 10 秒；总时长至少 13 秒')
    config = Config((Device('demo-left', 'demo-left'), Device('demo-right', 'demo-right')),
                    width=640, height=360, fps=60, demo=True)
    engine = Engine(config, demo=True); engine.start()
    start = time.perf_counter(); synced = recording = False
    try:
        while time.perf_counter()-start < args.seconds and engine.thread.is_alive():
            snapshot = engine.read()
            if engine.source and not synced:
                engine.request('sync'); synced = True
            if snapshot.get('initialized') and not recording:
                engine.request('record', args.output); recording = True
            time.sleep(.05)
    finally:
        engine.stop(); engine.thread.join(timeout=40)
    print(engine.read().get('status'))
    if engine.error or engine.thread.is_alive(): raise RuntimeError(engine.error or '流程停止超时')
    if not recording or not (args.output/'lafan1/export_report.json').exists(): raise RuntimeError('模拟端到端验证未生成 BVH，请检查校时与记录状态。')
    print('模拟验证完成（不代表真实相机同步、精度或性能）：', args.output)
    return 0


if __name__ == '__main__': raise SystemExit(main())
