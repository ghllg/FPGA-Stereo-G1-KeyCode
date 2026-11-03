"""Summarize the three explicitly selected local video inputs and replay variants."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from mocap.io import read_json, write_json
from . import ROOT
from .validation import summarize_action


def audit_mapping(folder):
    folder=Path(folder)
    rows=[json.loads(x) for x in (folder/'pose2d.jsonl').read_text(encoding='utf-8').splitlines()]
    for c,side in enumerate(('left','right')):
        with (folder/'raw'/f'{side}_timestamps.csv').open(encoding='utf-8',newline='') as f: table=list(csv.DictReader(f))
        indices=[r['views'][c]['saved_video_frame_index'] for r in rows]
        if any(a>=b for a,b in zip(indices,indices[1:])):raise ValueError('记录配对重复使用视频帧。')
        for row in rows:
            v=row['views'][c]
            if int(table[v['saved_video_frame_index']]['source_timestamp_ns'])!=v['raw_timestamp_ns']:
                raise ValueError('记录配对与保存视频时间表不一致。')
    npzs=list((folder/'lafan1').glob('*.npz'))
    for path in npzs:
        with np.load(path) as d:
            np.testing.assert_allclose(np.linalg.norm(d['local_quaternions_wxyz'],axis=2),1.,atol=1e-6)
            if not np.isfinite(d['global_positions_cm']).all():raise ValueError('NPZ 有非有限骨架。')
            lines=path.with_suffix('.bvh').read_text(encoding='ascii').splitlines(); idx=lines.index('MOTION')
            if int(lines[idx+1].split(':')[1])!=len(d['timestamps_ns']):raise ValueError('BVH 与 NPZ 帧数不同。')
            if any(len(line.split())!=69 for line in lines[idx+3:]):raise ValueError('BVH 通道数不符。')
    return dict(mapped_pairs_checked=len(rows),bvh_npz_segments_checked=len(npzs),passed=True)


def build_report(base):
    base=Path(base);fixed=read_json(base/'calibration_refit_fixed.json');world=read_json(base/'calibration_world_refit.json')
    actions=[]
    for name in ('action_raw','action_offset_only','action_aligned_n640','action_aligned_n640_roi','action_aligned_m960'):
        folder=base/name
        report=summarize_action(folder);report['audit']=audit_mapping(folder);actions.append(dict(name=name,**report))
    comparison=read_json(base/'same_observations_comparison/comparison.json')
    report=dict(overall_status='not_ready_for_full_30fps_realtime_motion',
                calibration_fresh=dict(status='failed',stereo_rms_px=9.29,
                                       principal_points_px=[[588.7,763.3],[1815.9,315.9]],
                                       detection=read_json(base/'calibration_fresh.detection.json')),
                calibration_fixed_intrinsics=dict(status='conditional_pass',quality=fixed['quality'],
                                                  intrinsics_reuse=fixed['intrinsics_reuse']),
                world=dict(status='stereo_geometry_pass',ground_board=world['ground_board'],
                           multi_frame_check=read_json(base/'world_multiframe_check.json')),
                action_tests=actions,same_observation_comparison=comparison,
                preview=str((base/'review_actual_causal_m960.mp4').resolve()),
                notes=['No new physical-camera capture, hardware-trigger test, or physical motion ground truth.',
                       'Aligned replay schedules packets using a prior offline content offset; it is not proof of live flash synchronization.',
                       'Initial replay tests overlapped CPU calibration work; performance is a local observation, not an isolated benchmark.',
                       'Occasional pre-existing JPEG marker warnings occurred in unaligned replay; videos were not edited.',
                       'Existing source videos and calibration files are read-only inputs; new outputs live only in this validation folder.'])
    write_json(base/'validation_report.json',report)
    text=['Windows 端三段录像流程校验（2026-10-05）','结论：标定几何可条件通过，完整实时 30 FPS 动作流程未通过。','',
          '输入：',
          '1. 20261004_193708_354537_双目：9×6 内角点、41 mm 棋盘格。',
          '2. 20261004_193927_155275_竖直：世界坐标；沿用旧标定记录的图案上边缘高度 435 mm。',
          '3. 20261004_194029_159134_动作：原始动作录像。','',
          '标定结果：',
          '从零估计内外参：双目 RMS 9.29 px，拒绝保存为有效标定；单目误差小但主点严重漂移。',
          f'重新检测 21 组原录像标定帧，复用历史 calibration_2 内参重求外参：RMS {fixed["quality"]["stereo_rms_px"]:.3f} px。',
          '这依赖镜头、焦点、分辨率未改变的历史前提，不能说第一段独立完成了可靠的新内参标定。',
          f'竖直板联合位姿 RMS {world["ground_board"]["pose_quality"]["joint_point_rms_px"]:.3f} px；跨三组录像帧复核通过，其中两组未用于首次世界定位。',
          '未验证实物重力方向或离地高度真值。','',
          '动作对照（以下 BVH 帧数是停止后的 30 FPS 重采样帧数，不是实时有效姿态数）：']
    for a in actions:
        exp=a['offline_export'];detail=f'{exp["exported_frames"]} 帧 / {exp["exported_duration_s"]:.2f} 秒 / {len(exp["segments"])} 片段 / 覆盖 {exp["coverage_ratio"]:.1%}' if exp else '无法导出 BVH'
        text.append(f'{a["name"]}: 实测处理 {a["actual_skeleton_fps"]:.2f} FPS；完整身体点 {a["full_core_frames"]}/{a["recorded_rows"]}；实际有效实时 LAFAN 姿态 {a["live_valid_frames"]}；{detail}。')
    text.extend(['','为什么旧版能导出 7–8 秒：',
                 '已用旧版同一份 m960 二维观测、同样的因果三角化/滤波和 BVH 导出器复现：'])
    for mode in comparison['modes']:
        text.append(f'{mode["mode"]}: {mode["input_rows"]} 输入帧 → {mode["exported_frames"]} BVH 帧，{mode["exported_duration_s"]:.2f} 秒，覆盖 {mode["coverage_ratio"]:.1%}。')
    text.extend(['全部观测复现旧版 235 帧 / 7.83 秒；仅降低观测采样到约 8 FPS 后为 157 帧 / 5.23 秒。',
                 '实时原视频回放得到 107 帧 / 3.57 秒：丢采样只能解释部分差异。实时回放匹配原生视频帧，旧对齐二维点用了相邻右路帧时间插值，两者的观测并不完全相同。',
                 '较大的模型和 n 模型人体区域精细识别已开放为界面选项；提高精度会增加处理耗时。','',
                 '同步边界：',
                 '原始录像两路内容约差 935 ms。按原始收帧节奏仅修改右路时间戳时，150 ms 有限缓冲不能容纳这段错位，测试正确拒绝配对，生成 0 帧骨架。',
                 '预对齐文件回放使用已经录好的未来视频包，单纯用于检验后续算法；不能据此宣称新摄像头入口同步已经通过。',
                 '停止后的短缺口插值和整段离线补救，不是实时骨架计算。旧版恢复后的 445 帧完整输出仍保留在 windows_vision。','',
                 '检查内容：保存视频索引与原始时间逐项一致、无重复配对、BVH/NPZ 帧数和 69 通道一致、四元数归一化、有限 FK 位置。',
                 '原文件未覆盖。详细数据见 validation_report.json、各动作目录 quality_summary.json、world_multiframe_check.json。',
                 '对照预览：review_actual_causal_m960.mp4。'])
    (base/'测试报告.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
    r=build_report(a.folder);print(r['overall_status'])
