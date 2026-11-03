from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description='Windows 双摄视频 → COCO17 三维骨架 → LAFAN1 BVH')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('gui', help='中文操作界面')
    rebind = sub.add_parser('rebind',help='USB 换口后，按已确认的物理相机映射生成新标定文件')
    rebind.add_argument('--calibration',required=True); rebind.add_argument('--session',required=True)
    rebind.add_argument('--output',required=True); rebind.add_argument('--old-indices',required=True,choices=['0,1','1,0'])
    board = sub.add_parser('board', help='生成实际尺寸标定板 SVG')
    board.add_argument('--output', required=True)
    board.add_argument('--cols', type=int, default=8); board.add_argument('--rows', type=int, default=5)
    board.add_argument('--square-mm', type=float, default=25)
    cal = sub.add_parser('calibrate', help='从双路标定录像估计相机参数')
    cal.add_argument('--session', required=True); cal.add_argument('--output', required=True)
    cal.add_argument('--cols', type=int, default=8); cal.add_argument('--rows', type=int, default=5)
    cal.add_argument('--square-mm', type=float, required=True)
    cal.add_argument('--every', type=int, default=30); cal.add_argument('--max-views', type=int, default=60)
    cal.add_argument('--max-delta-ms', type=float, default=5)
    cal.add_argument('--intrinsics-from', help='仅机位改变且镜头/焦点/分辨率不变时，复用该已验证标定的内参；本次重新求外参')
    world = sub.add_parser('world', help='地面板确定世界原点和方向')
    world.add_argument('--calibration', required=True); world.add_argument('--session', required=True)
    world.add_argument('--output', required=True); world.add_argument('--pair', type=int, default=-1)
    world.add_argument('--cols', type=int, default=8); world.add_argument('--rows', type=int, default=5)
    world.add_argument('--square-mm', type=float, required=True)
    world.add_argument('--board-height-mm', type=float, default=0, help='正面朝上水平板的印刷面离地高度；地面板为 0')
    world.add_argument('--placement', choices=['horizontal', 'vertical'], default='horizontal')
    world.add_argument('--vertical-origin-height-mm', type=float, help='竖直板棋盘图案上边缘离地高度（不含白边）；普通棋盘自动扣除第一内角点的一格偏移')
    for item in (board, cal, world):
        item.add_argument('--board-type', choices=['checkerboard', 'charuco'], default='checkerboard',
                          help='checkerboard 的 cols/rows 为内角点数；charuco 为格子列/行数')
        item.add_argument('--marker-mm', type=float)
        item.add_argument('--dictionary', default='DICT_5X5_100')
        item.add_argument('--legacy-pattern', action='store_true', help='匹配 OpenCV 4.6 前的偶数行 ChArUco 排列')
        item.add_argument('--first-marker-id', type=int, default=0)
        item.add_argument('--top-left-white', action='store_true', help='匹配实物左上角为带标记白格的排列')
    for name in ('pose', 'run'):
        item = sub.add_parser(name, help='二维识别' if name == 'pose' else '一次运行 视频 → LAFAN1 BVH')
        item.add_argument('--session', required=True); item.add_argument('--output', required=True)
        item.add_argument('--model'); item.add_argument('--device', default='auto', choices=['auto', 'cpu', '0'])
        item.add_argument('--every', type=int, default=2); item.add_argument('--max-delta-ms', type=float, default=10)
        item.add_argument('--limit', type=int); item.add_argument('--image-size', type=int, default=640)
        item.add_argument('--refine-person', action='store_true', help='先检测整图中的单人，再对人体区域高分辨率精细识别')
        if name == 'run':
            item.add_argument('--calibration', required=True)
            item.add_argument('--fps', type=float, default=30); item.add_argument('--max-gap-s', type=float, default=.15)
            item.add_argument('--tau', type=float, default=.04)
            item.add_argument('--align-content', action='store_true', help='离线从人体运动估计并校正两路内容时间差，保留原始时间戳')
            item.add_argument('--recover-articulated', action='store_true', help='校正内容同步后，以骨长/时序约束恢复；保留严格骨架，恢复版标记为估计')
            item.add_argument('--ensemble-person', action='store_true', help='补救模式：全图/人体区域双候选识别、内容同步、有限候选选择及离线约束恢复')
    align = sub.add_parser('align', help='从现有二维点校正录像内容时间差；不修改视频或标定')
    align.add_argument('--pose', required=True); align.add_argument('--calibration', required=True)
    align.add_argument('--output', required=True); align.add_argument('--max-offset-s', type=float, default=2.)
    align.add_argument('--offset-s', type=float, help='人工核对的右路原始时间减去的秒数；未指定则自动估计')
    recovery = sub.add_parser('recover', help='已核对内容同步的二维点 → 离线骨长/时序约束估计骨架')
    recovery.add_argument('--pose', required=True); recovery.add_argument('--calibration', required=True)
    recovery.add_argument('--output', required=True)
    fuse = sub.add_parser('fuse', help='已同步的全图/人体区域二维点 → 有限候选选择')
    fuse.add_argument('--full', required=True); fuse.add_argument('--roi', required=True)
    fuse.add_argument('--calibration', required=True); fuse.add_argument('--output', required=True)
    recon = sub.add_parser('reconstruct', help='双路二维点 → 三维骨架')
    recon.add_argument('--pose', required=True); recon.add_argument('--calibration', required=True)
    recon.add_argument('--output', required=True); recon.add_argument('--max-delta-ms', type=float, default=10)
    recon.add_argument('--max-reprojection-px', type=float, default=5); recon.add_argument('--tau', type=float, default=.04)
    export = sub.add_parser('export', help='三维骨架 → LAFAN1 BVH')
    export.add_argument('--skeleton', required=True); export.add_argument('--output', required=True)
    export.add_argument('--fps', type=float, default=30); export.add_argument('--max-gap-s', type=float, default=.15)
    sample = sub.add_parser('sample', help='人工抬手协议样例及 BVH，不是实测')
    sample.add_argument('--output', required=True)
    view = sub.add_parser('view', help='查看导出 BVH 对应的三维骨架')
    view.add_argument('--npz', required=True)
    return p


def generate_board(output, cols, rows, square_mm):
    from .calibration import board_points
    board_points(cols, rows, square_mm/1000)
    margin = 15.; width = (cols+1)*square_mm+2*margin; height = (rows+1)*square_mm+2*margin+12
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>']
    for y in range(rows+1):
        for x in range(cols+1):
            if (x+y)%2 == 0:
                svg.append(f'<rect x="{margin+x*square_mm}" y="{margin+y*square_mm}" width="{square_mm}" height="{square_mm}" fill="black"/>')
    svg.append(f'<text x="{margin}" y="{height-6}" font-size="4">Inner corners {cols} x {rows}; square {square_mm:g} mm; print 100%, measure actual size</text></svg>')
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(svg), encoding='utf-8')
    print(f'标定板已生成：{path}（页面 {width:g} × {height:g} mm，请按 100% 打印并实测格子边长）')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        spec = None
        if args.command in ('board', 'calibrate', 'world'):
            from .boards import BoardSpec
            spec = BoardSpec(args.board_type, args.cols, args.rows, args.square_mm/1000,
                             args.marker_mm/1000 if args.marker_mm is not None else None,
                             args.dictionary, args.legacy_pattern, args.first_marker_id, args.top_left_white)
        if args.command == 'gui':
            from .gui import launch
            return launch()
        if args.command == 'board':
            if spec.kind == 'charuco':
                from .boards import generate_charuco_svg
                generate_charuco_svg(args.output, spec)
            else:
                generate_board(args.output, args.cols, args.rows, args.square_mm)
        elif args.command == 'calibrate':
            from .calibration import calibrate
            if args.every < 1 or args.max_views < 12: raise ValueError('步长至少 1，视角上限至少 12。')
            if Path(args.output).exists(): raise ValueError('标定输出已存在，请使用新文件。')
            calibrate(args.session, args.output, args.cols, args.rows, args.square_mm/1000, args.every, args.max_views, args.max_delta_ms, board_spec=spec, intrinsics_from=args.intrinsics_from)
        elif args.command == 'world':
            from .calibration import set_world
            if Path(args.output).exists(): raise ValueError('世界标定输出已存在，请使用新文件。')
            set_world(args.calibration, args.session, args.output, args.pair, args.cols, args.rows, args.square_mm/1000, board_spec=spec, board_height_m=args.board_height_mm/1000, placement=args.placement, vertical_origin_height_m=None if args.vertical_origin_height_mm is None else args.vertical_origin_height_mm/1000)
            print('地面世界坐标已保存。请按 README 核对原点、前方和 +Z 朝上。')
        elif args.command == 'rebind':
            from .calibration import rebind_calibration
            rebind_calibration(args.calibration,args.session,args.output,[int(x) for x in args.old_indices.split(',')])
        elif args.command == 'pose':
            from .pose import extract
            extract(args.session, args.output, args.model, args.device, args.every, args.max_delta_ms, args.limit, args.image_size, args.refine_person)
        elif args.command == 'reconstruct':
            from .reconstruction import reconstruct
            reconstruct(args.pose, args.calibration, args.output, args.max_delta_ms, args.max_reprojection_px, args.tau)
        elif args.command == 'align':
            from .temporal import align_pose
            align_pose(args.pose, args.calibration, args.output, args.max_offset_s, args.offset_s)
        elif args.command == 'recover':
            from .recovery import recover
            recover(args.pose, args.calibration, args.output)
        elif args.command == 'fuse':
            from .ensemble import fuse_pose
            fuse_pose(args.full, args.roi, args.calibration, args.output)
        elif args.command == 'export':
            from .lafan import export_lafan
            export_lafan(args.skeleton, args.output, args.fps, args.max_gap_s)
        elif args.command == 'sample':
            from .samples import generate_sample
            generate_sample(args.output)
        elif args.command == 'view':
            from .viewer import view_npz
            view_npz(args.npz)
        elif args.command == 'run':
            from .io import session_info, write_json
            from .calibration import load_calibration
            from .pose import extract
            from .reconstruction import reconstruct
            from .lafan import export_lafan
            cal = load_calibration(args.calibration, session_info(args.session))
            if cal.get('world_from_cam0') is None: raise ValueError('一键导出需要先完成地面世界坐标标定。')
            output = Path(args.output)
            if output.exists(): raise ValueError('一键流程输出目录已存在，请选择新目录。')
            output.mkdir(parents=True)
            report = dict(state='incomplete', session=str(Path(args.session).resolve()), calibration_id=cal['calibration_id'],
                          stage='pose2d', output_side='Windows PC')
            write_json(output/'pipeline.json', report)
            try:
                extract(args.session, output/'pose2d.jsonl', args.model, args.device, args.every, args.max_delta_ms, args.limit, args.image_size,
                        args.refine_person and not args.ensemble_person)
                if args.ensemble_person:
                    report['stage'] = 'pose2d_roi'; write_json(output/'pipeline.json', report)
                    extract(args.session, output/'pose2d_roi.jsonl', args.model, args.device, args.every, args.max_delta_ms, args.limit, args.image_size, True)
                report['stage'] = 'reconstruction'; write_json(output/'pipeline.json', report)
                pose_input = output/'pose2d.jsonl'
                if args.align_content or args.recover_articulated or args.ensemble_person:
                    from .temporal import align_pose
                    report['stage'] = 'content_alignment'; write_json(output/'pipeline.json', report)
                    align_pose(pose_input, args.calibration, output/'pose2d_aligned.jsonl')
                    pose_input = output/'pose2d_aligned.jsonl'
                if args.ensemble_person:
                    from .ensemble import fuse_pose
                    align_pose(output/'pose2d_roi.jsonl', args.calibration, output/'pose2d_roi_aligned.jsonl')
                    fuse_pose(pose_input, output/'pose2d_roi_aligned.jsonl', args.calibration, output/'pose2d_fused.jsonl')
                    pose_input = output/'pose2d_fused.jsonl'
                reconstruct(pose_input, args.calibration, output/'skeleton3d.jsonl', args.max_delta_ms, tau=args.tau)
                skeleton_input = output/'skeleton3d.jsonl'
                if args.recover_articulated or args.ensemble_person:
                    from .recovery import recover
                    report['stage'] = 'offline_articulated_recovery'; write_json(output/'pipeline.json', report)
                    recover(pose_input, args.calibration, output/'skeleton3d_recovered.jsonl')
                    skeleton_input = output/'skeleton3d_recovered.jsonl'
                report['stage'] = 'lafan1_export'; write_json(output/'pipeline.json', report)
                quality = export_lafan(skeleton_input, output/'lafan1', args.fps, args.max_gap_s)
                report.update(state='complete', stage='done', quality_status=quality['quality_status'],
                              coverage_ratio=quality['coverage_ratio'], exported_frames=quality['exported_frames'],
                              pose_file=str(pose_input), skeleton_file=str(skeleton_input),
                              offline_estimated_recovery=bool(args.recover_articulated or args.ensemble_person))
                print(f'视频 → LAFAN1 完成：{output}', flush=True)
            except BaseException as exc:
                report['error'] = str(exc); raise
            finally: write_json(output/'pipeline.json', report)
        return 0
    except Exception as exc:
        print(f'处理失败：{exc}', file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
