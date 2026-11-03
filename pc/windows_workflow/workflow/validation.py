"""Independent quality summaries and repeatable checks of recorded validation inputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import av
import cv2
import numpy as np
from mocap.boards import BoardSpec, Target
from mocap.calibration import calibrate_views, load_calibration, stereo_board_pose
from mocap.io import read_json, write_json, pairs, session_info, camera_ids
from mocap.lafan import CORE


def refit_selected_views(session, selection_file, intrinsics_file, output):
    """Re-detect actual pixels of recorded accepted views, fixing documented prior intrinsics."""
    session, output = Path(session), Path(output)
    if output.exists(): raise ValueError('复核输出已存在。')
    selection = read_json(selection_file); meta = session_info(session)
    prior = load_calibration(intrinsics_file, meta)
    spec = BoardSpec(**{k:v for k,v in selection['board'].items() if k != 'count_semantics'})
    target = Target(spec); requested = [{row[side+'_frame'] for row in selection['selected_frames']} for side in ('left','right')]
    detections = [{}, {}]
    for c, side in enumerate(('left','right')):
        with av.open(str(session/(side+'.mkv'))) as video:
            index = -1
            for packet in video.demux(video=0):
                if not packet.size: continue
                index += 1
                if index not in requested[c]: continue
                image = cv2.imdecode(np.frombuffer(bytes(packet), np.uint8), cv2.IMREAD_COLOR)
                if image is None: raise ValueError('复核帧解码失败。')
                found = target.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if found is None: raise ValueError(f'复核 cam{c} 帧 {index} 没有检测到棋盘。')
                detections[c][index] = found
                print('内参复用复核', side, index, flush=True)
                if len(detections[c]) == len(requested[c]): break
    objects, left, right = [], [], []
    for row in selection['selected_frames']:
        obj, a, b, ids = target.match(detections[0][row['left_frame']], detections[1][row['right_frame']])
        objects.append(obj); left.append(a); right.append(b)
    result = calibrate_views(objects, left, right, tuple(prior['image_size']), align_orders=True, intrinsics=prior)
    result.update(schema='stereo_calibration/1.0', unit='m', image_size=prior['image_size'],
                  camera_ids=camera_ids(meta), board=spec.to_dict(), selected_frames=selection['selected_frames'],
                  source_session=str(session.resolve()), coordinate_system='cam0_optical_right_down_forward',
                  world_from_cam0=None, world_coordinate_system=None,
                  intrinsics_reuse=dict(source=str(Path(intrinsics_file).resolve()), source_calibration_id=prior['calibration_id'],
                                        condition='Historical reuse documented in calibration_3: unchanged lenses/focus/resolution; no independent new intrinsic validation.'),
                  selection_source=str(Path(selection_file).resolve()))
    result['calibration_id'] = 'stereo-'+hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    write_json(output, result); return result


def validate_world_views(session, calibration, output):
    cal = load_calibration(calibration, session_info(session)); board = cal['board']
    target = Target(BoardSpec(**{k:v for k,v in board.items() if k != 'count_semantics'}))
    views = []
    for i, pair in enumerate(pairs(session, 5)):
        if i % 80: continue
        match = target.common(pair['left_rgb'], pair['right_rgb'])
        if match is None: views.append(dict(pair=i, detected=False)); continue
        obj, left, right, ids = match
        rv, tv, quality = stereo_board_pose(obj, left, right, cal, reversible=True)
        views.append(dict(pair=i, detected=True, corners=len(ids), quality=quality))
        print('世界坐标多帧复核',i,quality['joint_point_rms_px'],flush=True)
    report = dict(views=views, checked=len(views), detected=sum(v['detected'] for v in views),
                  calibration=str(Path(calibration).resolve()), physical_gravity_verified=False,
                  height_source=cal.get('ground_board',{}).get('pattern_top_edge_height_m'),
                  note='Checks stereo geometry on distinct video frames, not independent physical height/plumb ground truth.')
    write_json(output, report); return report


def summarize_action(folder):
    folder = Path(folder); manifest = read_json(folder/'workflow.json')
    rows = [json.loads(x) for x in (folder/'skeleton3d.jsonl').read_text().splitlines() if x]
    live = [json.loads(x) for x in (folder/'live_lafan.jsonl').read_text().splitlines() if x]
    timing = [r['timing'] for r in rows]
    duration = (rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])/1e9 if len(rows)>1 else 0.
    valid_core = [all(r['joints'][j]['valid'] for j in CORE) for r in rows]
    reasons = Counter(j.get('reason','unknown') for row in rows for j in row['joints'] if not j['valid'])
    report = dict(folder=str(folder.resolve()), recorded_rows=len(rows), span_s=duration,
                  actual_skeleton_fps=(len(rows)-1)/duration if duration else 0.,
                  full_core_frames=sum(valid_core), full_core_fraction=float(np.mean(valid_core)) if rows else 0.,
                  live_valid_frames=sum(r['valid'] for r in live), invalid_reasons=dict(reasons),
                  per_joint_valid_ratio=[float(np.mean([r['joints'][j]['valid'] for r in rows])) if rows else 0. for j in range(17)],
                  initialization_completed=any(r['valid'] for r in live),
                  offline_export=read_json(folder/'lafan1/export_report.json') if (folder/'lafan1/export_report.json').exists() else None,
                  export_error=manifest.get('bvh_export_error'), synchronization=manifest['synchronization'],
                  external_motion_accuracy_validated=False)
    for field in ('pair_delta_ms','processing_ms','decode_ms','inference_ms','reconstruction_ms','lafan_ms','host_receive_to_pose_ms'):
        values = [t[field] for t in timing if field in t]
        report[field] = dict(median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else None
    write_json(folder/'quality_summary.json',report); return report


def compare_cached_pose(pose_file, calibration, output):
    """Keep model observations fixed; isolate the effect of dropping inference samples."""
    from mocap.reconstruction import triangulate, CausalFilter
    from mocap.io import frame_packet
    from mocap.lafan import export_lafan
    output=Path(output)
    if output.exists():raise ValueError('对照目录已存在。')
    output.mkdir(parents=True)
    rows=[json.loads(x) for x in Path(pose_file).read_text(encoding='utf-8').splitlines() if x]
    meta=read_json(Path(pose_file).with_suffix('.meta.json'));cal=load_calibration(calibration)
    if meta['camera_ids']!=cal['camera_ids']:raise ValueError('对照二维点相机顺序不同。')
    report=dict(source_pose=str(Path(pose_file).resolve()), uses_cached_offline_content_aligned_2d=True,
                not_a_live_sync_test=True, modes=[])
    for name, rate in [('all_observations',None),('sampled_8fps',8.)]:
        selected=[];due=rows[0]['timestamp_ns']
        for row in rows:
            if rate is None or row['timestamp_ns']>=due:
                selected.append(row)
                if rate is not None:
                    due=rows[0]['timestamp_ns']+round(len(selected)/rate*1e9)
        folder=output/name;folder.mkdir();filt=CausalFilter();complete=0
        with (folder/'skeleton3d.jsonl').open('x',encoding='utf-8') as dst:
            for i,row in enumerate(selected):
                joints=filt.apply(triangulate(row['views'][0]['joints'],row['views'][1]['joints'],cal),row['timestamp_ns'])
                complete+=int(all(joints[j]['valid'] for j in CORE))
                packet=frame_packet(i,row['timestamp_ns'],joints,session_id='cached_2d_comparison',
                                    calibration_id=cal['calibration_id'],model_id=meta['model_id'],
                                    coordinate_system='world_right_handed_X_forward_Y_left_Z_up',clock_id='recorded_offline_aligned')
                dst.write(json.dumps(packet,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(folder/'skeleton3d.meta.json',dict(state='complete',temporal_alignment=meta.get('temporal_alignment')))
        try:
            export_lafan(folder/'skeleton3d.jsonl',folder/'lafan1',fps=30)
            exported=read_json(folder/'lafan1/export_report.json')
            result=dict(mode=name,input_rows=len(selected),full_core_frames=complete,
                        exported_frames=exported['exported_frames'],exported_duration_s=exported['exported_duration_s'],
                        coverage_ratio=exported['coverage_ratio'],segments=len(exported['segments']))
        except ValueError as exc:result=dict(mode=name,input_rows=len(selected),full_core_frames=complete,error=str(exc))
        report['modes'].append(result)
    write_json(output/'comparison.json',report);return report


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    refit=sub.add_parser('refit'); refit.add_argument('--session',required=True); refit.add_argument('--selection',required=True)
    refit.add_argument('--intrinsics',required=True); refit.add_argument('--output',required=True)
    world=sub.add_parser('world-check'); world.add_argument('--session',required=True); world.add_argument('--calibration',required=True); world.add_argument('--output',required=True)
    summary=sub.add_parser('summary'); summary.add_argument('--folder',required=True)
    compare=sub.add_parser('compare');compare.add_argument('--pose',required=True);compare.add_argument('--calibration',required=True);compare.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='refit': refit_selected_views(a.session,a.selection,a.intrinsics,a.output)
    elif a.command=='world-check': validate_world_views(a.session,a.calibration,a.output)
    elif a.command=='compare':print(json.dumps(compare_cached_pose(a.pose,a.calibration,a.output),ensure_ascii=False,indent=2))
    else: print(json.dumps(summarize_action(a.folder),ensure_ascii=False,indent=2))
