"""Physical checkerboard calibration; all geometry is in metres."""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .io import pairs, session_info, camera_ids, write_json, read_json, calibration_samples


def board_points(cols, rows, square_m):
    if cols < 3 or rows < 3 or not 0 < square_m < 1:
        raise ValueError('内角点至少 3×3，格子边长以米输入。')
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m
    return points


def corners(rgb, cols, rows):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, points = cv2.findChessboardCornersSB(gray, (cols, rows),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok and max(gray.shape) > 1000:
        # Detect on native-resolution regions, then restore original pixels.
        h, w = gray.shape; cw, ch = (w+1)//2, (h*2+2)//3
        for y in sorted({0, (h-ch)//2, h-ch}):
            for x in sorted({0, (w-cw)//2, w-cw}):
                ok, points = cv2.findChessboardCornersSB(gray[y:y+ch, x:x+cw], (cols, rows),
                                                       flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    points = points.copy(); points[..., 0] += x; points[..., 1] += y
                    break
            if ok: break
    return points.astype(np.float32) if ok else None


def align_corner_orders(object_points, left, right, k0, d0, k1, d1):
    """Resolve checkerboard 180-degree ordering with consensus across varied poses."""
    candidates = []
    for obj, a, b in zip(object_points, left, right):
        ok, rv0, tv0 = cv2.solvePnP(obj, a, k0, d0)
        if not ok: raise ValueError('标定板位姿求解失败。')
        r0 = cv2.Rodrigues(rv0)[0]
        view = []
        for points in (b, b[::-1].copy()):
            ok, rv1, tv1 = cv2.solvePnP(obj, points, k1, d1)
            if not ok: raise ValueError('标定板位姿求解失败。')
            r = cv2.Rodrigues(rv1)[0] @ r0.T
            t = (tv1-r @ tv0).ravel()
            view.append((r, t))
        candidates.append(view)
    def distance(first, second):
        angle = np.arccos(np.clip((np.trace(first[0].T @ second[0])-1)/2, -1, 1))
        translation = np.linalg.norm(first[1]-second[1]) / max(.1, np.linalg.norm(first[1]))
        return angle + translation
    anchors = [item for view in candidates for item in view]
    scores = [np.median([min(distance(anchor, item) for item in view) for view in candidates]) for anchor in anchors]
    anchor = anchors[int(np.argmin(scores))]
    # NumPy comparisons yield np.bool_, which the JSON encoder cannot write.
    reversed_views = [bool(distance(anchor, view[1]) < distance(anchor, view[0])) for view in candidates]
    return [b[::-1].copy() if reverse else b for b, reverse in zip(right, reversed_views)], reversed_views


def corner_displacement(before, after, *, reversible=False):
    """Motion screening only; unmarked checkerboards can reverse enumeration."""
    before = np.asarray(before).reshape(-1, 2); after = np.asarray(after).reshape(-1, 2)
    def rms(points):
        return float(np.sqrt(np.mean(np.sum((before-points)**2, axis=1))))
    return min(rms(after), rms(after[::-1])) if reversible else rms(after)


def upright_checker_corners(points, cols, rows):
    """Choose upper-left enumeration for a plumb board in an upright image."""
    points = np.asarray(points).copy(); grid = points.reshape(rows, cols, 2)
    reverse = bool(grid[-1, :, 1].mean() < grid[0, :, 1].mean())
    if reverse:
        points = points[::-1].copy(); grid = points.reshape(rows, cols, 2)
    across = grid[:, -1].mean(axis=0)-grid[:, 0].mean(axis=0)
    down = grid[-1].mean(axis=0)-grid[0].mean(axis=0)
    if across[0] <= abs(across[1]) or down[1] <= abs(down[0]):
        raise ValueError('竖直棋盘方向不明确：请保持相机画面正立，按标定的列数横向摆放，棋盘列水平、行竖直向下；不可旋转90°摆放。')
    return points, reverse


def stereo_board_pose(obj, ca, cb, data, *, reversible=False):
    """Estimate only the board pose with BOTH cameras; all calibration stays fixed.

    A small planar board's monocular pose can extrapolate poorly into the other
    view. Seed from each camera, handle checker ordering, and verify both fits.
    """
    from scipy.optimize import least_squares
    obj = np.asarray(obj, dtype=np.float64)
    ca = np.asarray(ca, dtype=np.float64).reshape(-1, 2)
    cb = np.asarray(cb, dtype=np.float64).reshape(-1, 2)
    ks = [np.asarray(k, dtype=np.float64) for k in data['K']]
    ds = [np.asarray(d, dtype=np.float64) for d in data['distortion']]
    rotation = np.asarray(data['R_cam0_to_cam1'], dtype=np.float64)
    translation = np.asarray(data['t_cam0_to_cam1_m'], dtype=np.float64).reshape(3, 1)
    def projections(x):
        r0 = cv2.Rodrigues(x[:3])[0]; t0 = x[3:].reshape(3, 1)
        return (cv2.projectPoints(obj, x[:3], t0, ks[0], ds[0])[0].reshape(-1, 2),
                cv2.projectPoints(obj, cv2.Rodrigues(rotation@r0)[0], rotation@t0+translation, ks[1], ds[1])[0].reshape(-1, 2))
    best = None
    for reverse in ([False, True] if reversible else [False]):
        right = cb[::-1] if reverse else cb
        seeds = []
        for camera, points in enumerate((ca, right)):
            ok, rv, tv = cv2.solvePnP(obj, points, ks[camera], ds[camera])
            if not ok: continue
            if camera:
                rv = cv2.Rodrigues(rotation.T@cv2.Rodrigues(rv)[0])[0]
                tv = rotation.T@(tv-translation)
            seeds.append(np.r_[rv.ravel(), tv.ravel()])
        def residual(x):
            left_prediction, right_prediction = projections(x)
            return np.r_[(left_prediction-ca).ravel(), (right_prediction-right).ravel()]
        for seed in seeds:
            fitted = least_squares(residual, seed, max_nfev=200, x_scale='jac')
            if not fitted.success or not np.isfinite(fitted.x).all(): continue
            r0 = cv2.Rodrigues(fitted.x[:3])[0]; t0 = fitted.x[3:]
            xyz0 = obj@r0.T+t0; xyz1 = xyz0@rotation.T+translation.ravel()
            if np.any(xyz0[:, 2] <= 0) or np.any(xyz1[:, 2] <= 0): continue
            predicted = projections(fitted.x)
            errors = [np.linalg.norm(predicted[0]-ca, axis=1), np.linalg.norm(predicted[1]-right, axis=1)]
            score = float(np.sqrt(np.mean(np.concatenate(errors)**2)))
            if best is None or score < best[0]:
                quality = dict(method='fixed_calibration_stereo_board_pose',
                               per_camera_median_px=[float(np.median(e)) for e in errors],
                               per_camera_rms_px=[float(np.sqrt(np.mean(e**2))) for e in errors],
                               per_camera_p95_px=[float(np.percentile(e, 95)) for e in errors],
                               joint_point_rms_px=score, cam1_order_reversed=reverse)
                best = score, fitted.x, quality
    if best is None: raise ValueError('双路标定板位姿求解失败：没有两路均为正深度且收敛的解。')
    return best[1][:3].reshape(3, 1), best[1][3:].reshape(3, 1), best[2]


def calibrate_views(object_points, left, right, size, align_orders=True, *, intrinsics=None):
    if len(left) < 12:
        raise ValueError(f'只有 {len(left)} 对合格标定图，至少需要 12 对不同位置和倾角。')
    if intrinsics is None:
        ret0, k0, d0, rv0, tv0 = cv2.calibrateCamera(object_points, left, size, None, None)
        ret1, k1, d1, rv1, tv1 = cv2.calibrateCamera(object_points, right, size, None, None)
    else:
        k0, k1 = [np.asarray(k, dtype=np.float64).copy() for k in intrinsics['K']]
        d0, d1 = [np.asarray(d, dtype=np.float64).copy() for d in intrinsics['distortion']]
        def fit_views(points, k, d):
            rvs, tvs, errors = [], [], []
            for obj, pts in zip(object_points, points):
                ok, rv, tv = cv2.solvePnP(obj, pts, k, d)
                if not ok: raise ValueError('复用内参时标定板位姿求解失败。')
                rvs.append(rv); tvs.append(tv)
                projected = cv2.projectPoints(obj, rv, tv, k, d)[0]
                errors.extend(np.sum((projected.reshape(-1, 2)-pts.reshape(-1, 2))**2, axis=1).tolist())
            return float(np.sqrt(np.mean(errors))), rvs, tvs
        ret0, rv0, tv0 = fit_views(left, k0, d0)
        ret1, rv1, tv1 = fit_views(right, k1, d1)
    print(f'单路拟合 RMS cam0/cam1={ret0:.3f}/{ret1:.3f} px；'
          f'主点=({k0[0,2]:.1f},{k0[1,2]:.1f})/({k1[0,2]:.1f},{k1[1,2]:.1f})；'
          f'内参模式={"复用已有内参" if intrinsics is not None else "本次录像估计"}', flush=True)
    original_right = right
    if align_orders:
        right, reversed_views = align_corner_orders(object_points, left, right, k0, d0, k1, d1)
    else:
        reversed_views = [False]*len(right)
    ret, k0, d0, k1, d1, r, t, e, f = cv2.stereoCalibrate(
        object_points, left, right, k0, d0, k1, d1, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))
    errors = []
    for k, d, rvs, tvs, image_points in [(k0, d0, rv0, tv0, left), (k1, d1, rv1, tv1, original_right)]:
        for obj, points, rv, tv in zip(object_points, image_points, rvs, tvs):
            projected, _ = cv2.projectPoints(obj, rv, tv, k, d)
            errors.extend(np.linalg.norm(projected.reshape(-1, 2) - points.reshape(-1, 2), axis=1).tolist())
    baseline = float(np.linalg.norm(t))
    if baseline < .02:
        raise ValueError('估计基线小于 2 cm，请检查角点匹配、左右顺序和格子尺寸。')
    if ret > 3:
        raise ValueError(f'双目重投影 RMS={ret:.2f} px，标定质量不足。单路拟合={ret0:.2f}/{ret1:.2f} px；'
                         '单路误差小不代表内参可靠，请检查姿态倾角与位置覆盖、两路对应和停稳情况。'
                         '若仅机位改变且镜头/焦点/分辨率不变，可选择已验证标定复用内参；否则补拍不同倾角的停稳姿态。')
    return dict(K=[k0.tolist(), k1.tolist()], distortion=[d0.ravel().tolist(), d1.ravel().tolist()],
                R_cam0_to_cam1=r.tolist(), t_cam0_to_cam1_m=t.ravel().tolist(),
                cam1_corner_order_reversed=reversed_views,
                quality=dict(intrinsic_rms_px=[ret0, ret1], stereo_rms_px=ret,
                             intrinsics_mode='reused_fixed' if intrinsics is not None else 'estimated_from_session',
                             reprojection_median_px=float(np.median(errors)),
                             reprojection_p95_px=float(np.percentile(errors, 95)), baseline_m=baseline,
                             acceptance='diagnostic_only; validate held-out images and known distances'))


def calibrate(session, output, cols, rows, square_m, every=30, max_views=60, max_delta_ms=5, *, board_spec=None, intrinsics_from=None):
    from .boards import BoardSpec, Target
    meta = session_info(session)
    prior_intrinsics = load_calibration(intrinsics_from, meta) if intrinsics_from else None
    spec = board_spec or BoardSpec(cols=cols, rows=rows, square_m=square_m)
    target = Target(spec)
    left, right, objects, selected, signatures = [], [], [], [], []
    size = None
    checked = 0; best_corners = [0, 0]; best_common = 0
    detected_frames = [0, 0]
    rejected_motion = 0; rejected_motion_visibility = 0
    for pair in calibration_samples(session, max_delta_ms, every):
        n = pair['pair_number']
        a, b = pair['left_rgb'], pair['right_rgb']
        current = (a.shape[1], a.shape[0])
        if current != (b.shape[1], b.shape[0]) or (size and current != size):
            raise ValueError('标定图分辨率不一致。')
        size = current
        detections = [target.detect(a), target.detect(b)]
        checked += 1
        counts = [0 if found is None else len(found[2]) for found in detections]
        for i, count in enumerate(counts):
            best_corners[i] = max(best_corners[i], count)
            detected_frames[i] += int(count > 0)
        common_count = len(np.intersect1d(detections[0][2], detections[1][2])) if all(found is not None for found in detections) else 0
        best_common = max(best_common, common_count)
        if checked == 1 or checked % 10 == 0:
            print(f'检测进度：已检查 {checked} 对；本帧 cam0/cam1 角点 {counts[0]}/{counts[1]}，共同 {common_count}；已选 {len(left)} 对；移动帧剔除 {rejected_motion} 对', flush=True)
        detected = target.match(*detections)
        if detected is None: continue
        obj, ca, cb, ids = detected
        motions = []
        for detection, later in zip(detections, pair['stability_rgb']):
            after = target.detect(later)
            if after is None: break
            common, ia, ib = np.intersect1d(detection[2], after[2], return_indices=True)
            if len(common) < 12: break
            motions.append(corner_displacement(detection[1][ia], after[1][ib],
                                               reversible=spec.kind == 'checkerboard'))
        if len(motions) != 2:
            rejected_motion_visibility += 1; continue
        if max(motions) >= 2:
            rejected_motion += 1; continue
        # Reject near-identical views in both cameras; pose diversity still needs inspection.
        signature = np.concatenate([ca.ravel(), cb.ravel()])
        key = tuple(ids)
        if any(key == old_key and np.sqrt(np.mean((signature - old) ** 2)) < 12 for old_key, old in signatures): continue
        signatures.append((key, signature))
        left.append(ca); right.append(cb); objects.append(obj)
        selected.append(dict(left_frame=pair['left_frame_index'], right_frame=pair['right_frame_index'], corner_ids=ids.tolist(), motion_6_frames_px=motions))
        print(f'标定板：已选 {len(left)} 对；停稳检查位移 cam0/cam1={motions[0]:.2f}/{motions[1]:.2f} px', flush=True)
        if len(left) >= max_views: break
    diagnostics = dict(checked_pairs=checked, best_corner_counts=best_corners, best_common_corners=best_common,
                       detected_frame_counts=detected_frames, accepted_views=len(left), board=spec.to_dict(),
                       rejected_motion_pairs=rejected_motion, rejected_motion_visibility_pairs=rejected_motion_visibility,
                       motion_check=dict(frame_interval=6, maximum_rms_displacement_px=2), selected_frames=selected,
                       session=str(Path(session).resolve()))
    write_json(Path(output).with_suffix('.detection.json'), diagnostics)
    if len(left) < 12:
        raise ValueError(f'只有 {len(left)} 对合格标定图，至少需要 12 对不同位置和倾角。'
                         f'剔除移动帧 {rejected_motion} 对、停稳检查角点不足 {rejected_motion_visibility} 对。'
                         f'已检查 {checked} 对，cam0/cam1 最多检测 {best_corners[0]}/{best_corners[1]} 个角点，'
                         f'最多共同 {best_common} 个。相机不动，把板拿近、保持静止、避免反光；标定时无需全身入镜。'
                         '如果近距离仍没有角点，请检查板型/字典/排列参数。详情见 .detection.json。')
    result = calibrate_views(objects, left, right, size, align_orders=spec.kind == 'checkerboard', intrinsics=prior_intrinsics)
    result.update(schema='stereo_calibration/1.0', unit='m', image_size=list(size),
                  camera_ids=camera_ids(meta), board=spec.to_dict(),
                  selected_frames=selected, source_session=str(Path(session).resolve()),
                  coordinate_system='cam0_optical_right_down_forward',
                  world_from_cam0=None, world_coordinate_system=None)
    if prior_intrinsics is not None:
        result['intrinsics_reuse'] = dict(source=str(Path(intrinsics_from).resolve()),
                                         source_calibration_id=prior_intrinsics['calibration_id'],
                                         condition='Same physical camera order, resolution, lens and focus; only extrinsics estimated from new session.')
    digest = hashlib.sha256(str(result).encode()).hexdigest()[:16]
    result['calibration_id'] = 'stereo-' + digest
    write_json(output, result)
    print(f'标定已保存：{output}；双目 RMS={result["quality"]["stereo_rms_px"]:.3f} px', flush=True)
    return result


def set_world(calibration, session, output, pair_number, cols, rows, square_m, *, board_spec=None, board_height_m=0., placement='horizontal', vertical_origin_height_m=None):
    """Anchor Z-up world using a physically level or plumb board and measured height."""
    from .boards import BoardSpec, Target
    if not np.isfinite(board_height_m) or not 0 <= board_height_m <= 3:
        raise ValueError('水平板印刷面离地高度必须在 0–3 米之间。')
    data = load_calibration(calibration, session_info(session))
    spec = board_spec or BoardSpec(cols=cols, rows=rows, square_m=square_m)
    if placement not in ('horizontal', 'vertical'):
        raise ValueError('世界标定摆放方式必须为 horizontal 或 vertical。')
    if placement == 'vertical':
        if board_height_m != 0: raise ValueError('竖直板使用图案上边缘高度，不可同时填写水平板高度。')
        pattern_height_m = (spec.rows+1 if spec.kind == 'checkerboard' else spec.rows)*spec.square_m
        if vertical_origin_height_m is None or not np.isfinite(vertical_origin_height_m) or not pattern_height_m <= vertical_origin_height_m <= 3:
            raise ValueError('请实测竖直板棋盘格图案上边缘离地高度（不含外围白边）；图案须在地面以上，高度不超过 3 米。')
    elif vertical_origin_height_m is not None:
        raise ValueError('水平板模式不可填写竖直图案上边缘高度。')
    target = Target(spec)
    saved_spec = data.get('board', {})
    if saved_spec.get('kind') and saved_spec != spec.to_dict():
        raise ValueError('地面板规格必须与相机标定板完全一致，包括板型、格数、尺寸、字典和排列版本。')
    ca = cb = chosen = None
    checked = 0
    best_counts = np.zeros(3, dtype=int)
    for i, candidate in enumerate(pairs(session, 5)):
        if pair_number >= 0 and i != pair_number: continue
        if pair_number < 0 and i % 5: continue
        detected = target.common(candidate['left_rgb'], candidate['right_rgb'])
        checked += 1
        counts = target.last_counts
        best_counts = np.maximum(best_counts, counts)
        if checked == 1 or i % 100 == 0:
            print(f'世界标定板检测（{placement}）：配对 {i}，cam0/cam1 角点 {counts[0]}/{counts[1]}，共同 {counts[2]}（至少需 12 个且覆盖二维区域）', flush=True)
        if detected is not None:
            obj, ca, cb, ids = detected
            chosen = candidate; pair_number = i; break
        if pair_number >= 0: break
    if chosen is None:
        report = dict(checked_pairs=checked, maximum_cam0_corners=int(best_counts[0]),
                      maximum_cam1_corners=int(best_counts[1]), maximum_common_corners=int(best_counts[2]),
                      board=spec.to_dict(), session=str(session), status='no_usable_pair', placement=placement,
                      horizontal_face_height_m=board_height_m if placement == 'horizontal' else None,
                      vertical_pattern_top_height_m=vertical_origin_height_m)
        write_json(Path(output).with_suffix('.world_detection.json'), report)
        raise ValueError(f'没有合格世界标定板配对：检查 {checked} 对，cam0/cam1 最多角点 {best_counts[0]}/{best_counts[1]}，'
                         f'共同角点最多 {best_counts[2]}。需至少 12 个共同角点且覆盖二维区域；'
                         '若某一路接近 0，请检查该路是否视角太斜、反光、遮挡或图案细节不足。保持相机不动，调整标定板位置/高度后短录重试。')
    # Printed board X-right/Y-down normal points into its face. Reverse Y and Z
    # with a proper 180-degree rotation so a face-up ground board yields Z-up.
    checker_order_reversed = False
    if placement == 'vertical':
        if spec.kind == 'checkerboard':
            ca, checker_order_reversed = upright_checker_corners(ca, spec.cols, spec.rows)
        # Board +Y goes vertically down; its inward normal becomes world +Y.
        # This is a proper rotation (det=+1), preserving right-handed axes.
        obj = obj @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        # Checkerboard origin is an inner corner, one square below the measured
        # pattern top edge; ChArUco origin is the pattern's outer corner.
        obj[:, 2] += vertical_origin_height_m-(spec.square_m if spec.kind == 'checkerboard' else 0.)
        axis_convention = 'world_X=board_X, world_Y=board_Z, world_Z=-board_Y'
    else:
        obj = obj * np.array([1, -1, -1], dtype=np.float32)
        obj[:, 2] += board_height_m
        axis_convention = 'world_X=board_X, world_Y=-board_Y, world_Z=-board_Z'
    # A level face-up board can sit above the floor. Its measured printed-face
    # height anchors the floor origin directly below the board origin.
    rv, tv, pose_quality = stereo_board_pose(obj, ca, cb, data, reversible=spec.kind == 'checkerboard')
    r, _ = cv2.Rodrigues(rv)
    camera_from_world = np.eye(4); camera_from_world[:3, :3] = r; camera_from_world[:3, 3] = tv.ravel()
    error = pose_quality['per_camera_median_px'][1]
    print(f'双路位姿校验：中位误差 cam0/cam1={pose_quality["per_camera_median_px"][0]:.2f}/{error:.2f} px；'
          f'RMS={pose_quality["joint_point_rms_px"]:.2f} px（相机参数固定）', flush=True)
    worst_error = max(pose_quality['per_camera_median_px']+pose_quality['per_camera_rms_px'])
    if worst_error > 3:
        write_json(Path(output).with_suffix('.world_validation.json'),
                   dict(status='stereo_inconsistent', pair_number=pair_number,
                        common_corners=len(ids), cam1_median_reprojection_px=error,
                        maximum_allowed_px=3, pose_quality=pose_quality, calibration=str(calibration), session=str(session),
                        placement=placement, horizontal_face_height_m=board_height_m if placement == 'horizontal' else None,
                        vertical_pattern_top_height_m=vertical_origin_height_m,
                        note='Changing the world origin height or assumed gravity direction cannot repair stereo inconsistency.'))
        raise ValueError(f'标定板已识别，但双路联合位姿校验不合格：最大单路中位误差/RMS={worst_error:.2f} px（上限 3 px）。'
                         '这不是板子水平/竖直程度或离地高度造成的；核对物理相机对应、机位/焦点是否改变，'
                         '若均未改变，需要重新验证或重新进行相机内外参标定。更换 USB 身份绑定不能修复几何参数。')
    data['world_from_cam0'] = np.linalg.inv(camera_from_world).tolist()
    data['world_coordinate_system'] = 'world_right_handed_X_forward_Y_left_Z_up'
    data['ground_board'] = dict(pair_number=pair_number, cam1_median_reprojection_px=error,
                               pose_quality=pose_quality,
                               board=spec.to_dict(), corner_ids=ids.tolist(),
                               printed_face_height_m=board_height_m if placement == 'horizontal' else None,
                               pattern_top_edge_height_m=vertical_origin_height_m,
                               height_reference='pattern_top_edge_excluding_white_margin' if placement == 'vertical' else 'printed_face',
                               origin_reference='first_inner_corner_projected_to_floor' if spec.kind == 'checkerboard' else 'pattern_outer_origin_projected_to_floor',
                               checker_cam0_order_reversed=checker_order_reversed,
                               placement='level_horizontal_face_up' if placement == 'horizontal' else 'vertical_rows_down_columns_level',
                               axis_convention=axis_convention,
                               note='Physical level/plumb placement and measured height are required; not inferred from images.')
    data['calibration_id'] += '-world-' + hashlib.sha256(str(data['world_from_cam0']).encode()).hexdigest()[:8]
    write_json(output, data)
    preview = cv2.cvtColor(chosen['left_rgb'], cv2.COLOR_RGB2BGR)
    axes = np.array([[0, 0, 0], [.15, 0, 0], [0, .15, 0], [0, 0, .15]], dtype=np.float64)
    projected, _ = cv2.projectPoints(axes, rv, tv, np.array(data['K'][0]), np.array(data['distortion'][0]))
    pts = np.round(projected.reshape(-1, 2)).astype(int)
    for i, (label, color) in enumerate([('X forward', (0, 0, 255)), ('Y left', (0, 200, 0)), ('Z up', (255, 100, 0))], 1):
        cv2.line(preview, tuple(pts[0]), tuple(pts[i]), color, 4)
        cv2.putText(preview, label, tuple(pts[i]), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    ok, encoded = cv2.imencode('.png', preview)
    if ok: encoded.tofile(str(Path(output).with_suffix('.axes.png')))


def load_calibration(path, meta=None):
    data = read_json(path)
    if data.get('schema') != 'stereo_calibration/1.0' or data.get('unit') != 'm':
        raise ValueError('标定格式或单位不正确。')
    if meta and camera_ids(meta) != data['camera_ids']:
        raise ValueError('录像相机路径/顺序与标定不一致。请核对 cam0/cam1；禁止静默交换。')
    if meta:
        mode = meta['requested_mode']
        if data['image_size'] != [mode['width'], mode['height']]:
            raise ValueError('录像分辨率与标定不一致。')
    for k in data['K']:
        if np.shape(k) != (3, 3) or not np.isfinite(k).all(): raise ValueError('相机内参无效。')
    r = np.array(data['R_cam0_to_cam1']); t = np.array(data['t_cam0_to_cam1_m'])
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .99:
        raise ValueError('双目旋转无效。')
    if t.shape != (3,) or not np.isfinite(t).all() or np.linalg.norm(t) < .02: raise ValueError('双目基线无效。')
    if data.get('world_from_cam0') is not None:
        if data.get('ground_board') and data['ground_board'].get('axis_convention') not in ('world_X=board_X, world_Y=-board_Y, world_Z=-board_Z', 'world_X=board_X, world_Y=board_Z, world_Z=-board_Y'):
            raise ValueError('旧版地面标定的板面法向约定已修正，请从原内外参 JSON 重新生成地面世界标定。')
        w = np.array(data['world_from_cam0'])
        if w.shape != (4, 4) or not np.isfinite(w).all() or not np.allclose(w[3], [0, 0, 0, 1]) or not np.allclose(w[:3, :3].T @ w[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(w[:3, :3]) < .99:
            raise ValueError('世界坐标变换无效。')
    return data


def rebind_calibration(calibration, session, output, old_indices):
    """Explicit physical-camera mapping; never infer identity from USB names."""
    import copy
    if list(old_indices) not in ([0, 1], [1, 0]):
        raise ValueError('相机映射必须为 0,1 或 1,0，需先通过画面确认物理相机身份。')
    if Path(output).exists(): raise ValueError('适配文件已存在，请使用新文件。')
    data = copy.deepcopy(load_calibration(calibration))
    meta = session_info(session)
    mode = meta['requested_mode']
    if data['image_size'] != [mode['width'],mode['height']]:
        raise ValueError('USB 适配不能改变标定分辨率。')
    original_ids = data['camera_ids'].copy(); original_id = data['calibration_id']
    if list(old_indices) == [1,0]:
        r = np.array(data['R_cam0_to_cam1']); t = np.array(data['t_cam0_to_cam1_m'])
        inverse_r = r.T; inverse_t = -inverse_r @ t
        data['K'] = data['K'][::-1]; data['distortion'] = data['distortion'][::-1]
        data['R_cam0_to_cam1'] = inverse_r.tolist(); data['t_cam0_to_cam1_m'] = inverse_t.tolist()
        if data.get('world_from_cam0') is not None:
            old_from_new = np.eye(4); old_from_new[:3,:3] = inverse_r; old_from_new[:3,3] = inverse_t
            data['world_from_cam0'] = (np.array(data['world_from_cam0']) @ old_from_new).tolist()
        quality = data.get('quality',{})
        if 'intrinsic_rms_px' in quality: quality['intrinsic_rms_px'] = quality['intrinsic_rms_px'][::-1]
    data['camera_ids'] = camera_ids(meta)
    data['camera_rebind'] = dict(original_calibration=str(Path(calibration).resolve()), original_camera_ids=original_ids,
                                 new_session=str(Path(session).resolve()), new_camera_to_old_camera=list(old_indices),
                                 authorization='Explicit operator mapping; camera poses, lenses and focus must be unchanged.',
                                 old_diagnostics_order='selected_frames and historical ground_board refer to original session order')
    data['calibration_id'] = original_id+'-rebind-'+hashlib.sha256(str(data['camera_rebind']).encode()).hexdigest()[:8]
    write_json(output,data)
    print(f'相机身份适配已保存：{output}；新 cam0/cam1 对应原 {list(old_indices)}',flush=True)
    return data
