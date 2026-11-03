"""Single-person COCO17 inference in original, unmirrored image pixels."""
from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
import numpy as np
from .io import COCO17, pairs, session_info, camera_ids, write_json


class PoseEstimator:
    def __init__(self, model=None, device='auto', confidence=.35, image_size=640, refine_person=False):
        config_dir = Path(__file__).resolve().parents[1] / 'models' / 'runtime_settings'
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
        import torch
        from ultralytics import YOLO, settings
        settings.update({'sync': False})
        if device == 'auto': device = '0' if torch.cuda.is_available() else 'cpu'
        if device != 'cpu' and not torch.cuda.is_available():
            raise ValueError('当前环境没有可用 CUDA，请选择 CPU 或安装 CUDA 版 PyTorch。')
        if model is None:
            model = Path(__file__).resolve().parents[1] / 'models' / 'yolo11n-pose.pt'
        model = Path(model).resolve(); model.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model), task='pose')
        self.model_id = model.name + ':' + hashlib.sha256(model.read_bytes()).hexdigest()[:16]
        self.device, self.confidence, self.image_size = device, confidence, image_size
        self.refine_person = refine_person

    def predict(self, images):
        # Ultralytics ndarray input is BGR; pipeline images are RGB.
        results = self.model.predict([np.ascontiguousarray(im[:, :, ::-1]) for im in images],
                    device=self.device, imgsz=self.image_size, conf=self.confidence, verbose=False)
        # A second pass spends the input resolution on the person instead of the whole room.
        # Full-image person count remains the authority: never hide multi-person ambiguity by cropping.
        refined = {}
        crops = []; crop_metadata = []
        if self.refine_person:
            for i, (image, result) in enumerate(zip(images, results)):
                if len(result.boxes) != 1: continue
                x0, y0, x1, y1 = result.boxes.xyxy[0].cpu().numpy()
                pad_x, pad_y = (x1-x0)*.2, (y1-y0)*.12
                x0, y0 = max(0, int(x0-pad_x)), max(0, int(y0-pad_y))
                x1, y1 = min(image.shape[1], int(x1+pad_x)+1), min(image.shape[0], int(y1+pad_y)+1)
                if min(x1-x0, y1-y0) < 32: continue
                crops.append(np.ascontiguousarray(image[y0:y1, x0:x1, ::-1]))
                crop_metadata.append((i, x0, y0))
            if crops:
                crop_results = self.model.predict(crops, device=self.device, imgsz=self.image_size,
                                                   conf=self.confidence, verbose=False)
                for (i, x0, y0), result in zip(crop_metadata, crop_results):
                    if len(result.boxes) == 1 and result.keypoints is not None:
                        data = result.keypoints.data.cpu().numpy().copy()
                        if data.shape == (1, 17, 3):
                            data[:, :, 0] += x0; data[:, :, 1] += y0
                            refined[i] = data
        output = []
        for i, (image, result) in enumerate(zip(images, results)):
            count = len(result.boxes)
            # Fixed single-person project: ambiguity is rejected rather than selecting different people.
            data = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 3))
            if i in refined: data = refined[i]
            points = []
            for j in range(17):
                x, y, score = data[0, j] if count == 1 and data.shape == (1, 17, 3) else (0, 0, 0)
                valid = bool(np.isfinite([x, y, score]).all() and score >= self.confidence and
                             0 <= x < image.shape[1] and 0 <= y < image.shape[0])
                points.append(dict(xy=[float(x), float(y)] if valid else None,
                                   confidence=float(score) if np.isfinite(score) else 0., valid=valid))
            output.append(dict(image_size=[image.shape[1], image.shape[0]], person_count=count,
                               status='single_person' if count == 1 else ('missing' if count == 0 else 'ambiguous_multiple_people'),
                               person_roi_refined=i in refined, joints=points))
        return output


def extract(session, output, model=None, device='auto', every=2, max_delta_ms=10, limit=None, image_size=640, refine_person=False):
    import json
    if every < 1: raise ValueError('抽帧步长至少为 1。')
    meta = session_info(session)
    estimator = PoseEstimator(model, device, image_size=image_size, refine_person=refine_person)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise ValueError('二维关键点输出已存在，请选择新文件，避免覆盖。')
    count = 0; started = time.perf_counter()
    manifest = dict(schema='pose2d_manifest/1.0', state='incomplete', source_session=str(Path(session).resolve()),
                    camera_ids=camera_ids(meta), joint_names=list(COCO17), model_id=estimator.model_id,
                    clock_id=f'windows_dshow:{Path(session).name}', unit='original_image_pixels',
                    mirrored=False, every=every, max_delta_ms=max_delta_ms, device=estimator.device,
                    timestamp_origin_ns=meta['session_origin_source_ns'], image_size=image_size,
                    person_roi_refinement=refine_person, rows=0)
    write_json(output.with_suffix('.meta.json'), manifest)
    try:
        with output.open('x', encoding='utf-8') as f:
            for n, pair in enumerate(pairs(session, max_delta_ms)):
                if n % every: continue
                if limit is not None and count >= limit: break
                begin = time.perf_counter()
                views = estimator.predict([pair['left_rgb'], pair['right_rgb']])
                row = dict(schema='pose2d/1.0', seq=count, joint_names=list(COCO17),
                           timestamp_ns=(pair['left_source_ns'] + pair['right_source_ns']) // 2,
                           model_id=estimator.model_id, pair_delta_ms=pair['right_minus_left_ms'],
                           inference_ms=(time.perf_counter() - begin) * 1000, views=views)
                for i, (view, prefix) in enumerate(zip(views, ('left', 'right'))):
                    view.update(camera_id=manifest['camera_ids'][i],
                                frame_index=pair[f'{prefix}_frame_index'],
                                timestamp_ns=pair[f'{prefix}_source_ns'])
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                count += 1
                if count % 10 == 0: print(f'人体识别：{count} 对；已耗时 {time.perf_counter()-started:.1f} 秒', flush=True)
        if not count: raise ValueError('没有满足时间阈值的双路配对帧。')
        manifest.update(state='complete', rows=count, elapsed_s=time.perf_counter()-started)
    finally:
        manifest['rows'] = count
        write_json(output.with_suffix('.meta.json'), manifest)
    print(f'二维关键点已保存：{output}', flush=True)
