"""Offline review movie: original two views, model points and fitted estimated skeleton."""
from __future__ import annotations
import json
from pathlib import Path
import av
import cv2
import numpy as np
from .calibration import load_calibration
from .io import read_json

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


def make_preview(pose_file, skeleton_file, calibration_file, output, fps=30):
    output = Path(output)
    if output.exists(): raise ValueError('预览视频已存在，请选择新文件。')
    poses = [json.loads(s) for s in Path(pose_file).read_text(encoding='utf-8').splitlines() if s.strip()]
    skeletons = [json.loads(s) for s in Path(skeleton_file).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(poses) != len(skeletons): raise ValueError('二维/三维帧数不一致。')
    meta = read_json(Path(pose_file).with_suffix('.meta.json')); cal = load_calibration(calibration_file)
    source = Path(meta['source_session'])
    times = np.array([(p['timestamp_ns']-poses[0]['timestamp_ns'])/1e9 for p in poses])
    inverse_world = np.linalg.inv(cal['world_from_cam0'])
    r = np.array(cal['R_cam0_to_cam1']); t = np.array(cal['t_cam0_to_cam1_m'])
    k = [np.array(a) for a in cal['K']]; d = [np.array(a) for a in cal['distortion']]
    width, height = cal['image_size']; scale = np.array([640/width, 360/height])
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 420))
    if not writer.isOpened(): raise ValueError('无法创建 MP4 预览文件。')
    class Reader:
        def __init__(self, container): self.frames=iter(container.decode(video=0)); self.index=-1; self.last=None
        def at(self, index):
            if index < self.index: raise ValueError('预览源帧号必须递增。')
            while self.index < index:
                frame = next(self.frames); self.index += 1
                if self.index == index: self.last=cv2.resize(frame.to_ndarray(format='bgr24'), (640, 360))
            return self.last.copy()
    try:
        with av.open(str(source/'left.mkv')) as left, av.open(str(source/'right.mkv')) as right:
            readers = [Reader(left), Reader(right)]
            for sample in np.arange(int(np.floor(times[-1]*fps))+1)/fps:
                hi = int(np.searchsorted(times, sample).clip(0, len(times)-1)); lo=max(0, hi-1)
                i=lo if abs(times[lo]-sample) < abs(times[hi]-sample) else hi
                row, packet = poses[i], skeletons[i]
                images = [readers[cam].at(row['views'][cam]['frame_index']) for cam in range(2)]
                points=np.array([j['xyz'] if j['valid'] else [np.nan]*3 for j in packet['joints']])
                finite=np.isfinite(points).all(axis=1)
                optical=points@inverse_world[:3,:3].T+inverse_world[:3,3]
                for cam, image in enumerate(images):
                    xy=cv2.projectPoints(optical[finite], np.zeros(3) if cam==0 else cv2.Rodrigues(r)[0],
                                         np.zeros(3) if cam==0 else t,k[cam],d[cam])[0].reshape(-1,2)*scale if np.any(finite) else np.empty((0,2))
                    projected={int(j):tuple(np.round(p).astype(int)) for j,p in zip(np.flatnonzero(finite),xy)}
                    for a,b in EDGES:
                        if a in projected and b in projected:cv2.line(image,projected[a],projected[b],(220,50,230),2)
                    for joint in row['views'][cam]['joints']:
                        if joint['valid']:cv2.circle(image,tuple(np.round(np.array(joint['xy'])*scale).astype(int)),3,(0,255,0),-1)
                    cv2.putText(image,f'CAM{cam}  content aligned / original pixels',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                panel=np.full((360,640,3),28,np.uint8)
                view=np.array([[.8,-.6,0],[.08,.1,-1.]])
                positions=(points-np.array([.15,.9,0.]))@view.T*175+np.array([320,320])
                for a,b in EDGES:
                    if finite[a] and finite[b]:cv2.line(panel,tuple(positions[a].astype(int)),tuple(positions[b].astype(int)),(80,200,255),3)
                for j in np.flatnonzero(finite):cv2.circle(panel,tuple(positions[j].astype(int)),4,(230,230,230),-1)
                cv2.line(panel,(40,320),(600,320),(80,80,80),1)
                cv2.putText(panel,'OFFLINE ESTIMATED 3D - not ground truth',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1)
                missing=[str(j) for j in (5,6,7,8,9,10,11,12,13,14,15,16) if not finite[j]]
                cv2.putText(panel,'Missing body joints: '+(','.join(missing) or 'none'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,80,255) if missing else (0,220,0),1)
                frame=np.full((420,1920,3),20,np.uint8);frame[:360]=np.hstack(images+[panel])
                cv2.putText(frame,f't={sample:.2f}s  Green: model 2D | Pink: fitted 3D projection | Boundary loss and gaps are not invented',(15,395),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1)
                writer.write(frame)
    finally:writer.release()
    print(f'原视频与估计骨架对照预览：{output}',flush=True)
