"""File-playback data, virtual cameras and an explicit temporal completion demo."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline


SCHEMA = 'bvh_playback/1'
BODY_NAMES = ['Head', None, None, None, None, 'LeftArm', 'RightArm',
              'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
              'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot']
BODY_EDGES = [(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
              (11,12),(11,13),(13,15),(12,14),(14,16)]


def virtual_cameras(body):
    """Two fixed pinhole cameras; Z up, metres. No physical calibration implied."""
    left = np.nanmedian(body[:,5]-body[:,6], axis=0)
    left[2] = 0
    left /= np.linalg.norm(left)
    front = np.cross(left, [0.,0.,1.])
    target = np.nanmedian(body[:,11:13], axis=(0,1))
    target[2] = .95
    intrinsic = np.array([[1050.,0,640],[0,1050.,480],[0,0,1]])
    matrices = []
    for offset in (-.65,.65):
        eye = target + 3.5*front + offset*left
        forward = target-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.,0.,1.]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.array([right,down,forward])
        matrices.append(intrinsic @ np.column_stack((rotation, -rotation@eye)))
    return np.asarray(matrices)


def project(points, matrices):
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1],1))), axis=-1)
    camera = np.einsum('cij,tkj->tcki', matrices, homogeneous)
    with np.errstate(invalid='ignore', divide='ignore'):
        pixels = camera[...,:2] / camera[...,2:]
    valid = np.isfinite(pixels).all(-1) & (camera[...,2] > 0)
    valid &= (pixels[...,0]>=0) & (pixels[...,0]<1280) & (pixels[...,1]>=0) & (pixels[...,1]<960)
    return pixels, valid


def triangulate(pixels, valid, matrices):
    result = np.full((len(pixels),pixels.shape[2],3),np.nan)
    for frame,joint in np.argwhere(valid.all(axis=1)):
        rows = []
        for camera in range(2):
            u,v = pixels[frame,camera,joint]
            p = matrices[camera]
            rows.extend((u*p[2]-p[0], v*p[2]-p[1]))
        _,_,vh = np.linalg.svd(rows)
        if abs(vh[-1,3])>1e-10:result[frame,joint] = vh[-1,:3]/vh[-1,3]
    return result


def complete_short_gaps(observed, times, max_gap=.4):
    """Offline local cubic interpolation of observations only, never ground truth."""
    completed = observed.copy()
    estimated = np.zeros(observed.shape[:2],bool)
    for joint in range(observed.shape[1]):
        present = np.flatnonzero(np.isfinite(observed[:,joint]).all(-1))
        for index,(lo,hi) in enumerate(zip(present[:-1],present[1:])):
            if hi-lo<=1 or times[hi]-times[lo]>max_gap+1e-8:continue
            context = present[max(0,index-1):min(len(present),index+3)]
            completed[lo+1:hi,joint] = CubicSpline(times[context],observed[context,joint])(times[lo+1:hi])
            estimated[lo+1:hi,joint] = True
    return completed,estimated


def build_skeleton_tracks(human, names, times):
    body = np.full((len(times),17,3),np.nan)
    for joint,name in enumerate(BODY_NAMES):
        if name is not None:body[:,joint] = human[:,names.index(name)]
    matrices = virtual_cameras(body)
    pixels,valid = project(body,matrices)
    masked = np.zeros(valid.shape,bool)
    duration = times[-1]
    for fraction,joint in ((.33,9),(.68,8)):
        window = (times>=duration*fraction) & (times<duration*fraction+.24)
        masked[window,:,joint] = valid[window,:,joint]
    visible = valid & ~masked
    # Remove masked coordinates too, so downstream code cannot use hidden truth.
    observations = np.where(visible[...,None],pixels,np.nan)
    raw = triangulate(observations,visible,matrices)
    completed,estimated = complete_short_gaps(raw,times)
    error = np.linalg.norm(completed-body,axis=-1)
    stats = dict(masked_joint_frames=int(masked.any(1).sum()),
                 completed_joint_frames=int(estimated.sum()),
                 completion_mean_cm=float(np.mean(error[estimated])*100) if estimated.any() else None,
                 completion_max_cm=float(np.max(error[estimated])*100) if estimated.any() else None,
                 triangulation_max_m=float(np.nanmax(np.linalg.norm(raw-body,axis=-1))))
    return dict(body=body,projections=observations,visible=visible,masked=masked,
                camera_matrices=matrices,observed=raw,completed=completed,estimated=estimated),stats


def pose_packet(points, estimated=None, edges=None):
    valid = np.isfinite(points).all(-1)
    return dict(joints=[dict(xyz=point.tolist() if valid[i] else None,valid=bool(valid[i]),
                            estimated=bool(estimated[i]) if estimated is not None else False)
                        for i,point in enumerate(points)], edges=edges or BODY_EDGES)


class PlaybackSession:
    def __init__(self,path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'session.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema')!=SCHEMA or self.manifest.get('state')!='complete':
            raise ValueError('请选择生成完成的文件回放会话')
        self.name = self.manifest['name']
        with np.load(self.path/'tracks.npz',allow_pickle=False) as data:
            self.tracks = {name:data[name] for name in data.files}
        self.times = self.tracks['times']
        self.duration = float(self.times[-1])
        self.dt = float(self.manifest['frame_time'])
        if len(self.times)<2 or not np.all(np.diff(self.times)>0):raise ValueError('会话时间轴无效')
        self.robot_modes = ['reference']
        if 'physics_qpos' in self.tracks:self.robot_modes.insert(0,'physics')

    def resolve(self,path):
        return self.path/Path(path)

    def index(self,stamp):
        return int(np.clip(np.searchsorted(self.times,stamp+1e-8,side='right')-1,0,len(self.times)-1))

    def robot_packet(self,stamp,mode):
        times = self.tracks[mode+'_times']
        index = int(np.clip(np.searchsorted(times,stamp+1e-8,side='right')-1,0,len(times)-1))
        return dict(qpos=self.tracks[mode+'_qpos'][index].tolist(),
                    body_pos=self.tracks[mode+'_body'][index].tolist(),
                    source_time=float(times[index]),seq=index,session=str(self.path),physics=mode=='physics')

    def video_frame(self,stamp):
        video = self.manifest.get('media',{}).get('overview')
        if not video:return None
        time = stamp+video.get('offset_seconds',0.)
        if time<0 or time>=video['frames']/video['fps']:return None
        return int(time*video['fps'])
