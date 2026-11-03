"""Optional reference conditioning. Never modifies actual physics state."""
import numpy as np
from scipy.spatial.transform import Rotation


def limit_pelvis_tilt(qpos, degrees=5):
    q=np.asarray(qpos,dtype=float).copy()
    roll,pitch,yaw=Rotation.from_quat(q[[4,5,6,3]]).as_euler('xyz')
    limit=np.deg2rad(degrees)
    q[3:7]=Rotation.from_euler('xyz',[np.clip(roll,-limit,limit),np.clip(pitch,-limit,limit),yaw]).as_quat()[[3,0,1,2]]
    return q
