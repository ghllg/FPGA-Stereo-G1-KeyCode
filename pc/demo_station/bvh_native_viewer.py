"""MuJoCo native menus displaying UI-owned saved poses, without another clock."""
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np


def run(root):
    import mujoco
    import mujoco.viewer
    import glfw
    glfw.init();glfw.window_hint(glfw.FLOATING,glfw.TRUE);glfw.window_hint(glfw.AUTO_ICONIFY,glfw.FALSE)
    source=mujoco.MjModel.from_xml_path(str(root/'g1_realtime/vendor/gmr/unitree_g1/g1_mocap_29dof.xml'))
    scene=root/'g1_unitree_official/standalone/simulate_python/policy_scene.xml'
    model=mujoco.MjModel.from_xml_path(str(scene))
    names=[model.joint(i).name for i in range(1,model.njnt)]
    if (source.nq!=36 or model.nq!=36 or source.njnt!=30 or model.njnt!=30
            or set(names)!={source.joint(i).name for i in range(1,source.njnt)}):
        raise ValueError('Native scene must contain the same 29 G1 joints')
    source_joints=[source.joint(name).id for name in names]
    source_addresses=source.jnt_qposadr[source_joints].copy()
    target_addresses=model.jnt_qposadr[1:].copy()
    source_bodies=source.jnt_bodyid[[0,*source_joints]].copy()
    target_bodies=model.jnt_bodyid.copy()
    source_body_count=source.nbody
    del source
    data=mujoco.MjData(model)
    mailbox={'request':None};lock=threading.Lock();ended=threading.Event()
    def read():
        try:
            for line in sys.stdin:
                request=json.loads(line)
                with lock:mailbox['request']=request
        finally:ended.set()
    threading.Thread(target=read,daemon=True).start()
    while mailbox['request'] is None and not ended.wait(.01):pass
    if mailbox['request'] is None:return
    def apply(request):
        q=np.asarray(request['qpos'],float)
        if q.shape!=(model.nq,) or not np.isfinite(q).all():raise ValueError('Invalid saved qpos')
        data.qpos[:7]=q[:7];data.qpos[target_addresses]=q[source_addresses]
        data.qvel[:]=0;data.time=request['source_time'];mujoco.mj_forward(model,data)
        expected=np.asarray(request['body_pos'],float)
        # The old scene has different fixed visual bodies; compare all moving
        # bodies by joint name, including the floating pelvis, instead of row order.
        if (expected.shape!=(source_body_count,3)
                or not np.allclose(expected[source_bodies],data.xpos[target_bodies],atol=2e-5)):
            raise ValueError('Saved state does not match the native G1 model')
    apply(mailbox['request'])
    with mujoco.viewer.launch_passive(model,data,show_left_ui=True,show_right_ui=True) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;viewer.cam.trackbodyid=model.body('pelvis').id
        viewer.cam.distance=3.;viewer.cam.azimuth=130.;viewer.cam.elevation=-20.
        viewer.opt.geomgroup[3:]=0
        last_id=None
        while viewer.is_running() and not ended.is_set():
            with lock:request=mailbox['request']
            with viewer.lock():
                apply(request)
                viewer.cam.lookat[:]=[data.qpos[0],data.qpos[1],.8]
            if request['id']!=last_id:
                mode='PHYSICS REPLAY' if request['physics'] else 'REFERENCE ANIMATION'
                state='PLAYING' if request['playing'] else 'PAUSED'
                viewer.set_texts((None,None,'Motion Studio | FILE PLAYBACK',
                                 f"{mode} | {state}\n{request['position']:.3f} s | controls in Motion Studio"))
            viewer.sync()
            if request['id']!=last_id:
                canonical=data.qpos.copy();canonical[source_addresses]=data.qpos[target_addresses]
                print(json.dumps(dict(id=request['id'],session=request['session'],seq=request['seq'],
                    physics=request['physics'],playing=request['playing'],source_time=request['source_time'],
                    qpos=canonical.tolist(),scene=str(scene),scene_body_count=model.nbody,
                    matched_moving_bodies=len(target_bodies))),flush=True)
                last_id=request['id']
            time.sleep(.01)


if __name__=='__main__':
    try:run(Path(sys.argv[1]))
    except Exception as error:
        print(json.dumps(dict(error=str(error))),flush=True)
        raise SystemExit(1)
