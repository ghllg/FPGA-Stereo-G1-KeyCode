"""Map this run's official viewer initial state to the UI mesh model; never step physics."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_mapping(root):
    import mujoco
    import numpy as np
    import yaml
    sys.path.insert(0, str(root/'g1_physical'))
    from physical import MODEL, load_mujoco_model
    folder = root/'g1_unitree_official/standalone/simulate_python'
    scene = folder/'policy_scene.xml'
    config = runpy.run_path(str(folder/'config.py'))
    if Path(config['ROBOT_SCENE']).resolve() != scene.resolve():
        raise ValueError('Official viewer scene differs from the released policy scene')
    entry = folder/'unitree_mujoco.py'
    original = root/'g1_unitree_official/unitree_mujoco-main/simulate_python/unitree_mujoco.py'
    if digest(entry) != digest(original):
        raise ValueError('Official simulator entry identity mismatch')
    meta = yaml.safe_load(MODEL.with_suffix('.yaml').read_text())
    control, timing = meta['control'], meta['timing']
    expected, _ = load_mujoco_model(meta['robot']['mjcf_path'], control['stiffness'], control['damping'], timing['physics_dt'])
    native = mujoco.MjModel.from_xml_path(str(scene))
    for field in ('body_mass','body_inertia','body_pos','body_quat','jnt_qposadr','jnt_type',
                  'geom_friction','jnt_stiffness','dof_damping','actuator_gainprm','actuator_biasprm',
                  'actuator_biastype','actuator_gaintype','actuator_ctrllimited','actuator_forcelimited','actuator_forcerange'):
        np.testing.assert_allclose(getattr(native,field),getattr(expected,field),rtol=1e-5,atol=1e-7,
                                   err_msg='Official model differs from released policy: '+field)
    names = [mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,native.njnt)]
    if names != meta['joint_names'] or native.nq != 36:
        raise ValueError('Official model joint order differs from released policy')
    render_path = root/'g1_realtime/vendor/gmr/unitree_g1/g1_mocap_29dof.xml'
    render = mujoco.MjModel.from_xml_path(str(render_path))
    render_names = [mujoco.mj_id2name(render,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,render.njnt)]
    if render.nq != native.nq or set(render_names) != set(names):
        raise ValueError('UI and official model joint names differ')
    indices = list(range(7)) + [int(native.jnt_qposadr[mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in render_names]
    proof = dict(native_scene=str(scene),native_scene_sha256=digest(scene),official_entry_sha256=digest(entry),
        policy_yaml_sha256=digest(MODEL.with_suffix('.yaml')),render_model_sha256=digest(render_path),
        native_joint_names=names,render_joint_names=render_names,native_to_render_qpos_indices=indices,
        released_model_dynamics_match=True)
    return render,indices,proof


def snapshot(event, model, indices, session, not_before_ns):
    import mujoco
    import numpy as np
    now = time.perf_counter_ns()
    if (event.get('kind')!='render' or event.get('reset_id')!=0 or event.get('sequence')!=0
            or event.get('simulation_time')!=0 or not not_before_ns<=event.get('copied_ns',0)<=event.get('submitted_ns',0)<=now):
        raise ValueError('Require this run\'s initial official viewer event before human input')
    native = np.asarray(event.get('qpos'),float)
    if native.shape!=(36,) or not np.isfinite(native).all() or abs(np.linalg.norm(native[3:7])-1)>1e-5:
        raise ValueError('Invalid official initial qpos')
    qpos=native[indices];data=mujoco.MjData(model);data.qpos[:]=qpos
    mujoco.mj_forward(model,data)
    return dict(type='pose',session=session,seq=0,qpos=qpos.tolist(),body_pos=data.xpos.tolist(),
        root_pos=qpos[:3].tolist(),simulation_time=0.,physics=False,preparation=True,
        preparation_kind='official_model_initial_pose',preparation_steps=0,human_observed=False,
        human_observation_count=0,pc_receive_ns=now,native_copied_ns=event['copied_ns'],
        native_submitted_ns=event['submitted_ns'],native_reset_id=0,native_sequence=0,
        reference_scope='model_preparation_only',motion_mode='full_body',model_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--session',required=True);parser.add_argument('--not-before-ns',type=int,required=True)
    args=parser.parse_args()
    sys.path.insert(0,r'C:\Users\developer\Documents\Codex\2026-10-20\iq\work\full_pipeline_exec_20261022')
    from state_io import write_json
    model,indices,proof=model_mapping(args.root)
    trace=args.output/'native_events.ndjson';deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if (args.output/'native_trace_error.txt').exists():raise RuntimeError('Official viewer trace failed')
        if trace.exists():
            with trace.open(encoding='utf8') as stream:
                line=stream.readline()
            if line.endswith('\n'):
                event=json.loads(line);pose=snapshot(event,model,indices,args.session,args.not_before_ns)
                pose['session_output']=str(args.output.resolve())
                write_json(args.output/'full_body_preparation_pose.json',pose)
                write_json(args.output/'full_body_ready.json',dict(ready=True,session=args.session,
                    pose_path='full_body_preparation_pose.json',motion_mode='full_body',model_verified=True,
                    preparation_kind=pose['preparation_kind'],preparation_steps=0,human_observation_count=0,
                    physics=False,pc_ns=time.perf_counter_ns(),human_calibration_ready=False,model_identity=proof,
                    session_output=str(args.output.resolve()),backend_start_ns=args.not_before_ns))
                return
        time.sleep(.02)
    raise TimeoutError('No current official model initial viewer state')


if __name__=='__main__':main()
