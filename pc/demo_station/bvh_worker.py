"""Build a self-contained BVH session with the existing GMR / G1 runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo_station.bvh_playback import SCHEMA,build_skeleton_tracks


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def generate(source,destination,g1_root,start,end,name,physics_enabled=True):
    sys.path.insert(0,str(g1_root/'g1_realtime'))
    sys.path.insert(0,str(g1_root/'g1_physical'))
    import mujoco
    from bvh import read_clip
    from tracker import Tracker
    from physical import PhysicsTracker,ReferenceBuffer
    header,dt,frames = read_clip(source.read_text(encoding='utf-8-sig'))
    first = round(start/dt); stop = round(end/dt)
    if not (0<=first<stop<=len(frames)) or not 1<=end-start<=60:
        raise ValueError('片段必须在文件范围内，时长为 1–60 秒')
    selected = frames[first:stop]
    times = np.arange(len(selected))*dt
    destination.mkdir(parents=True,exist_ok=False)
    manifest = dict(schema=SCHEMA,state='building',name=name,frame_time=dt,
                    source=dict(file=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                start_frame=first,stop_frame_exclusive=stop,start_seconds=first*dt,
                                end_seconds_exclusive=stop*dt,unit='cm',up='Y'),
                    provenance='file_playback',media={},default_robot_mode='reference')
    save_json(destination/'session.json',manifest)
    shutil.copy2(source,destination/'source.bvh')
    tracker = Tracker(dict(header=header))
    # LAFAN1's official reader replaces the root OFFSET with position channels.
    # The portable generic parser adds them; adapt this file path only.
    original_root_offset = tracker.skeleton.offsets[0].copy()
    tracker.skeleton.offsets[0] = 0
    manifest['root_position_convention'] = 'LAFAN1 absolute root channels; root OFFSET not added'
    manifest['original_root_offset_cm'] = original_root_offset.tolist()
    human,reference,reference_body = [],[],[]
    started = time.perf_counter()
    for index,frame in enumerate(selected):
        result = tracker.solve(frame)
        human.append(result['human_pos']); reference.append(result['qpos']); reference_body.append(result['body_pos'])
        if index%30==0:print(f'IK {index+1}/{len(selected)}',flush=True)
    human = np.asarray(human); reference = np.asarray(reference)
    tracks,completion = build_skeleton_tracks(human,tracker.skeleton.canonical,times)
    tracks.update(times=times,human=human,reference_times=times,reference_qpos=reference,
                  reference_body=np.asarray(reference_body))
    manifest.update(human_names=tracker.skeleton.names,human_parents=tracker.skeleton.parents,
                    robot_joint_names=tracker.metadata['joint_names'],completion=completion,
                    completion_method='offline local cubic interpolation of surviving triangulated observations, max gap 0.4 s',
                    projection_kind='virtual pinhole stereo; no real camera inference',
                    head_mapping='Head centre at index 0; face landmarks 1-4 unavailable',
                    robot_input='full BVH, independent of completion demonstration')
    report = dict(attempted=physics_enabled,completed=False,fallen=None)
    if physics_enabled:
        physics = PhysicsTracker()
        order = [tracker.metadata['joint_names'].index(joint) for joint in physics.joints]
        reverse = np.argsort(order)
        targets = reference.copy(); targets[:,7:] = targets[:,7:][:,order]
        buffer = ReferenceBuffer(physics.model)
        # Append a constant final reference for policy lookahead; no truncated clip.
        padding = float(max(physics.offsets))+2*dt
        all_times = np.arange(len(times)+int(np.ceil(padding/dt))+1)*dt
        consumed = 0
        physics.reset(targets[0])
        render_data = mujoco.MjData(tracker.model)
        qposes,bodies,stamps,errors,heights,upright,costs = [],[],[],[],[],[],[]
        arm_ids = [i for i,n in enumerate(physics.joints) if any(s in n for s in ('shoulder','elbow','wrist'))]
        arm_errors = []
        def record(stamp):
            q = physics.data.qpos.copy(); q[7:] = q[7:][reverse]
            render_data.qpos[:] = q; mujoco.mj_forward(tracker.model,render_data)
            qposes.append(q); bodies.append(render_data.xpos.copy()); stamps.append(float(stamp))
            heights.append(float(q[2])); upright.append(float(physics.data.xmat[1,8]))
        record(0.)
        count = int(np.ceil(times[-1]/physics.dt-1e-8))
        for step in range(count):
            stamp = step*physics.dt
            need = stamp+max(float(max(physics.offsets)),physics.dt)
            while consumed<len(all_times) and (not buffer.times or buffer.times[-1]<need-1e-8):
                buffer.push(all_times[consumed],targets[min(consumed,len(targets)-1)])
                consumed += 1
            costs.append(physics.step(buffer.references(stamp+physics.offsets)))
            target = buffer.sample(stamp+physics.dt)[0]
            delta = physics.data.qpos[7:]-target[7:]
            errors.append(float(np.sqrt(np.mean(delta**2))))
            arm_errors.append(float(np.sqrt(np.mean(delta[arm_ids]**2))))
            record((step+1)*physics.dt)
            if step%50==0:print(f'Physics {step+1}/{count}',flush=True)
            if physics.fallen():break
        complete = len(costs)==count and not physics.fallen()
        reference_xy = np.column_stack([np.interp(stamps,times,reference[:,axis]) for axis in range(2)])
        root_errors = np.linalg.norm(np.asarray(qposes)[:,:2]-reference_xy,axis=1)
        tracking_accepted = complete and float(max(root_errors))<=.5
        report.update(completed=complete,fallen=bool(physics.fallen()),
                      simulated_seconds=float(len(costs)*physics.dt),source_seconds=float(times[-1]),
                      minimum_root_height_m=min(heights),minimum_upright=min(upright),
                      joint_rmse_rad=float(np.mean(errors)),arm_rmse_rad=float(np.mean(arm_errors)),
                      step_p95_ms=float(np.percentile(costs,95)),initial_floor_offset_m=physics.initial_floor_offset,
                      maximum_root_xy_error_m=float(max(root_errors)),tracking_accepted=tracking_accepted,
                      tracking_gate='complete, no fall, maximum root XY error <= 0.5 m; visual review still required',
                      control_dt=physics.dt,final_reference_padding_seconds=padding,
                      startup='reset at first retargeted pose, no hidden stabilization or root locking')
        # Failed trajectories are retained as evidence, never advertised as a complete mode.
        np.savez_compressed(destination/'physics_attempt.npz',times=stamps,qpos=qposes,body=bodies)
        if complete:
            tracks.update(physics_times=np.asarray(stamps),physics_qpos=np.asarray(qposes),physics_body=np.asarray(bodies))
            if tracking_accepted:manifest['default_robot_mode'] = 'physics'
    manifest.update(physics=report,state='complete',generation_seconds=time.perf_counter()-started)
    np.savez_compressed(destination/'tracks.npz',**tracks)
    save_json(destination/'session.json',manifest)
    print(json.dumps(dict(output=str(destination),physics=report,completion=completion),ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--g1-root',type=Path,required=True)
    parser.add_argument('--start',type=float,required=True)
    parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--reference-only',action='store_true')
    args = parser.parse_args()
    generate(args.source,args.output,args.g1_root,args.start,args.end,args.name,not args.reference_only)
