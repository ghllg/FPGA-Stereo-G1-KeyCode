"""Observed arm references and explicit standing holds through real policy/DDS."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import mujoco
import numpy as np

PREDICTION = Path(r'C:\Users\developer\Documents\Codex\2026-10-23\new-chat\work\offline_closure_20261024\timing\combined4e4')
PREPARATION_STEPS = 25


def install():
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(PREDICTION))
    import prediction_server  # Sets the frozen policy/controller dependency paths.
    import server
    import control_server
    from upper_body_tracker import UpperBodyTracker, SCOPE
    original_create = server.create_app

    def upper_app(*args, **kwargs):
        base = server.Stream

        class UpperStream(base):
            def __init__(self, message):
                super().__init__(message)
                self.lag = int(os.environ['G1_REFERENCE_LAG_MS'])/1000
                if not .16 <= self.lag <= .5: raise ValueError('Invalid reference lag')
                self.metadata.update(lag_ms=self.lag*1000, preparation_required=True,
                    reference_mode='observed_arms_with_explicit_robot_hold', maximum_source_silence_ms=440)
                self.upper_state = None
                self.last_receipt_ns = self.source_pc_origin_ns = self.last_input_source = None
                self.transport_paused = False
                self.preparation_steps = 0
                self.preparation_ready = False
                self.control_q = self.ik.qpos.copy()
                self.control_q[7:] = self.control_q[7:][self.order]
                folder = Path(os.environ['G1_PERFORMANCE_OUTPUT'])
                self.upper_log = (folder/'upper_body_references.ndjson').open('x', encoding='utf8', buffering=1)
                self.states = (folder/'states.ndjson').open('x', encoding='utf8', buffering=1)
                self.preparation_log = (folder/'preparation_states.ndjson').open('x', encoding='utf8', buffering=1)

            def validate_frame(self, message):
                stamp = float(message['source_time'])
                if (not np.isfinite(stamp) or type(message['seq']) is not int or message['seq'] <= self.last_seq
                        or (self.last_input_source is not None and stamp <= self.last_input_source)):
                    raise ValueError('Input source timestamps and sequence numbers must increase')

            def receive(self, message):
                with self.frame_lock:
                    try:
                        self.validate_frame(message)
                        receipt, origin = message['source_pc_receive_ns'], message['source_pc_origin_ns']
                        now = time.perf_counter_ns()
                        if any(type(value) is not int or value <= 0 or value > now for value in (receipt, origin)):
                            raise ValueError('Valid same-PC input clocks are required')
                        if self.source_pc_origin_ns is not None and origin != self.source_pc_origin_ns:
                            raise ValueError('Source clock alignment changed')
                        result = self.ik.solve(message['values'])
                    except (ValueError, KeyError, TypeError):
                        self.finished = True
                        self.physics.controller.stop_reference()
                        raise
                    stamp = float(message['source_time'])
                    if self.source_origin is None:
                        self.source_origin = stamp
                        self.source_pc_origin_ns = origin
                        self.wall_origin = origin/1e9+stamp
                        self.client_origin = self.wall_origin*1000
                    self.last_receipt_ns = receipt
                    self.last_input_source = stamp
                    self.transport_paused = False
                    self.upper_state = result['upper_body']
                    updated = any(arm['reference_updated'] for arm in self.upper_state['arms'].values())
                    if updated:
                        qpos = np.array(result['qpos']); qpos[7:] = qpos[7:][self.order]
                        self.buffer.push(stamp-self.source_origin, qpos)
                        self.control_q = qpos
                    self.received += 1
                    self.last_seq = message['seq']
                    accepted = time.perf_counter_ns()
                    response = dict(type='ack', seq=message['seq'], ik_ms=result['solve_ms'],
                        server_accepted_ns=accepted, reference_accepted=updated,
                        upper_body=self.upper_state, reference_scope=SCOPE,
                        control_state='tracking' if updated else 'holding_standing')
                    self.upper_log.write(json.dumps(dict(seq=message['seq'], source_time=stamp,
                        server_accepted_ns=accepted, reference_accepted=updated,
                        qpos=result['qpos'], **self.upper_state), allow_nan=False)+'\n')
                    return response

            def constant_references(self, qpos):
                self.buffer.fk.qpos[:] = qpos
                mujoco.mj_forward(self.physics.model, self.buffer.fk)
                count = len(self.physics.offsets)
                return dict(dof_pos=np.tile(qpos[7:], (count, 1)).astype(np.float32),
                    dof_vel=np.zeros((count, len(self.physics.joints)), dtype=np.float32),
                    body_rot=np.tile(self.buffer.fk.xquat[1:, [1,2,3,0]], (count,1,1)).astype(np.float32))

            def tick_locked(self):
                if self.finished: return None
                started = time.perf_counter()
                preparation = self.source_origin is None
                if not self.started:
                    self.physics.reset(self.control_q)
                    self.started = True
                if (not preparation and not self.ended and
                        time.perf_counter_ns()-self.last_receipt_ns > 440_000_000):
                    if not self.transport_paused:
                        self.transport_paused = True
                        self.physics.controller.stop_reference()
                        return dict(type='tracking_state', state='paused', finished=False,
                            reason='source transport silent for more than 440 ms', transport_paused=True)
                    return None
                source_now = 0. if preparation else max(0., time.perf_counter()-self.wall_origin-self.lag)
                target = source_now
                reference = self.control_q
                held = True
                if self.buffer.times:
                    target = float(np.clip(source_now, self.buffer.times[0], self.buffer.times[-1]))
                    reference = self.buffer.sample(target)[0]
                    held = source_now >= self.buffer.times[-1]
                if held:
                    policy_refs = self.constant_references(reference)
                else:
                    stamps = np.clip(target+self.physics.offsets, self.buffer.times[0], self.buffer.times[-1])
                    policy_refs = self.buffer.references(stamps)
                    policy_refs['dof_vel'][stamps >= self.buffer.times[-1]] = 0.
                self.physics.step(policy_refs)
                fallen = self.physics.fallen()
                if preparation:
                    self.preparation_steps += 1
                    self.preparation_ready = self.preparation_steps >= PREPARATION_STEPS and not fallen
                else:
                    self.processed += 1
                    if self.ended and source_now >= self.last_input_source-self.source_origin:
                        self.finished = True
                if fallen:
                    self.finished = True
                    self.physics.controller.stop_reference()
                qpos = self.physics.data.qpos.copy()
                render = qpos.copy(); render[7:] = render[7:][self.reverse]
                self.render_data.qpos[:] = render
                mujoco.mj_forward(self.ik.model, self.render_data)
                source_time = None if preparation else source_now+self.source_origin
                result = dict(type='preparation_pose' if preparation else 'pose', session=self.session,
                    seq=self.preparation_steps if preparation else self.processed, source_time=source_time,
                    source_time_kind='control_clock_not_new_human_observation',
                    reference_source_time=None if not self.buffer.times else target+self.source_origin,
                    latest_input_source_time=self.last_input_source, new_human_observation=False,
                    client_time=None if preparation else self.client_origin+source_now*1000,
                    qpos=render.tolist(), root_pos=qpos[:3].tolist(), root_quat=qpos[3:7].tolist(),
                    joint_pos=render[7:].tolist(), body_pos=self.render_data.xpos.tolist(),
                    body_quat=self.render_data.xquat.tolist(), human_pos=[],
                    geom_pos=self.render_data.geom_xpos.tolist(), geom_mat=self.render_data.geom_xmat.tolist(),
                    solve_ms=(time.perf_counter()-started)*1000,
                    process_ms=None if preparation else (time.perf_counter()-self.wall_origin-source_now)*1000,
                    received=self.received, processed=self.processed, dropped=0, waiting=False,
                    fallen=fallen, finished=self.finished, physics=True,
                    simulation_time=self.physics.data.time, initial_floor_offset=self.physics.initial_floor_offset,
                    preparation=preparation, human_observed=False,
                    preparation_steps=self.preparation_steps, preparation_ready=self.preparation_ready,
                    reference_scope=SCOPE, fixed_lower_body_reference=True, lower_body_measured=False,
                    robot_reference_held=held, control_state='preparing' if preparation else 'holding_standing' if held else 'tracking',
                    upper_body=self.upper_state, terminal_hold_drained=bool(self.ended and self.finished and not fallen))
                record = dict(result, native_qpos=qpos.tolist(), server_completed_ns=time.perf_counter_ns(),
                              policy_reference_qpos=reference.tolist(), policy_reference_dof_vel=policy_refs['dof_vel'].tolist())
                (self.preparation_log if preparation else self.states).write(json.dumps(record, allow_nan=False)+'\n')
                return result

            def close(self):
                for name in ('upper_log', 'states', 'preparation_log'):
                    stream = getattr(self, name, None)
                    if stream and not stream.closed: stream.close()
                super().close()

        server.Stream = UpperStream
        return original_create(*args, **kwargs)

    server.create_app = upper_app
    control_server.RemoteTracker = UpperBodyTracker
    return control_server


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(install().main())
