"""Original native G1 service with per-test reference lag and timing evidence."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

G1 = Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020')
sys.path.insert(0, str(G1/'g1_unitree_official/standalone'))
sys.path.insert(0, str(G1/'g1_physical'))
import server
import control_server
sys.path.insert(0, 'C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\work\\full_pipeline_exec_20261022')
sys.path.insert(0, 'C:\\Users\\developer\\Documents\\Codex\\2026-10-23\\new-chat\\work\\offline_closure_20261024\\timing\\combined4e4')
sys.path.insert(0,str(Path(__file__).parent))
from position_control_rollhalf import PositionFeedback
server.PositionFeedback = PositionFeedback
from prediction_guard import tick_locked
from future_prediction import clear_forecast, pause_for_forecast
from native_policy_forecast import ForecastUnavailable
from atomic_reference_evidence import freeze_reference_queries
from tracking_quality import recovery_step
from actual_tracking_guard import ActualTrackingGuard


def install():
    create = server.create_app

    def recorded_app(*args, **kwargs):
        base = server.Stream

        class RecordedStream(base):
            tick_locked = tick_locked
            def __init__(self, message):
                super().__init__(message)
                self.lag = int(os.environ['G1_REFERENCE_LAG_MS'])/1000
                if not .16 <= self.lag <= .5:
                    raise ValueError('Reference lag must cover at least the 160ms policy horizon')
                self.metadata['lag_ms'] = self.lag*1000
                self.reference_history = []
                self.terminal_pause_reported=False
                self.paused = self.recovering = self.resume_pending = self.recovery_priming = False
                self.recovery_previous = self.recovery_stamp = None
                self.actual_tracking_guard = ActualTrackingGuard()
                self.actual_recovery_completion = None
                self.recording_max_ms=dict(states=0.,references=0.)
                folder = Path(os.environ['G1_PERFORMANCE_OUTPUT'])
                self.states = (folder/'states.ndjson').open('x', encoding='utf-8', buffering=1)
                self.references = (folder/'references.ndjson').open('x', encoding='utf-8', buffering=1)
                self.forecast_rejections = (folder/'forecast_rejections.ndjson').open('x', encoding='utf-8', buffering=1)

            def set_tracking_state(self, message):
                with self.frame_lock:
                    state=message['state']
                    if self.actual_tracking_guard.failure is not None:
                        raise ValueError(self.actual_tracking_guard.failure_message())
                    if state=='paused':
                        clear_forecast(self)
                        self.paused=True
                        self.physics.controller.stop_reference()
                    elif state=='recovering':
                        if self.paused:self.resume_pending=True
                    else:raise ValueError('Unknown tracking transition')
                    return dict(type=('tracking_state_ack' if os.environ.get('ROUND3_ACK_FIX') == '1' else 'tracking_state'),state=state,reason=message.get('reason'))

            def receive(self, message):
                started = time.perf_counter_ns()
                result = super().receive(message)
                accepted=result['server_accepted_ns']
                self.references.write(json.dumps(dict(session=self.session, seq=message['seq'],
                    source_time=message['source_time'], received_ns=started,
                    accepted_ns=accepted, ik_ms=result['ik_ms'],
                    recovery_conditioned=result.get('recovery_conditioned',False),
                    recovery_evidence=result.get('recovery_evidence')), separators=(',', ':'))+'\n')
                saved=time.perf_counter_ns()
                self.recording_max_ms['references']=max(self.recording_max_ms['references'],(saved-accepted)/1e6)
                result.update(server_accepted_ns=accepted,server_reference_saved_ns=saved)
                return result

            def validate_frame(self, message):
                if self.actual_tracking_guard.failure is not None:
                    raise ValueError(self.actual_tracking_guard.failure_message())
                return super().validate_frame(message)

            def accept_frame(self, message, result):
                if self.actual_tracking_guard.failure is not None:
                    raise ValueError(self.actual_tracking_guard.failure_message())
                first = self.source_origin is None
                receipt = message.get('source_pc_receive_ns')
                if os.environ.get('G1_RECEIPT_CLOCK') == '1':
                    if type(receipt) is not int or receipt<=0 or receipt>time.perf_counter_ns():
                        raise ValueError('A valid same-PC source receipt clock is required')
                    origin = message.get('source_pc_origin_ns')
                    if type(origin) is not int or origin<=0 or origin>time.perf_counter_ns():
                        raise ValueError('Source clock alignment must be a past PC monotonic time')
                    if not first and origin != self.source_pc_origin_ns:
                        raise ValueError('Source clock alignment changed within a session')
                if self.resume_pending:
                    self.actual_recovery_completion = None
                    clear_forecast(self)
                    anchor=float(np.clip(self.t,self.buffer.times[0],self.buffer.times[-1]))
                    self.recovery_previous=self.buffer.sample(anchor)[0]
                    self.recovery_height=float(np.interp(anchor,self.buffer.times,self.heights)) if self.heights else None
                    if self.started:
                        # The robot can lag its old reference at the moment of
                        # a quality pause. Resume from the actual paused pose,
                        # expressed back in reference coordinates, not that
                        # potentially distant pre-pause target. No state reset.
                        self.recovery_previous=self.physics.data.qpos.copy()
                        self.recovery_previous[2]-=self.physics.initial_floor_offset
                        if self.position_feedback and self.position_feedback.origin is not None:
                            self.recovery_previous[:2]-=self.position_feedback.origin
                        if self.height_feedback and self.height_feedback.origin is not None:
                            self.recovery_height=(self.physics.data.qpos[2]-self.height_feedback.origin)/self.height_feedback.scale
                    self.buffer.times.clear();self.buffer.poses.clear();self.buffer.curve=None
                    self.humans.clear();self.heights.clear()
                    self.recovery_stamp=float(message['source_time'])
                    self.resume_pending=False;self.recovering=True;self.recovery_priming=True
                conditioned=self.recovering
                recovery_evidence=None
                if conditioned:
                    stamp=float(message['source_time'])
                    target=np.array(result['qpos']);target[7:]=target[7:][self.order]
                    dt=min(max(stamp-self.recovery_stamp,0.),.2)
                    bounded,finished=recovery_step(self.recovery_previous,target,dt)
                    recovery_evidence=dict(dt=dt,previous=self.recovery_previous.tolist(),target=target.tolist(),bounded=bounded.tolist())
                    self.recovery_previous=bounded;self.recovery_stamp=stamp
                    bounded=bounded.copy();bounded[7:]=bounded[7:][self.reverse]
                    result=dict(result,qpos=bounded.tolist())
                    if self.recovery_height is not None:
                        desired=float(message['source_hip_height_m'])
                        self.recovery_height+=float(np.clip(desired-self.recovery_height,-.3*dt,.3*dt))
                        message=dict(message,source_hip_height_m=self.recovery_height)
                        finished &= abs(self.recovery_height-desired)<.001
                    # A completed reference must reach the actual physics clock.
                    # Latch its first completion time so normal input lag cannot
                    # move the goal forever; withdraw it if the target diverges.
                    reference_finished = bool(finished)
                    if not reference_finished:
                        self.actual_recovery_completion = None
                    elif self.actual_recovery_completion is None:
                        self.actual_recovery_completion = stamp
                    finished = reference_finished and self.actual_tracking_guard.recovery_ready(self.actual_recovery_completion)
                    recovery_evidence.update(reference_finished=reference_finished,
                        reference_completion_source_time=self.actual_recovery_completion,
                        latest_physics_source_time=(self.actual_tracking_guard.latest or {}).get('source_time'))
                    self.recovering=not finished
                response = super().accept_frame(message, result)
                if self.recovery_priming and self.buffer.times[-1]-self.buffer.times[0]>=max(self.physics.offsets):
                    self.paused=self.recovery_priming=False
                response.update(recovery_conditioned=conditioned,recovering=self.recovering or self.recovery_priming,
                    recovery_evidence=recovery_evidence)
                if first and os.environ.get('G1_RECEIPT_CLOCK') == '1':
                    self.source_pc_origin_ns = origin
                    self.wall_origin = origin/1e9+message['source_time']
                    self.client_origin = self.wall_origin*1000
                accepted=time.perf_counter_ns()
                response['server_accepted_ns']=accepted
                self.reference_history.append(dict(seq=message['seq'], source_time=float(message['source_time']),accepted_ns=accepted))
                self.reference_history = self.reference_history[-64:]
                return response

            def freeze_reference_queries(self, queries):
                return freeze_reference_queries(self.reference_history,self.buffer.times,self.source_origin,queries,time.perf_counter_ns())

            def tick(self):
                with self.frame_lock:
                    self._tick_terminal=bool(self.ended)
                    if self.paused:
                        if self._tick_terminal and not self.finished and not self.terminal_pause_reported:
                            self.terminal_pause_reported=True
                            return dict(type='tracking_state',state='paused',
                                reason='source ended while paused; references not drained',terminal_undrained=True)
                        return None
                    if self.started and not self._tick_terminal and os.environ.get('G1_QUALITY_GUARD')=='1':
                        wanted=max(0.,time.perf_counter()-self.wall_origin-self.lag)
                        known=self.buffer.times[-1]
                        # P3 measures current observed deficit, not unavailable
                        # policy future. Aligned source age is bounded by lag
                        # 240ms + current deficit 200ms; PC endpoints are measured
                        # separately rather than promised by this timing check.
                        if wanted-known>.2:
                            self.paused=True
                            self.physics.controller.stop_reference()
                            return dict(type='tracking_state',state='paused',
                                reason='observed reference coverage is more than 200ms late')
                    try:
                        result = super().tick()
                    except ForecastUnavailable as error:
                        return pause_for_forecast(self,str(error))
                    if result:
                        actual_evidence = self.actual_tracking_guard.observe(
                            result['root_pos'], result['human_pos'][0], result['source_time'])
                        result['actual_tracking_guard'] = actual_evidence
                        if actual_evidence['failed']:
                            self.paused = self.finished = True
                            clear_forecast(self)
                            self.physics.controller.stop_reference()
                        keys = ('session', 'seq', 'source_time', 'simulation_time', 'qpos', 'root_pos',
                                'human_pos', 'horizontal_error_m', 'waiting', 'fallen', 'finished',
                                'unguarded_missing_future', 'source_clock_guard_active', 'source_clock_delay_ms',
                                'future_reference_latest_time', 'reference_available_through', 'reference_terminal_drain',
                            'predicted_future', 'future_prediction', 'actual_tracking_guard')
                        state = {key:result[key] for key in keys}
                        state['native_qpos'] = self.physics.data.qpos.tolist()
                        state['server_completed_ns'] = time.perf_counter_ns()
                        state['recovery_conditioned'] = self.recovering
                        state.update(self._reference_use_snapshot)
                        state['reference_bracket']=state['reference_queries'][0]['observation_bracket']
                        state['future_bracket']=state['reference_queries'][-1]['observation_bracket']
                if result:
                    before=time.perf_counter_ns()
                    self.states.write(json.dumps(state, separators=(',', ':'), allow_nan=False)+'\n')
                    self.recording_max_ms['states']=max(self.recording_max_ms['states'],(time.perf_counter_ns()-before)/1e6)
                    if self.actual_tracking_guard.failure is not None and not result.get('fallen'):
                        return dict(type='error',message=self.actual_tracking_guard.failure_message(),
                            failure_evidence=self.actual_tracking_guard.failure,reference_drained=False)
                return result

            def close(self):
                (Path(os.environ['G1_PERFORMANCE_OUTPUT'])/'actual_tracking_guard.json').write_text(
                    json.dumps(dict(latest=self.actual_tracking_guard.latest, failure=self.actual_tracking_guard.failure,
                        reference_baseline_reset_on_recovery=False),ensure_ascii=False,indent=2),encoding='utf8')
                for name in ('states', 'references', 'forecast_rejections'):
                    stream = getattr(self, name, None)
                    if stream and not stream.closed:
                        stream.flush()
                (Path(os.environ['G1_PERFORMANCE_OUTPUT'])/'server_recording.json').write_text(json.dumps(self.recording_max_ms),encoding='utf8')
                super().close()

        server.Stream = RecordedStream
        return create(*args, **kwargs)

    server.create_app = recorded_app


if __name__ == '__main__':
    install()
    asyncio.run(control_server.main())
