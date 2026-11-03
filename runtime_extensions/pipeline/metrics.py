"""Join actual native physics/render states to timestamped source references."""
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/full_pipeline_plan_20261022'))
from performance_report import rate, rows, statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def crossing_latencies(frames, events, end_key, packets, camera_rows=None):
    """First actual state whose nominal current reference reaches each source frame.

    This is a source-reference timeline latency, not a measured human step response.
    Future-reference influence and physical tracking error remain separate metrics.
    """
    times = [e['source_time'] for e in events]
    output = []
    for frame in frames:
        if not times or not times[0] <= frame['source_time'] <= times[-1]: continue
        event = events[bisect_left(times, frame['source_time'])]
        packet = packets[frame['source_timestamp_ns']]
        elapsed = (event[end_key]-packet['pc_receive_ns'])/1e6
        if elapsed < 0: raise ValueError('Source frame reached the endpoint before PC receipt')
        item = dict(source_timestamp_ns=frame['source_timestamp_ns'], sequence=event['sequence'],
                    source_time=frame['source_time'], pc_ms=elapsed, endpoint_ns=event[end_key])
        if camera_rows is not None:
            raw = camera_rows[frame['source_timestamp_ns']]
            clock = raw.get('latency')
            if clock:
                capture = min(c['capture_timestamp_us'] for c in raw['message']['cameras'])*1000
                item['camera_lower_ms'] = (event[end_key]-capture+clock['offset_min_ns'])/1e6
                item['camera_upper_ms'] = (event[end_key]-capture+clock['offset_max_ns'])/1e6
                item['clock_bound_width_ms'] = (clock['offset_max_ns']-clock['offset_min_ns'])/1e6
        output.append(item)
    return output


def analyze(folder):
    folder = Path(folder)
    scope = read(folder/'scope.json')
    live = scope['mode'] == 'live' and not scope.get('pc_receipts_reissued', False)
    input_path = folder/'pc_receiver/received.ndjson' if live else folder/'replayed.ndjson'
    raw = rows(input_path)
    packets_list = rows(folder/'online_3d/skeleton3d.jsonl')
    backend = read(folder/'backend.json')
    bridge, processing = backend['bridge'], backend['processing']
    if not bridge.get('sent_bvh_frames') or not bridge.get('output_frames'):
        rejected = sum(abs(r['message']['pair_delta_us'])>10000 for r in raw)
        report = dict(headline=dict(pc_half_passed=False,live_full_chain_passed=False,
            source_hz=0.,physics_hz=0.,render_hz=0.,pc_to_physics_ms=None,
            camera_to_render_upper_ms=None,failed_checks=['no_valid_G1_tracking']),
            scope=scope,source_hash=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            source_frames=len(raw),processed_frames=len(packets_list),sent_frames=0,
            stereo_gate_ms=10.,synchronization_rejected_pairs=rejected,
            checks=dict(valid_G1_tracking=False),
            reason='No valid BVH/physics output. The opened window is an idle viewer, not a successful tracking test.',
            bridge_errors=bridge['errors'],formal_performance_accepted=False)
        (folder/'performance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report
    packets = {p['timestamp_ns']:p for p in packets_list}
    camera_rows = {r['message']['cameras'][0]['capture_timestamp_us']*1000:r for r in raw}
    bvh = rows(folder/'g1/bvh_frames.ndjson')
    references = rows(folder/'references.ndjson')
    states = rows(folder/'states.ndjson')
    events = rows(folder/'native_events.ndjson')
    if len(packets) != len(packets_list): raise ValueError('Duplicate source 3D frame')
    sent = {f['seq']:f for f in bvh}
    for reference in references:
        frame = sent[reference['seq']]
        if frame['source_time'] != reference['source_time']:
            raise ValueError('Server reference differs from the sent source frame')
        if frame['source_pc_monotonic_ns'] != packets[frame['source_timestamp_ns']]['pc_receive_ns']:
            raise ValueError('Source receipt identity mismatch')
    epochs = {e['reset_id'] for e in events if e['kind']=='physics'}
    if len(epochs) != 1: raise ValueError('Missing physics trace or mixed native reset epochs')
    physics = {e['sequence']:e for e in events if e['kind']=='physics'}
    if len(physics) != sum(e['kind']=='physics' for e in events): raise ValueError('Repeated physics sequence')
    state_by_seq = {s['seq']:s for s in states}
    joined = []
    for seq, physical in sorted(physics.items()):
        state = state_by_seq.get(seq)
        if state is None: continue
        if abs(state['simulation_time']-physical['simulation_time']) > 1e-5:
            raise ValueError('Native/server simulation time mismatch')
        if not np.allclose(state['native_qpos'], physical['qpos'], atol=2e-6, rtol=0):
            raise ValueError('Native/server physical state mismatch')
        joined.append(dict(physical, source_time=state['source_time'], waiting=state['waiting']))
    displayed = []
    seen = set()
    for event in sorted(events, key=lambda e:e.get('submitted_ns', e.get('completed_ns'))):
        if event['kind'] != 'render' or event['sequence'] not in physics: continue
        seq = event['sequence']
        physical = physics[seq]
        if not np.allclose(event['qpos'], physical['qpos'], atol=1e-12, rtol=0):
            raise ValueError('Displayed native pose differs from actual physics state')
        if event['copied_ns'] < physical['completed_ns']:
            raise ValueError('Display copy predates its actual physics completion')
        if seq not in seen and seq in state_by_seq:
            displayed.append(dict(event, source_time=state_by_seq[seq]['source_time']))
            seen.add(seq)
    if not joined or not displayed: raise ValueError('No actual native physics/render join')
    # Explicit steady-state boundary: two seconds after the first physics state.
    # End at last source receipt, so intentional producer-stop drainage is separate.
    lo = joined[0]['completed_ns']+2_000_000_000
    hi = raw[-1]['pc_receive_ns']
    steady_physics = [x for x in joined if lo <= x['completed_ns'] <= hi]
    steady_display = [x for x in displayed if lo <= x['submitted_ns'] <= hi]
    pc_physics = crossing_latencies(bvh, steady_physics, 'completed_ns', packets, camera_rows if live else None)
    pc_render = crossing_latencies(bvh, steady_display, 'submitted_ns', packets, camera_rows if live else None)
    camera_bounds = [x for x in pc_render if 'camera_upper_ms' in x]
    first_receipt = raw[0]['pc_receive_ns']
    compute = statistics([p['compute_ms'] for p in packets_list])
    receive_3d = statistics([(p['pc_complete_ns']-p['pc_receive_ns'])/1e6 for p in packets_list])
    state_hz = rate([x['completed_ns'] for x in steady_physics])
    render_hz = rate([x['submitted_ns'] for x in steady_display])
    full_source_hz = rate([x['source_timestamp_ns'] for x in bvh])
    physics_delay = statistics([x['pc_ms'] for x in pc_physics])
    render_delay = statistics([x['pc_ms'] for x in pc_render])
    camera_delay = statistics([x['camera_upper_ms'] for x in camera_bounds]) if live else None
    source_frames = [f for f in bvh if lo <= f['send_pc_ns'] <= hi]
    windows = []
    if len(source_frames)>1:
        start, end = source_frames[0]['source_timestamp_ns'], source_frames[-1]['source_timestamp_ns']
        while start+5_000_000_000 <= end:
            count = sum(start <= f['source_timestamp_ns'] < start+5_000_000_000 for f in source_frames)
            windows.append(count/5)
            start += 5_000_000_000
    waiting_fraction = sum(x['waiting'] for x in steady_physics)/len(steady_physics) if steady_physics else None
    steady_states = [state_by_seq[x['sequence']] for x in steady_physics]
    guarded = [s for s in steady_states if 'source_clock_guard_active' in s]
    guard_details = None
    if guarded:
        guard_details = dict(
            active_fraction=sum(s['source_clock_guard_active'] for s in guarded)/len(guarded),
            # This signed difference compares the nominal lag cursor with the
            # first available recovery reference, not two elapsed clocks.
            # Preserve leads separately instead of treating them as negative
            # measured endpoint latency; the original signed values stay in states.
            added_reference_delay_ms=statistics([max(0.,s['source_clock_delay_ms']) for s in guarded]),
            recovery_reference_lead_ms=statistics([max(0.,-s['source_clock_delay_ms']) for s in guarded]),
            unguarded_future_shortage_fraction=sum(s['unguarded_missing_future'] for s in guarded)/len(guarded),
            future_reference_violations=sum(not s['reference_terminal_drain'] and
                s['future_reference_latest_time']>s['reference_available_through']+1e-8 for s in guarded),
            scope='Reference progression waits for observed future samples; wait time is included in measured endpoint latency.')
    roots = np.array([s['root_pos'] for s in states])
    humans = np.array([s['human_pos'][0] for s in states])
    horizontal = np.linalg.norm((roots[:,:2]-roots[0,:2])-(humans[:,:2]-humans[0,:2]), axis=1)
    max_gap = max(np.diff([x['completed_ns'] for x in steady_physics]))/1e6 if len(steady_physics)>1 else None
    startup_ms = (joined[0]['completed_ns']-first_receipt)/1e6
    checks = dict(no_native_trace_errors=not (folder/'native_trace_error.txt').exists(),
        native_states_joined=len(joined)==len(states), rendered_values_match=True,
        source_pose_hz=bool(full_source_hz and full_source_hz>=13),
        five_second_pose_windows=bool(windows and min(windows)>=12),
        physics_output_hz=bool(state_hz and state_hz>=48), render_hz=bool(render_hz and render_hz>=45),
        compute_3d_p95=compute['p95']<=10, receipt_to_3d_p95=receive_3d['p95']<=10,
        bvh_p95=bridge['convert_p95_ms']<=5, ik_p95=bridge['ik_p95_ms']<=15,
        pc_to_physics_p95=bool(physics_delay and physics_delay['p95']<=300),
        pc_to_physics_max=bool(physics_delay and physics_delay['maximum']<=500),
        output_gap=bool(max_gap is not None and max_gap<=100),
        queue_drops=processing['dropped_queue_packets']/len(raw)<=.01,
        all_source_packets_accounted=len(packets_list)+processing['dropped_queue_packets']==len(raw),
        reference_coverage=bool(waiting_fraction is not None and waiting_fraction<=.01),
        startup_under_5s=startup_ms<=5000,
        no_fall=not bridge['fallen'], horizontal_error=float(max(horizontal))<=.2,
        expected_producer_stop=backend['producer_end_expected'])
    if guard_details:
        checks['no_unknown_future_references'] = guard_details['future_reference_violations']==0
    reference_folder = folder if live else Path(scope['source'])
    reference_file = reference_folder/'audit/reconstruction/skeleton3d.jsonl'
    if reference_file.exists():
        expected = {p['timestamp_ns']:p['joints'] for p in rows(reference_file)}
        checks['online_3d_matches_saved_geometry'] = all(p['joints']==expected.get(p['timestamp_ns']) for p in packets_list)
    if live:
        capture = read(folder/'audit/summary.json')
        checks['network_missing_pairs'] = capture['network_missing_pairs']==0
        checks.update(camera_clock_coverage=len(camera_bounds)>=.95*len(pc_render),
            camera_mean=bool(camera_delay and camera_delay['mean']<=600),
            camera_p95=bool(camera_delay and camera_delay['p95']<=700),
            camera_max=bool(camera_delay and camera_delay['maximum']<=1000))
    headline = dict(lag_ms=scope['lag_ms'], pc_half_passed=all(checks.values()),
        live_full_chain_passed=live and all(checks.values()), source_hz=full_source_hz,
        physics_hz=state_hz, render_hz=render_hz,
        pc_to_physics_ms=physics_delay, pc_to_render_ms=render_delay,
        camera_to_render_upper_ms=camera_delay, reference_wait_fraction=waiting_fraction,
        maximum_horizontal_error_m=float(max(horizontal)), startup_ms=startup_ms,
        failed_checks=[k for k,v in checks.items() if not v])
    report = dict(headline=headline, checks=checks, scope=scope,
        source_hash=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        steady_boundary_ns=[lo,hi], source_5s_window_hz=windows,
        source_frames=len(raw), processed_frames=len(packets_list), sent_frames=len(bvh),
        native_physics_frames=len(physics), server_states=len(states), joined_render_states=len(displayed),
        receive_to_3d_ms=receive_3d, compute_3d_ms=compute,
        observed_future_guard=guard_details,
        G1_input_conditioning=bridge.get('G1_input_conditioning'),
        temporally_held_body_frames=sum(bool(f.get('held_joints')) for f in bvh),
        body_without_temporal_hold_source_hz=rate([f['source_timestamp_ns'] for f in bvh if not f.get('held_joints')]),
        maximum_held_joint_age_ms=max((h['age_ms'] for f in bvh for h in f.get('held_joints',[])),default=0.),
        fresh_pose_count_scope='Distinct input source timestamps; partial body frames with at least half of the 12 body joints current and up to 100ms labeled holds are reported separately. Repeated output states are not fresh source poses.',
        camera_clock_bound_width_ms=statistics([x['clock_bound_width_ms'] for x in camera_bounds]) if live else None,
        pc_receipt_to_first_physics_ms=startup_ms,
        all_frame_latency_scope='First native state/render whose nominal current reference reaches the source frame time; not human step-response or physical screen scanout.',
        source_timestamps_unchanged=scope.get('source_timestamps_preserved',True),
        fixture_timestamps_preserved_during_replay=scope.get('fixture_timestamps_preserved_during_replay'),
        formal_performance_accepted=False, physical_state_verified=True, wrist_accuracy_accepted=False,
        long_run_accepted=False, absolute_ground_accepted=False)
    (folder/'performance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (folder/'frame_latencies.json').write_text(json.dumps(dict(physics=pc_physics,render=pc_render), indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1])['headline'], ensure_ascii=False, indent=2))
