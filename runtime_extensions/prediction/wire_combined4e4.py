"""P4 plus roll-half and smooth80: a two-factor combination, not an ablation."""
import argparse,hashlib,json,os,runpy,shutil,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK))
from common import PC,ROUND3,G1
sys.path[:0]=[str(PC),str(ROUND3)]
import backend
from offline_eof_receiver import receiver_for_fixture
import live_receiver
eof_parser=argparse.ArgumentParser(add_help=False)
eof_parser.add_argument('--source',required=True,type=Path)
eof_args,_=eof_parser.parse_known_args()
live_receiver.LiveReceiver=receiver_for_fixture(eof_args.source)

os.environ['ROUND3_ACK_FIX']='1'
os.environ['ROUND3_STARTUP_FRESH']='0'
stream_path=WORK/'lifecycle_stream_v2.py'
server_path=ROOT/'prediction_server.py'
spawn=backend.launcher.spawn
def selected_spawn(path,*args,**kwargs):
    if backend.Path(path)==PC/'performance_server.py':path=server_path
    return spawn(path,*args,**kwargs)
backend.launcher.spawn=selected_spawn
popen=backend.subprocess.Popen
def selected_popen(command,*args,**kwargs):
    if isinstance(command,list):
        command=[str(stream_path) if str(item)==str(PC/'performance_stream.py') else item for item in command]
    return popen(command,*args,**kwargs)
backend.subprocess.Popen=selected_popen
BaseBackend=backend.Backend
class RecordedBackend(BaseBackend):
    def __init__(self,output,binding,lag_ms=380,smooth_ms=80,quality_guard=True):
        if (lag_ms,smooth_ms,quality_guard)!=(240,80,True):
            raise ValueError('Frozen combined4 requires --lag-ms 240 --smooth-ms 80 and quality guard')
        super().__init__(output,binding,lag_ms,smooth_ms,quality_guard)
        folder=self.output/'candidate_runtime_sources';folder.mkdir()
        paths=[*ROOT.glob('*.py'),WORK/'common.py',stream_path,
            PC/'backend.py',PC/'wire_replay.py',PC/'live_receiver.py',PC/'performance_receiver.py',
            G1/'g1_physical/server.py',G1/'g1_physical/position_control.py',
            G1/'g1_physical/height_control.py',G1/'g1_physical/physical.py',
            G1/'g1_physical/reference_trajectory.py',
            G1/'g1_physical/reference_conditioning.py',
            G1/'g1_unitree_official/standalone/control_server.py']
        manifest=[]
        for path in paths:
            shutil.copy2(path,folder/path.name)
            manifest.append(dict(path=str(path),snapshot=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (self.output/'candidate_runtime_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        config=dict(variant='P4_rollhalf_smooth80_offline_eof4',raw_calibration_prefix='stage2_v2',lag_ms=240,
            future_ms=160,smooth_ms=80,accepted_ack_fix=True,lifecycle='v2',
            max_prediction_ms=160,decay_ms=80,arrival_revision='diagnostic_not_admission',forecast_velocity='saturated_at_unchanged_bounds',
            maximum_observation_gap_ms=200,current_reference_latest_clamp=True,
            current_coverage_deficit_limit_ms=200,aligned_source_age_limit_ms=440,forecast_joint_limit_envelope=True,
            terminal_state_frozen_at_tick_entry=True,
            joint_limit_tolerance_rad=1e-10,forecast_rejection_log='forecast_rejections.ndjson',
            predictions_are_observations=False,predictions_count_for_recovery=False,
            current_reference='observed_only',hardware_access=False,
            lateral_rotation_scale=.5,feedback_state_update='unchanged',
            forward_rotation='unchanged',height_feedback='unchanged',
            comparator='P4 common baseline plus completed P3e, rollhalf and S80 controls',
            combination_factors=['final_lateral_rotvec_scale_0.5','smooth_ms_80'],
            single_variable_experiment=False,terminal_snapshot='P3e_one_tick_inherited_by_P4',
            tilt_log_scope='legacy XY filter state plus explicit applied world rotvec and final policy anchor quaternion')
        config['offline_eof']='Complete ordered identity sequence and empty parser state required at saved-wire orderly EOF; original worker drain retained'
        config['offline_eof_version']='e4_e3_checks_plus_bounded_probe_reader_join_before_producer_close'
        config['offline_probe_reader_join_timeout_s']=2.0
        (self.output/'candidate_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
backend.Backend=RecordedBackend
runpy.run_path(str(ROOT/'saved_wire_replay.py'),run_name='__main__')
