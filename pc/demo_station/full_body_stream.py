"""Full-body check diagnostics around the frozen stream; original admission/map/control remain authoritative."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

PC=Path(r'C:\Users\developer\Documents\Codex\2026-10-20\iq\work\full_pipeline_exec_20261022')
STREAM=Path(r'C:\Users\developer\Documents\Codex\2026-10-23\new-chat\outputs\remaining_failures_20261024\release\sources\semantic\transaction_stream.py')
BRIDGE=Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020\g1_stereo_bridge')


def startup_failure_message(report):
    preparation=report.get('standing_preparation') or {}
    if report.get('sent_bvh_frames')==0 and preparation and not preparation.get('ready'):
        return '本组未建立可靠站立基准，G1 未启动：'+preparation.get('reason','人体观测不足')
    gate=report.get('round3_startup',{})
    if (report.get('sent_bvh_frames')==0 and report.get('warmup_valid_frames',0)>0
            and gate.get('skipped_frames',0)>0):
        return ('人体初始化已通过，但首帧时效检查未通过，G1 没有开始跟随；'
                f"{gate['skipped_frames']} 帧超过 {gate.get('maximum_aligned_age_ms',80):g} ms 启动预算。")
    return None


class InitializationProgress:
    def __init__(self, output, writer, clock=time.perf_counter):
        self.output=Path(output);self.writer=writer;self.clock=clock
        self.converter=None;self.quality=None;self.packet=None;self.admission=None
        self.first_stamp=self.first_wall=self.last_warmup_stamp=None
        self.warmup_span=0.;self.last_status=None;self.last_write=0.;self.has_tracked=False
        self.mapper_failed=False
        self.standing_preparation=None

    def assess(self, quality, packet, admission):
        self.quality,self.packet,self.admission=quality,packet,admission
        if self.first_stamp is None:self.first_stamp,self.first_wall=packet['timestamp_ns'],self.clock()
        self.publish()

    def mapped(self, converter, packet, frame):
        self.converter=converter
        if converter.prefix or converter.offsets is not None:
            if self.last_warmup_stamp is None or converter.offsets is None or not self.warmup_span:
                self.last_warmup_stamp=packet['timestamp_ns']
            if converter.offsets is not None and not self.warmup_span:
                self.warmup_span=(packet['timestamp_ns']-converter.prefix_start_ns)/1e9
        self.mapper_failed=bool(frame is not None and frame['fit_p95_m']>.08)
        self.publish()

    def initialization(self):
        packet=self.packet;converter=self.converter;stamp=packet['timestamp_ns'] if packet else None
        anchors={}
        for i,name in [(15,'left_ankle'),(16,'right_ankle')]:
            history=self.quality.observations.get(i,[]) if self.quality else []
            anchors[name]=dict(observed_reliable_count=len(history),required_reliable_count=3,
                last_observation_age_ms=(stamp-history[-1])/1e6 if history else None)
        blockers=[]
        for reason in (self.admission or {}).get('reasons',[]):
            i=reason.get('joint');name=packet['joint_names'][i] if i is not None else None
            blocker=dict(joint=name,index=i,reason=reason['reason'])
            if name in anchors:blocker.update(anchors[name])
            blockers.append(blocker)
        count=0;span=0.;calibrated=bool(converter and converter.offsets is not None)
        if converter:
            count=converter.warmup_frames_used if calibrated else len(converter.prefix)
            if calibrated:span=self.warmup_span
            elif count and self.last_warmup_stamp is not None:
                span=max(0.,(self.last_warmup_stamp-converter.prefix_start_ns)/1e9)
                # The frozen caller clears its prefix immediately after this rejected assessment.
                if self.admission and not self.admission['accepted'] and stamp-self.last_warmup_stamp>200_000_000:
                    count=0;span=0.
        preparation=self.standing_preparation
        standing_ready=(preparation or {}).get('ready',False) if os.environ.get('G1_STANDING_SMALL_MOTION_CHECK')=='1' else True
        if self.has_tracked:stage='tracking'
        elif packet is None:stage='waiting_input'
        elif preparation and not preparation['ready'] and self.admission and self.admission['accepted']:
            stage='waiting_standing_reference'
        elif any(b['reason'] in ('body_bone_proportion','body_bone_length_invalid','observed_body_bone_length',
                                'estimated_body_bone_length','estimated_leg_length') for b in blockers):
            stage='waiting_body_quality'
        elif any(a['observed_reliable_count']<3 for a in anchors.values()):stage='waiting_ankle_anchors'
        elif self.mapper_failed:
            stage='waiting_body_quality';blockers.append(dict(joint=None,index=None,reason='fixed_bone_mapping_above_80mm'))
        elif calibrated and (self.last_status or {}).get('startup_freshness'):stage='waiting_fresh_reference'
        elif calibrated:stage='initializing_gmr' if standing_ready else 'waiting_standing_reference'
        elif blockers:stage='waiting_body_quality'
        else:stage='collecting_body_bones'
        return dict(stage=stage,elapsed_s=0. if self.first_wall is None else self.clock()-self.first_wall,
            source_elapsed_s=0. if stamp is None else (stamp-self.first_stamp)/1e9,
            blockers=blockers,ankle_anchors=anchors,
            warmup=dict(accepted_frames=count,required_frames=10,span_s=span,required_span_s=1.4,max_gap_ms=200),
            human_calibration_ready=calibrated and standing_ready,complete=self.has_tracked,
            standing_preparation=preparation,
            source_seq=packet['seq'] if packet else None,source_timestamp_ns=stamp,
            startup_freshness_enabled=True,startup_freshness_budget_ms=float(os.environ.get('G1_REFERENCE_LAG_MS','240'))-160,
            startup_freshness=(self.last_status or {}).get('startup_freshness'),
            diagnostic_only=True)

    def write_status(self,path,value):
        if Path(path).resolve()!=(self.output/'tracking_status.json').resolve():
            return self.writer(path,value)
        self.last_status=dict(value)
        if 'standing_preparation' in value:self.standing_preparation=value['standing_preparation']
        if value.get('state')=='tracking':self.has_tracked=True
        self.publish(force=True)

    def publish(self,force=False):
        if self.last_status is None:return
        now=self.clock()
        if not force and now-self.last_write<.1:return
        self.last_write=now
        value=dict(self.last_status,initialization=self.initialization())
        self.writer(self.output/'tracking_status.json',value)
        with (self.output/'initialization_progress.ndjson').open('a',encoding='utf8') as log:
            log.write(json.dumps(dict(progress_pc_ns=time.perf_counter_ns(),**value),allow_nan=False)+'\n')


def install(output):
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(STREAM.parent),str(PC),str(BRIDGE)]
    import conversion
    import tracking_quality
    import state_io
    from demo_station.core import write_json, read_json
    from demo_station.full_body_quality import SkeletonQuality
    progress=InitializationProgress(output,write_json)
    original_quality=SkeletonQuality;original_converter=conversion.LiveBVH
    class ReportedQuality(original_quality):
        def assess(self,packet):
            result=super().assess(packet);progress.assess(self,packet,result);return result
    class ReportedConverter(original_converter):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);progress.converter=self
        def push(self,packet):
            result=super().push(packet)
            if self.offsets is not None and progress.quality is not None:
                # Only the original accepted warmup can freeze the personal
                # profile; later observed outliers cannot move its lengths.
                indices=(16,17,20,21,2,3,6,7)
                progress.quality.freeze_bone_lengths({bone:math.sqrt(sum(float(v)**2 for v in self.offsets[index]))
                    for bone,index in zip(progress.quality.BONES,indices)})
            progress.mapped(self,packet,result);return result
    tracking_quality.SkeletonQuality=ReportedQuality;conversion.LiveBVH=ReportedConverter
    state_io.write_json=progress.write_status
    state_io.read_json=read_json
    return progress


def main():
    if '--help' not in sys.argv and '-h' not in sys.argv:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,required=True)
        args,_=parser.parse_known_args()
        install(args.output.parent)
        # Reuse the frozen, stricter startup gate: no old timestamp is rewritten or old pose made current.
        os.environ['ROUND3_STARTUP_FRESH']='1'
    stream = Path(__file__).with_name('standing_small_motion_stream.py') if os.environ.get('G1_STANDING_SMALL_MOTION_CHECK') == '1' else STREAM
    runpy.run_path(str(stream),run_name='__main__')


if __name__=='__main__':main()
