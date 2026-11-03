"""ABI9/1080p receiver-log adapter. No board or camera access."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from .board_adapter import PacketDecoder
from .online import SkeletonEngine

VENDOR = Path(__file__).parent/'vendor/abi9_14p24'
MODEL_SHA256 = 'edf5c65c812c4e1b441d66bd7dd0bea3fb2a12f27f1e847124edd32cfa1354fc'
PROFILE = '1080p_14p24_20261019'
PER_VIEW_PROFILE = '1080p_per_view_roi_20261022'


def receiver_module():
    if not __debug__:
        raise RuntimeError('The vendor validator requires Python without -O.')
    source = VENDOR/'receive_paired_pose.py'
    manifest = json.loads((VENDOR/'PROVENANCE.json').read_text())
    expected = manifest['files'][source.name]['sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned receiver checksum mismatch.')
    spec = importlib.util.spec_from_file_location('abi9_receiver', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PacketValidator:
    def __init__(self, binding=None):
        self.validate = receiver_module().validate_message
        self.identity = self.previous = None
        self.profile = PROFILE
        self.expected_rois = {side: [555,0,810,1080] for side in ('left','right')}
        self.bound_identity = None
        if binding is not None and ('roi_xywh' in binding or 'input_profile' in binding):
            if (binding.get('input_profile') in (None,PROFILE)
                    and binding.get('roi_xywh') == [555,0,810,1080]):
                return
            if (binding.get('input_profile') != PER_VIEW_PROFILE
                    or binding.get('schema') != 'board_session_binding/1.0'
                    or binding.get('model_sha256') != MODEL_SHA256
                    or not binding.get('boot_id') or binding.get('session_id') is None):
                raise ValueError('Per-view ROI requires an explicit model and session binding.')
            rois = binding.get('roi_xywh')
            if not isinstance(rois,dict) or set(rois) != {'left','right'}:
                raise ValueError('Binding requires left and right ROIs.')
            for roi in rois.values():
                if (not isinstance(roi,list) or len(roi) != 4
                        or any(type(v) not in (int,float) or not math.isfinite(v) for v in roi)
                        or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0
                        or roi[0]+roi[2] > 1920 or roi[1]+roi[3] > 1080):
                    raise ValueError('Bound ROI must lie inside the 1080p source image.')
            self.expected_rois = {side:list(roi) for side,roi in rois.items()}
            self.profile = PER_VIEW_PROFILE
            self.bound_identity = (binding['boot_id'],binding['session_id'])

    def check(self, row):
        packet = row.get('message', row)
        try:
            self.validate(packet, packet['session_id'], (1920,1080), 16667)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid ABI9/1080p half-frame packet.') from error
        if (packet['inflight_pairs'] != 2 or packet['stream_contexts'] != 4
                or packet['valid_confidence_threshold'] != .3
                or any(c.get('roi_xywh') != self.expected_rois[c['camera_id']] for c in packet['cameras'])
                or packet.get('model_sha256', MODEL_SHA256) != MODEL_SHA256):
            raise ValueError('Packet differs from the selected ABI9/1080p profile.')
        for camera in packet['cameras']:
            if camera.get('model_sha256', MODEL_SHA256) != MODEL_SHA256:
                raise ValueError('Camera model differs from the frozen profile.')
        identity = (packet['boot_id'], packet['session_id'])
        if self.bound_identity is not None and identity != self.bound_identity:
            raise ValueError('Packet differs from the ROI binding boot/session.')
        current = (packet['pair_id'], *(c['capture_timestamp_us'] for c in packet['cameras']),
                   *(c['frame_id'] for c in packet['cameras']))
        if any(type(value) is not int or value < 0 for value in current):
            raise ValueError('Pair, frame and timestamp fields must be nonnegative integers.')
        if self.identity is not None and identity != self.identity:
            raise ValueError('Boot/session changed; start a separate stream.')
        if self.previous is not None and any(a <= b for a,b in zip(current,self.previous)):
            raise ValueError('Duplicate or out-of-order pair/camera frame.')
        self.identity, self.previous = identity, current
        return packet


def check_live_calibration(calibration, max_delta_ms):
    if (calibration.get('image_size') != [1920,1080]
            or calibration.get('diagnostic_only', False)
            or calibration.get('accepted_for_reconstruction') is not True
            or calibration.get('independent_validation_complete') is not True):
        raise ValueError('Live mode requires independently validated current 1080p geometry.')
    if max_delta_ms > 10. and calibration.get('validated_pair_limit_us', 10000) < 16667:
        raise ValueError('Live half-frame reconstruction requires dynamic pairing validation.')


def records(path, tail=False, timeout=2.):
    with Path(path).open(encoding='utf-8-sig') as stream:
        last = time.perf_counter()
        while True:
            position = stream.tell()
            line = stream.readline(65537)
            if len(line) > 65536:
                raise ValueError('Receiver record exceeds 64 KiB.')
            if not line or not line.endswith('\n'):
                if not tail:
                    if line.strip():
                        yield json.loads(line)
                    return
                stream.seek(position)
                if time.perf_counter()-last > timeout:
                    raise TimeoutError('Receiver log stopped; no stale pose will be repeated.')
                time.sleep(.01)
                continue
            last = time.perf_counter()
            if line.strip():
                yield json.loads(line)


def run(source, output, calibration=None, checkpoint=None, binding=None,
        tail=False, max_delta_ms=10., timeout=2., stabilize_arms=False):
    if max_delta_ms not in (10.,16.667) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid reconstruction policy or timeout.')
    supplied = [calibration is not None, checkpoint is not None, binding is not None]
    if any(supplied) and not all(supplied):
        raise ValueError('Reconstruction requires calibration, checkpoint and session binding.')
    validator = PacketValidator()
    engine = decoder = None
    if all(supplied):
        cal = json.loads(Path(calibration).read_text(encoding='utf-8-sig'))
        bound = json.loads(Path(binding).read_text(encoding='utf-8-sig'))
        validator = PacketValidator(bound)
        if cal['image_size'] != [1920,1080] or bound['model_sha256'] != MODEL_SHA256:
            raise ValueError('1080p calibration and the frozen 2D model binding are required.')
        if tail:
            check_live_calibration(cal,max_delta_ms)
        engine = SkeletonEngine(calibration,checkpoint,stabilize_arms=stabilize_arms,
                                max_delta_ms=max_delta_ms)
        decoder = PacketDecoder(engine.stereo,bound)
    elif tail:
        raise ValueError('Live mode requires a validated reconstruction configuration.')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(profile=validator.profile, mode='live_receiver_log' if tail else 'offline',
                  reconstructed=engine is not None, transport_pair_limit_us=16667,
                  reconstruction_pair_limit_ms=max_delta_ms, records=0, quality_rejected_pairs=0,
                  extended_admission_pairs=0, output_frames=0, full_body_frames=0,
                  exposure_sync_verified=False, error=None, current_mount_validated=False)
    if engine:
        report['current_mount_validated'] = bool(
            cal.get('accepted_for_reconstruction') and cal.get('independent_validation_complete')
            and not cal.get('diagnostic_only',False))
    timings = []
    first = last = None
    try:
        with (output/'validated.ndjson').open('x',encoding='utf8') as raw, \
                (output/'skeleton3d.jsonl').open('x',encoding='utf8') as skeleton:
            for number,row in enumerate(records(source,tail,timeout),1):
                packet = validator.check(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
                stamp = packet['cameras'][0]['capture_timestamp_us']
                if first is None: first = stamp
                last = stamp
                report['records'] += 1
                report['quality_rejected_pairs'] += abs(packet['pair_delta_us']) > max_delta_ms*1000
                report['extended_admission_pairs'] += 10000 < abs(packet['pair_delta_us']) <= max_delta_ms*1000
                if engine is None: continue
                pose = decoder.decode(row,number)
                result = engine.process(pose,'zynq_v4l2_monotonic:'+packet['boot_id'],str(packet['session_id']))
                result.update(input_profile=validator.profile,source_pair_id=packet['pair_id'],
                              transport_pair_limit_us=16667,reconstruction_pair_limit_ms=max_delta_ms,
                              current_mount_validated=report['current_mount_validated'])
                if 'pc_receive_ns' in row:
                    result['source_pc_receive_ns'] = row['pc_receive_ns']
                result['pc_complete_ns'] = time.perf_counter_ns()
                timings.append(result['compute_ms'])
                report['output_frames'] += 1
                report['full_body_frames'] += all(j['valid'] for j in result['joints'][5:])
                skeleton.write(json.dumps(result,allow_nan=False)+'\n');skeleton.flush()
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        import numpy as np
        report['source_pairs_per_s'] = (report['records']-1)*1e6/(last-first) if first is not None and last>first else None
        report['compute_p95_ms'] = float(np.percentile(timings,95)) if timings else None
        report['timing_scope'] = 'Compute only; saved PC receive times are not live latency measurements.'
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):
        parser.add_argument('--'+name,required=True)
    for name in ('calibration','checkpoint','binding'):
        parser.add_argument('--'+name)
    parser.add_argument('--tail',action='store_true')
    parser.add_argument('--max-delta-ms',type=float,choices=(10.,16.667),default=10.)
    parser.add_argument('--timeout',type=float,default=2.)
    parser.add_argument('--stabilize-arms',action='store_true')
    print(json.dumps(run(**vars(parser.parse_args())),ensure_ascii=False,indent=2))
