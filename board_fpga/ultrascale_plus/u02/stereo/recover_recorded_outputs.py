"""Resume output transport only after a shell line-limit failure; no PL rerun."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from run_recorded_board import U02,REPO,sha,Console,connect_file_server,SCHEMA,validate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    args=parser.parse_args();out=args.record.resolve()
    assert not (out/'recovered_result.json').exists()
    original=json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert original['driver_unloaded'] and not original['complete'] and len(original['runs'])==36
    remote=original['remote'];assert re.fullmatch(r'/tmp/u02_[0-9a-f]{32}',remote)
    manifest=json.loads((Path(original['inputs'])/'manifest.json').read_text(encoding='utf-8'))
    names=['clock_probe.json']
    for run in original['runs']:
        names.extend([run['prefix']+'.json',run['prefix']+'.json.heatmap',run['prefix']+'.json.input.f32',run['prefix']+'.clock.json'])
    result=dict(original)
    result.update(original_failure_record_sha256=sha(out/'result.json'),transport_recovery_only=True,complete=False)
    with (out/'transport_recovery_uart.log').open('wb') as log:
        console=Console('COM9',log)
        try:
            # Discard the unfinished quoted command at the continuation prompt.
            # The accepted driver was already normally unloaded before failure.
            console.send(b'\x03\n');console.sync()
            code,text=console.run('cat /proc/sys/kernel/random/boot_id; cat /proc/modules')
            assert code==0 and original['boot_id'] in text and not re.search(r'(?m)^u02_\w+\s+\d+\s',text)
            code,text=console.run(f'cd {remote} && tar -czf stereo_outputs.tar.gz clock_probe.json cpu_*.json* stream_csim_*.json* pl_stream_*.json* && sha256sum stereo_outputs.tar.gz',timeout=60)
            assert code==0
            digest=re.search(r'(?m)^([a-f0-9]{64})\s+stereo_outputs.tar.gz\r?$',text).group(1)
            code,_=console.run(f'timeout 45 nc -l -p 18087 < {remote}/stereo_outputs.tar.gz > /dev/null & sleep 1')
            assert code==0
            raw=bytearray()
            with connect_file_server(('192.168.50.2',18087)) as connection:
                while True:
                    part=connection.recv(65536)
                    if not part:break
                    raw.extend(part);assert len(raw)<24*1024*1024
            assert hashlib.sha256(raw).hexdigest()==digest
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
                assert sorted(archive.getnames())==sorted(names)
                for member in archive.getmembers():
                    assert member.isfile() and member.size<1024*1024
                    assert not (out/member.name).exists()
                    (out/member.name).write_bytes(archive.extractfile(member).read())
            messages=[];forward=[]
            for image in manifest['images']:
                suffix=image['file'][:-4]
                reference=json.loads((out/('stream_csim_'+suffix+'.json')).read_text())
                pl=json.loads((out/('pl_stream_'+suffix+'.json')).read_text())
                assert (out/('stream_csim_'+suffix+'.json.heatmap')).read_bytes()==(out/('pl_stream_'+suffix+'.json.heatmap')).read_bytes()
                assert reference['keypoints']==pl['keypoints']
                assert len({sha(out/(mode+'_'+suffix+'.json.input.f32')) for mode in ('cpu','stream_csim','pl_stream')})==1
                assert pl['pl_acceleration'] and len(pl['forward_profiles'])==1 and pl['forward_profiles'][0]['calls']==110
                clock=json.loads((out/('pl_stream_'+suffix+'.clock.json')).read_text())
                assert clock['child_exit_code']==0 and clock['end_ns']>=clock['start_ns']>0
                packet=dict(schema=SCHEMA,whole_network_on_board=True,camera_id=image['camera_id'],
                    session_id=Path(manifest['source_session']).name,clock_id=manifest['clock_id'],
                    timestamp_source=manifest['timestamp_source'],inference_clock_id='board_monotonic:'+original['boot_id'],
                    inference_interval_scope=clock['scope'],model_sha256=original['files']['model.opt.bin'],
                    input_mode='recorded',backend='pl-stream',frame_id=image['frame_id'],
                    capture_timestamp_ns=image['capture_timestamp_ns'],inference_start_ns=clock['start_ns'],inference_end_ns=clock['end_ns'],
                    pl_calls=110,forward_ms=pl['forward_times_ms'][0],coordinate_space='original_image_pixels',
                    source_size=image['source_size'],roi_xywh=image['roi_xywh'],joint_order='COCO17',keypoints=pl['keypoints'],
                    exposure_timestamp_verified=False,pairing_status='unpaired',
                    anomalies=['recorded_exposure_sync_unverified','fixed_roi','calibration_mount_unverified'],
                    input_jpeg_sha256=image['sha256'],host_sha256=original['files']['light_pose_image'],bit_sha256=original['bit_sha256'])
                validate(packet);messages.append(packet);forward.append(packet['forward_ms'])
            (out/'board_pose.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in messages),encoding='utf-8')
            result.update(complete=True,integer_heatmaps_exact=True,images=len(messages),pairs=len(messages)//2,
                outputs_archive_sha256=digest,board_pose_sha256=sha(out/'board_pose.jsonl'),
                scope='Recovered transport of36 already completed CPU/integer/PL runs. No re-inference, restart, new module or boot change. Recorded stereo functionality, not sustained stereo rate.')
        except Exception as error:result['recovery_error']=str(error);raise
        finally:
            console.close()
            (out/'recovered_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print('STEREO_TRANSPORT_RECOVERY_RECORD',out,flush=True)

if __name__=='__main__':main()
