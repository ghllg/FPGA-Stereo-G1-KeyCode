"""Run real left/right JPEGs with frozen board Host; no PC 2D inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys
import tarfile
import time
import uuid

REPO=Path(__file__).resolve().parents[3]
U02=REPO/'ultrascale_plus/u02'
sys.path.insert(0,str(REPO/'bringup/axu5evb_e_boot'))
sys.path.insert(0,str(U02/'inference'))
from uart_console import Console
from stream_artifacts import stream_artifacts
from validate_image_pipeline_resident import connect_file_server
from board_pose_protocol import SCHEMA,validate

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path)
    args=parser.parse_args()
    inputs=args.inputs.resolve();manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['input_mode']=='recorded'
    runtime=json.loads((U02/'inference/stream_ingress_parallel_affine_runtime.json').read_text())
    paths=stream_artifacts('ingress');proof=json.loads(paths['proof'].read_text())
    assert sha(paths['bit'])==runtime['bit_sha256']==proof['bit_sha256']
    assert all(proof[k] for k in ('clock_and_reset_observed','ps_pl_initialization_verified','axi_control_response_verified_without_arm'))
    host=U02/runtime['image_host'];driver=U02/runtime['driver']
    assert sha(host)==runtime['image_host_sha256'] and sha(driver)==runtime['driver_sha256']
    assert len(manifest['images'])<=60
    files=[host,driver,U02/'reports/u02_run_image_clock',REPO/'board_light_pose/models/model.opt.bin',
           U02/runtime['cpu_param'],U02/runtime['stream_param'],
           U02/'inference/build_fast_nearest/resident_packet_host/test_stream_transport',
           U02/'reports/stream_fixtures_1791657207786097100/fixture_0.json.fixture']
    for image in manifest['images']:
        assert re.fullmatch(r'p\d+_(left|right)\.jpg',image['file'])
        assert sha(inputs/image['file'])==image['sha256']
        files.append(inputs/image['file'])
    hashes={p.name:sha(p) for p in files}
    assert hashes['model.opt.bin']==runtime['model_weights_sha256']
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as archive:
        for path in files:archive.add(path,arcname=path.name)
    raw=payload.getvalue();assert len(raw)<32*1024*1024
    token=uuid.uuid4().hex;remote='/tmp/u02_'+token
    out=U02/('reports/stereo_recorded_board_'+str(time.time_ns()));out.mkdir()
    result=dict(complete=False,input_mode='recorded',inputs=str(inputs),remote=remote,
        input_manifest_sha256=sha(inputs/'manifest.json'),files=hashes,bit_sha256=runtime['bit_sha256'],
        driver_loaded=False,driver_unloaded=False,whole_network_on_board=True,boot_id=proof['boot_id'],runs=[])
    names=[]
    with (out/'uart.log').open('wb') as log:
        console=Console('COM9',log)
        try:
            console.sync()
            def checked(command,timeout=30):
                code,text=console.run(command,timeout=timeout)
                assert code==0,(command,code,text[-1200:])
                return text
            identity=checked('uname -m; uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/cmdline; cat /proc/modules')
            assert proof['boot_id'] in identity and '6.6.40' in identity and 'clk_ignore_unused' in identity and 'cpuidle.off=1' in identity
            assert not re.search(r'(?m)^u02_\w+\s+\d+\s',identity)
            result['identity']=identity
            registers=checked('devmem 0xff5e00c0 32; devmem 0xfd1a0100 32; devmem 0xfd615000 32; devmem 0xff413030 32; devmem 0xff413040 32; devmem 0xff5e0030 32; devmem 0xff5e0040 32')
            values=[int(x,16) for x in re.findall(r'(?m)^(0x[0-9A-Fa-f]+)\r*$',registers)]
            assert len(values)==7
            clk,reset,width,request,ack,pll,status=values
            assert clk&0x013f3f07==0x01010b02 and reset&0x1f80==0 and width&0x300==0x200 and (request|ack)&3==0 and pll==0x14000 and status&2==2
            result['known_ps_readonly_registers']=values
            ready='__STEREO_READY_'+token;done='__STEREO_DONE_'+token
            console.send((f"mkdir {remote}; timeout 45 nc -l -p 18086 > {remote}/tools.tar.gz < /dev/null & u02_stereo_rx=$!; sleep 1; cat /proc/net/tcp; printf '\\n{ready}\\n'; wait $u02_stereo_rx; printf '\\n{done}:%s\\n' \"$?\"\n").encode())
            listener,_=console.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
            assert re.search(rb'\b[0-9A-F]{8}:46A6\s+[0-9A-F]{8}:[0-9A-F]{4}\s+0A\b',listener)
            with connect_file_server(('192.168.50.2',18086)) as connection:
                connection.sendall(raw);connection.shutdown(1)
            _,match=console.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',30)
            assert int(match.group(1))==0
            assert hashlib.sha256(raw).hexdigest() in checked(f'sha256sum {remote}/tools.tar.gz')
            checked(f'cd {remote} && tar -xzf tools.tar.gz')
            text=checked(f'cd {remote} && chmod 700 light_pose_image test_stream_transport u02_run_image_clock && sha256sum '+' '.join(hashes))
            assert all(re.search(r'(?m)^'+digest+r'\s+\*?'+re.escape(name)+r'\r?$',text) for name,digest in hashes.items())
            checked(f'cd {remote} && ./u02_run_image_clock clock_probe.json /bin/true')
            names.append('clock_probe.json')
            for mode in ('cpu','stream-csim','pl-stream'):
                if mode=='pl-stream':
                    checked(f'insmod {remote}/u02_stream_resident.ko bridge_checked=1 cached_dma=1 poll_spin_us=1000',15)
                    result['driver_loaded']=True
                    checked('cat /sys/module/u02_stream_resident/parameters/cached_dma; cat /sys/module/u02_stream_resident/parameters/poll_spin_us')
                    gate=checked(f'timeout 180 {remote}/test_stream_transport {remote}/fixture_0.json.fixture',185)
                    assert 'U02_RESIDENT_DMA_NUMERIC_TIMEOUT_RECOVERY_PASSED groups=110 rounds=2' in gate
                    result['transport_gate']=gate
                for image in manifest['images']:
                    prefix=mode.replace('-','_')+'_'+image['file'][:-4]
                    geometry=' '.join(map(str,image['source_size']+image['roi_xywh']))
                    param='model.opt.param' if mode=='cpu' else 'model.stream.param'
                    command=f'cd {remote} && POSE_IMAGE_INPUT=1 POSE_THREADS=4 POSE_WARMUP=0 POSE_REPEAT=1 POSE_QUANT_VECTOR=1 LIGHT_PL_DW=1 POSE_PL_CONV=1 timeout 60 ./u02_run_image_clock {prefix}.clock.json ./light_pose_image {mode} {param} model.opt.bin {image["file"]} {prefix}.json {geometry}'
                    text=checked(command,65);assert 'LIGHT_POSE_OK' in text
                    result['runs'].append(dict(mode=mode,file=image['file'],prefix=prefix,uart_output=text))
                    names.extend([prefix+'.json',prefix+'.json.heatmap',prefix+'.json.input.f32',prefix+'.clock.json'])
                    print('BOARD_STEREO_IMAGE_COMPLETE',mode,image['file'],flush=True)
            checked('rmmod u02_stream_resident && cat /proc/modules')
            result['driver_unloaded']=True
            # Keep the UART shell line below its input limit. Expansion occurs
            # on the board; exact archive membership is verified after fetch.
            text=checked(f'cd {remote} && tar -czf stereo_outputs.tar.gz clock_probe.json cpu_*.json* stream_csim_*.json* pl_stream_*.json* && sha256sum stereo_outputs.tar.gz')
            digest=re.search(r'(?m)^([a-f0-9]{64})\s+stereo_outputs.tar.gz\r?$',text).group(1)
            checked(f'timeout 45 nc -l -p 18087 < {remote}/stereo_outputs.tar.gz > /dev/null & sleep 1')
            data=bytearray()
            with connect_file_server(('192.168.50.2',18087)) as connection:
                while True:
                    chunk=connection.recv(65536)
                    if not chunk:break
                    data.extend(chunk);assert len(data)<24*1024*1024
            assert hashlib.sha256(data).hexdigest()==digest
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                assert sorted(archive.getnames())==sorted(names)
                for member in archive.getmembers():
                    assert member.isfile() and member.size<1024*1024
                    (out/member.name).write_bytes(archive.extractfile(member).read())
            probe=json.loads((out/'clock_probe.json').read_text())
            assert probe['child_exit_code']==0 and probe['end_ns']>=probe['start_ns']>0
            messages=[]
            for image in manifest['images']:
                suffix=image['file'][:-4]
                integer=out/('stream_csim_'+suffix+'.json')
                pl=out/('pl_stream_'+suffix+'.json')
                record=json.loads(pl.read_text());reference=json.loads(integer.read_text())
                assert pl.with_suffix('.json.heatmap').read_bytes()==integer.with_suffix('.json.heatmap').read_bytes()
                assert record['keypoints']==reference['keypoints']
                assert len({sha(out/(mode+'_'+suffix+'.json.input.f32')) for mode in ('cpu','stream_csim','pl_stream')})==1
                assert len(record['forward_profiles'])==1 and record['forward_profiles'][0]['calls']==110
                clock=json.loads((out/('pl_stream_'+suffix+'.clock.json')).read_text())
                assert clock['child_exit_code']==0
                packet=dict(schema=SCHEMA,whole_network_on_board=True,camera_id=image['camera_id'],
                    session_id=Path(manifest['source_session']).name,clock_id=manifest['clock_id'],
                    timestamp_source=manifest['timestamp_source'],inference_clock_id='board_monotonic:'+proof['boot_id'],
                    inference_interval_scope=clock['scope'],model_sha256=hashes['model.opt.bin'],input_mode='recorded',
                    backend='pl-stream',frame_id=image['frame_id'],capture_timestamp_ns=image['capture_timestamp_ns'],
                    inference_start_ns=clock['start_ns'],inference_end_ns=clock['end_ns'],pl_calls=110,
                    forward_ms=record['forward_times_ms'][0],coordinate_space='original_image_pixels',
                    source_size=image['source_size'],roi_xywh=image['roi_xywh'],joint_order='COCO17',
                    keypoints=record['keypoints'],exposure_timestamp_verified=False,pairing_status='unpaired',
                    anomalies=['recorded_exposure_sync_unverified','fixed_roi','calibration_mount_unverified'],
                    input_jpeg_sha256=image['sha256'],host_sha256=sha(host),bit_sha256=runtime['bit_sha256'])
                if 'pairing_timestamp_ns' in image:
                    for key in ('pairing_timestamp_ns','pairing_clock_id','time_alignment'):packet[key]=image[key]
                    packet['anomalies'].append('offline_content_offset_not_exposure_sync')
                validate(packet);messages.append(packet)
            (out/'board_pose.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in messages),encoding='utf-8')
            result.update(complete=True,integer_heatmaps_exact=True,images=len(messages),pairs=len(messages)//2,
                output_archive_sha256=digest,board_pose_sha256=sha(out/'board_pose.jsonl'),
                forward_median_ms=statistics.median(p['forward_ms'] for p in messages),
                scope='Six recorded pairs, separately launched frozen complete board inference. No sustained stereo rate, exposure sync, physical calibration or Genesis acceptance.')
        except Exception as error:
            result['error']=str(error);raise
        finally:
            console.close()
            (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print('BOARD_STEREO_RECORD',out,flush=True)

if __name__=='__main__':main()
