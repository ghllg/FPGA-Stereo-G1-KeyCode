"""Export ten complete-network output references on the board; never access PL."""
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
import time

U02=Path(__file__).resolve().parents[1]
ROOT=U02.parents[1]
sys.path.insert(0,str(ROOT/'bringup/axu5evb_e_boot'))
from uart_console import Console
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    host=U02/'inference/build_fast_nearest/fp32_fixture_export_host/light_pose'
    assert sha(host)=='81368fca4a07213d0326a3cc2a04f3a402ea9e0ecacdcf56746152041e220d9b'
    out=U02/('reports/fp32_output_fixtures_'+str(time.time_ns()));out.mkdir()
    state=dict(complete=False,board_reference_only=True,pl_access=False,driver_loaded=False,
        reboot=False,startup_files_modified=False,host_sha256=sha(host))
    console=None
    try:
        with (out/'transfer.log').open('wb') as log:
            code=subprocess.call([sys.executable,str(U02/'board/transfer_test_tools.py'),'--inference','--stream-model',
                '--host',str(host)],cwd=U02,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert code==0
        matches=re.findall(r'(?m)^U02_TOOLS_TRANSFERRED (/tmp/u02_[a-f0-9]{32})\r?$',(out/'transfer.log').read_text());assert len(matches)==1
        remote=matches[0];state['remote']=remote
        with (out/'uart.log').open('wb') as log:
            console=Console('COM9',log);console.sync()
            code,text=console.run('uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/modules')
            assert code==0 and '6.6.40' in text and not re.search(r'(?m)^(?:u02_pose|u02_stream|u02_stream_resident|u02_ddr_test)\s+\d+\s',text)
            state['board_identity']=text
            names=[]
            for i in range(10):
                meta=json.loads((ROOT/f'board_light_pose/outputs/samples/{i}.json').read_text())
                geometry=' '.join(map(str,meta['source_size']+meta['roi_xywh']))
                code,text=console.run(f'cd {remote} && POSE_THREADS=4 POSE_WARMUP=0 POSE_REPEAT=1 POSE_QUANT_VECTOR=1 '
                    f'POSE_STREAM_FP32_FIXTURE={remote}/fp32_{i}.fixture timeout 90 ./light_pose stream-csim model.stream.param model.opt.bin '
                    f'{i}.f32 ref_{i}.json {geometry}',timeout=95)
                assert code==0 and 'LIGHT_POSE_OK backend=stream-csim' in text
                names += [f'fp32_{i}.fixture',f'ref_{i}.json',f'ref_{i}.json.heatmap']
                print('FP32_REAL_FULL_NETWORK_REFERENCE_EXPORTED',i,flush=True)
            code,text=console.run(f'cd {remote} && tar -czf fp32_refs.tar.gz '+' '.join(names)+' && sha256sum fp32_refs.tar.gz',timeout=60)
            assert code==0;digest=re.search(r'(?m)^([a-f0-9]{64})\s+fp32_refs.tar.gz\r?$',text).group(1)
            code,text=console.run(f'timeout 45 nc -l -p 18083 < {remote}/fp32_refs.tar.gz > /dev/null & sleep 1');assert code==0
            data=bytearray()
            with socket.create_connection(('192.168.50.2',18083),timeout=15) as connection:
                connection.settimeout(30)
                while True:
                    block=connection.recv(65536)
                    if not block:break
                    data.extend(block);assert len(data)<=96*1024*1024
            assert hashlib.sha256(data).hexdigest()==digest
            (out/'fp32_refs.tar.gz').write_bytes(data)
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                assert set(archive.getnames())==set(names)
                for member in archive.getmembers():
                    assert member.isfile() and member.size<=16*1024*1024
                    (out/member.name).write_bytes(archive.extractfile(member).read())
            for i in range(10):
                ref=U02/f'reports/stream_reference_1791697546871935000/stream_{i}.json'
                actual=json.loads((out/f'ref_{i}.json').read_text());expected=json.loads(ref.read_text())
                assert (out/f'ref_{i}.json.heatmap').read_bytes()==Path(str(ref)+'.heatmap').read_bytes()
                assert all(actual[k]==expected[k] for k in ('keypoints','source_size','roi_xywh'))
                assert actual['backend']=='stream-csim' and not actual['pl_acceleration']
            code,text=console.run('cat /proc/modules; cat /proc/sys/kernel/random/boot_id');assert code==0
            state.update(complete=True,archive_sha256=digest,ten_final_heatmaps_and_coco17_exact=True,
                fixture_sha256={str(i):sha(out/f'fp32_{i}.fixture') for i in range(10)},after=text,
                scope='Real board CPU integer references for every layer of ten complete networks. Candidate packet export only; no new PL implementation or speed result.')
    except Exception as error:state['error']=str(error);raise
    finally:
        if console:console.close()
        (out/'result.json').write_text(json.dumps(state,indent=2)+'\n');print('FP32_OUTPUT_FIXTURE_RECORD',out,flush=True)

if __name__=='__main__':main()
