"""Stage isolated diagnostic build and test storage; never open cameras."""
import hashlib,io,json,re,socket,sys,tarfile,time,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/full_pipeline_exec_20261022/display_fix_20261022/startup_qualification';OUT.mkdir(parents=True,exist_ok=True);CAND=OUT/'candidate';CAND.mkdir(exist_ok=False)
BUILD=ROOT/'work/full_chain_20261022/build_ascii'
old=json.loads((ROOT/'outputs/completion_20261022/live_stage.json').read_text())
sys.path.insert(0,r'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000\pc')
from uart_console import Console
assets={'stereo_host':BUILD/'qualified_1080p_host/pose_stereo_pipeline_qualified_1080p','test_snapshots':BUILD/'test_diagnostic_snapshots'}
for name,path in assets.items():(CAND/name).write_bytes(path.read_bytes())
remote='/tmp/u02_qual_'+uuid.uuid4().hex[:12]
state=dict(complete=False,camera_opened=False,remote=remote,boot_id=old['boot_id'],files={k:v for k,v in old['files'].items() if k!='stereo_host'})
state['files'].update({n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in assets.items()})
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w:gz') as t:
    for name,path in assets.items():t.add(path,arcname=name)
raw=payload.getvalue()
def transfer(port,path):
    with socket.create_connection(('192.168.137.2',port),timeout=30,source_address=('192.168.137.1',0)) as sock,path.open('xb') as f:
        while True:
            part=sock.recv(65536)
            if not part:break
            f.write(part)
try:
    with (OUT/'snapshot_stage_uart.log').open('wb') as log:
        c=Console('COM22',log);send=c.send
        def paced(b):
            for i in range(0,len(b),24):send(b[i:i+24]);time.sleep(.004)
        c.send=paced
        def run(cmd,timeout=30):
            code,text=c.run(cmd,timeout);assert code==0,text;return text
        try:
            c.sync()
            assert re.search(r'(?m)^'+old['boot_id']+r'\r?$',run('cat /proc/sys/kernel/random/boot_id'))
            assert not re.search(r'(?m)^u02_',run('cat /proc/modules'))
            owners=run('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*|/dev/u02*) printf "OPEN:%s\\n" "$t";; esac; done; true')
            assert not re.search(r'(?m)^OPEN:',owners)
            run("! awk '$2 ~ /:46A2$/ && $4 == \"0A\" {found=1} END {exit !found}' /proc/net/tcp")
            run(f'mkdir {remote} && (timeout 30 nc -l -p 18082 > {remote}/assets.tgz; echo $? > {remote}/upload.exit) > {remote}/upload.log 2>&1 < /dev/null &')
            time.sleep(.3)
            with socket.create_connection(('192.168.137.2',18082),timeout=30,source_address=('192.168.137.1',0)) as sock:
                sock.sendall(raw);sock.shutdown(socket.SHUT_WR)
            for _ in range(30):
                code,_=c.run(f'test -s {remote}/upload.exit',5)
                if code==0:break
                time.sleep(.1)
            assert re.search(r'(?m)^0\r?$',run(f'cat {remote}/upload.exit'))
            assert hashlib.sha256(raw).hexdigest() in run(f'sha256sum {remote}/assets.tgz')
            run(f'cd {remote} && tar -xzf assets.tgz && chmod 700 stereo_host test_snapshots')
            for name,digest in old['files'].items():
                if name=='stereo_host':continue
                assert digest in run(f'sha256sum {old["remote"]}/{name}')
                run(f'cp {old["remote"]}/{name} {remote}/{name}')
            for name,digest in state['files'].items():assert digest in run(f'sha256sum {remote}/{name}')
            text=run(f'cd {remote} && ./test_snapshots fixture',30)
            assert 'SYNTHETIC_SNAPSHOT_STORAGE_TEST_PASS' in text
            run(f'cd {remote} && tar -czf fixture.tgz fixture.*.json*')
            digest=re.search(r'(?m)^([a-f0-9]{64})\s',run(f'sha256sum {remote}/fixture.tgz')).group(1)
            run(f'timeout 30 nc -l -p 18082 < {remote}/fixture.tgz > {remote}/download.log 2>&1 &')
            time.sleep(.2);transfer(18082,CAND/'synthetic_fixture.tgz')
            assert hashlib.sha256((CAND/'synthetic_fixture.tgz').read_bytes()).hexdigest()==digest
            state.update(complete=True,synthetic_storage_test_passed=True,fixture_archive_sha256=digest)
        finally:c.close()
finally:(OUT/'live_stage.json').write_text(json.dumps(state,indent=2))
with tarfile.open(CAND/'synthetic_fixture.tgz') as t:
    assert len(t.getmembers())==160
    for i in range(20):
        for side in ('left','right'):
            prefix=f'fixture.snapshot_{i}.{side}.json'
            meta=json.load(t.extractfile(prefix))
            assert meta['pair_id']==i and meta['camera_id']==side
            assert meta['roi']==([780,280,510,680] if side=='left' else [620,260,510,680])
            a=np.frombuffer(t.extractfile(prefix+'.input.f32').read(),np.float32)
            h=np.frombuffer(t.extractfile(prefix+'.heatmap').read(),np.float32)
            assert a.size==3*96*128 and np.all(a==(1 if i==0 else 7))
            assert h.size==17*32*24 and np.all(h==(2 if i==0 else 8))
(OUT/'snapshot_storage_verification.json').write_text(json.dumps(dict(passed=True,views=40,files=160,
    disabled_mode_tested=True,private_copy_tested=True,one_second_spacing_tested=True,bounded_20_pairs_tested=True,
    mismatched_pair_rejected=True,camera_opened=False,FPGA_inference=False,scope='Synthetic storage test executed on ARM board; not captured images'),indent=2))
print('QUALIFIED_STARTUP_STAGED_WITHOUT_CAMERAS',remote)
