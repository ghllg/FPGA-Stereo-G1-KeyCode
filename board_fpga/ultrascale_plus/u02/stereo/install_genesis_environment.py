"""Install the checked package plan into isolated E runtime, no old uninstall."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO=Path(__file__).resolve().parents[3]
U02=REPO/'ultrascale_plus/u02'
plan_record=U02/'reports/genesis_dependency_plan_1791804870118015800'
check=json.loads((plan_record/'space_check.json').read_text(encoding='utf-8'))
assert check['complete'] and not check['packages_installed']
assert hashlib.sha256((plan_record/'pip_plan.json').read_bytes()).hexdigest()==check['plan_sha256']
assert shutil.disk_usage(U02).free>check['conservative_install_and_temp_budget_bytes']+check['remaining_build_reserve_bytes']
environment=U02/'toolchain/genesis_runtime'
assert not (environment/'Lib/site-packages/genesis').exists(), 'Preserve previous installation'
vision=REPO/'windows_vision/.venv/Lib/site-packages'
snapshots={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in vision.glob('*.dist-info/RECORD')}
output=U02/('reports/genesis_install_'+str(time.time_ns()));output.mkdir()
variables=os.environ.copy()
variables.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
    PIP_CACHE_DIR=str(U02/'toolchain/genesis_cache/pip'),
    PYTHONPYCACHEPREFIX=str(U02/'toolchain/genesis_cache/pycache'))
command=[str(environment/'Scripts/python.exe'),'-m','pip','install','--no-deps','--ignore-installed',
         '--report',str(output/'install_report.json')]
command += [p['url']+'#sha256='+p['sha256'] for p in check['packages']]
state=dict(complete=False,environment=str(environment),old_vision_record_sha256=snapshots,
           dependency_plan_sha256=hashlib.sha256((plan_record/'space_check.json').read_bytes()).hexdigest(),
           existing_vision_environment_modified=False,new_torch_download=False)
def save():(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
try:
    with (output/'install.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=U02,env=variables,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        state['pid']=process.pid;save();code=process.wait()
    state.update(terminal=True,exit_code=code);save()
    assert code==0
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in snapshots.items())
    report=json.loads((output/'install_report.json').read_text(encoding='utf-8'))
    assert any(p['metadata']['name']=='genesis-world' and p['metadata']['version']=='1.4.0' for p in report['install'])
    state.update(complete=True,packages_installed=True,rendering_verified=False,
        scope='Checked hash-pinned packages installed only into separate E runtime with ignore-installed/no-deps. Existing vision RECORDs unchanged; no Genesis render or motion accuracy acceptance.')
finally:
    save();print('GENESIS_INSTALL_RECORD',output,flush=True)
