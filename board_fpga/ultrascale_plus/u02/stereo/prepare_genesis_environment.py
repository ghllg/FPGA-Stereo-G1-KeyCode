"""Isolated runtime dependency plan; read-only reuse of existing vision packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO=Path(__file__).resolve().parents[3]
U02=REPO/'ultrascale_plus/u02'
vision=REPO/'windows_vision/.venv'
environment=U02/'toolchain/genesis_runtime'
assert not environment.exists(), 'Preserve existing environment'
assert shutil.disk_usage(U02).free>10*1024**3
cache=U02/'toolchain/genesis_cache';cache.mkdir(exist_ok=True)
variables=os.environ.copy()
variables.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),PIP_CACHE_DIR=str(cache/'pip'))
record=U02/('reports/genesis_dependency_plan_'+str(time.time_ns()));record.mkdir()
state=dict(complete=False,packages_installed=False,existing_vision_environment_modified=False,
           environment=str(environment),vision_packages_reused_readonly=str(vision/'Lib/site-packages'))
try:
    subprocess.run([str(vision/'Scripts/python.exe'),'-m','venv',str(environment)],env=variables,check=True)
    # No writes to the old environment. New packages, if subsequently approved
    # by the recorded dependency/space check, live in this independent venv.
    (environment/'Lib/site-packages/vision_readonly.pth').write_text(str(vision/'Lib/site-packages')+'\n')
    python=environment/'Scripts/python.exe'
    with (record/'pip_plan.log').open('wb') as log:
        process=subprocess.Popen([str(python),'-m','pip','install','--dry-run','--report',str(record/'pip_plan.json'),
            'genesis-world==1.4.0'],env=variables,cwd=U02,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        state['pid']=process.pid
        (record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        code=process.wait()
    state.update(terminal=True,exit_code=code)
    assert code==0
    plan=json.loads((record/'pip_plan.json').read_text(encoding='utf-8'))
    state.update(complete=True,planned_packages=[dict(name=p['metadata']['name'],version=p['metadata']['version']) for p in plan['install']],
        scope='Dependency resolution only; isolated venv with read-only existing Torch/vision packages. No Genesis install, renderer or visualization acceptance.')
finally:
    (record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('GENESIS_DEPENDENCY_PLAN_RECORD',record,flush=True)
