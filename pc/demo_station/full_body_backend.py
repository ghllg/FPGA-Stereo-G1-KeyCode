"""Original full-body pipeline with a separately identified official-model preparation view."""
from pathlib import Path
import subprocess
import time
import uuid
import backend as production


class FullBodyCheckBackend(production.Backend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            command=[str(production.G1/'runtime/python.exe'),'-B','-X','utf8',
                str(Path(__file__).with_name('full_body_preparation.py')),'--root',str(production.G1),
                '--output',str(self.output),'--session',uuid.uuid4().hex,'--not-before-ns',str(self.start_ns)]
            result=production.subprocess.run(command,cwd=production.G1,env=production.launcher.environment(),
                capture_output=True,text=True,encoding='utf8',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
            (self.output/'full_body_preparation_stdout.txt').write_text(result.stdout,encoding='utf8')
            (self.output/'full_body_preparation_stderr.txt').write_text(result.stderr,encoding='utf8')
            if result.returncode:raise RuntimeError('Official model preparation failed: '+result.stderr[-1500:])
            self.prepared_ns=time.perf_counter_ns()
        except Exception:
            self.close()
            raise
