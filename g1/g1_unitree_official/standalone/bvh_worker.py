"""Keep GMR computation out of the Python DDS callback/GIL thread."""
import multiprocessing
from pathlib import Path
import sys
import weakref
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1_realtime'))
from tracker import Tracker

def work(pipe, message):
    tracker = Tracker(message)
    pipe.send('ready')
    while True:
        try: values = pipe.recv()
        except EOFError: break
        if values is None: break
        try: pipe.send((True,tracker.solve(values)))
        except Exception as error: pipe.send((False,str(error)))

def close(pipe, process):
    try: pipe.send(None)
    except (BrokenPipeError, EOFError, OSError): pass
    process.join(1)
    if process.is_alive(): process.terminate()
    pipe.close()

class RemoteTracker:
    def __init__(self, message):
        local = Tracker(message)
        self.model, self.metadata = local.model, local.metadata
        context = multiprocessing.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=work,args=(child,message),daemon=True)
        self.process.start()
        child.close()
        if not self.pipe.poll(15) or self.pipe.recv() != 'ready':
            close(self.pipe,self.process)
            raise ValueError('BVH worker initialization failed')
        weakref.finalize(self,close,self.pipe,self.process)
    def solve(self, values):
        self.pipe.send(values)
        if not self.pipe.poll(15): raise ValueError('BVH worker timed out')
        success, result = self.pipe.recv()
        if not success: raise ValueError(result)
        return result
