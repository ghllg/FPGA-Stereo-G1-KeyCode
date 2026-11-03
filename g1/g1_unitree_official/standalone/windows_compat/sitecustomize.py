"""Windows timers/loopback plus an explicit local policy simulation adapter."""
import sys
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import threading
import time
import types

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    class PowerThrottling(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    power = PowerThrottling(1, 1, 0)
    kernel.SetProcessInformation(kernel.GetCurrentProcess(), 4, ctypes.byref(power), ctypes.sizeof(power))
    class RecurrentThread:
        def __init__(self, interval, target, name, args=(), kwargs=None):
            self.stop = threading.Event()
            # Native adapter publishes immediately after every control block;
            # periodic SDK threads provide a lower-cost idle heartbeat.
            interval = max(interval, .05) if os.path.basename(sys.argv[0]) == 'unitree_mujoco.py' else max(interval,.005)
            def loop():
                deadline = time.perf_counter()
                while not self.stop.is_set():
                    target(*args, **(kwargs or {}))
                    deadline += interval
                    time.sleep(max(0, deadline-time.perf_counter()))
                    if time.perf_counter()-deadline > .1: deadline = time.perf_counter()
            self.thread = threading.Thread(target=loop, name=name, daemon=True)
        def Start(self): self.thread.start()
        def Wait(self, timeout=None):
            self.stop.set()
            self.thread.join(timeout)
    module = types.ModuleType('unitree_sdk2py.utils.thread')
    module.RecurrentThread = RecurrentThread
    sys.modules[module.__name__] = module

from unitree_sdk2py.core import channel
channel.ChannelConfigAutoDetermine = '''<CycloneDDS><Domain Id="any">
<General><Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces>
<AllowMulticast>false</AllowMulticast></General>
<Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery>
</Domain></CycloneDDS>'''

if os.path.basename(sys.argv[0]) == 'unitree_mujoco.py':
    import glfw
    import mujoco.viewer
    # Use GLFW's native floating-window mode: focusing the input webpage no
    # longer hides the simulator behind the browser. Physics is unchanged.
    glfw.init()
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
    launch_passive = mujoco.viewer.launch_passive
    from viewer_snapshot import launch_snapshot_viewer
    def launch_viewer(*args, **kwargs):
        kwargs.setdefault('show_left_ui', True)
        kwargs.setdefault('show_right_ui', True)
        return launch_snapshot_viewer(launch_passive, *args, **kwargs)
    mujoco.viewer.launch_passive = launch_viewer
    sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0])))
    import policy_bridge
