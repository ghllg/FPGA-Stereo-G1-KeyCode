"""Request millisecond scheduling only for this real-time application process."""
import atexit
import sys


def enable_timer():
    if sys.platform != 'win32':return lambda:None
    import ctypes
    timer = ctypes.WinDLL('winmm')
    if timer.timeBeginPeriod(1) != 0:
        raise RuntimeError('Windows could not enable the 1ms application timer')
    active = True
    def close():
        nonlocal active
        if active:
            timer.timeEndPeriod(1)
            active = False
    atexit.register(close)
    return close
