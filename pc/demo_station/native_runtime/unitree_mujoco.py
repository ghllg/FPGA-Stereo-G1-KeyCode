"""Full-body candidate viewer overlay; execute the frozen native entry once.

Keep this basename: the unchanged embedded-runtime sitecustomize uses it to
install native viewer/DDS hooks before this script starts.
"""
from pathlib import Path
import sys


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from demo_station.candidate import G1
    import viewer_snapshot
    from demo_station.native_viewer_snapshot import SnapshotViewer
    viewer_snapshot.SnapshotViewer=SnapshotViewer
    original=G1/'g1_unitree_official/standalone/simulate_python/unitree_mujoco.py'
    sys.path.insert(0,str(original.parent))
    sys.argv[0]=str(original)
    # Native threads outlive the entry script and resolve locker/viewer through
    # sys.modules['__main__']; runpy's temporary module would disappear on return.
    namespace=sys.modules['__main__'].__dict__
    namespace.update(__file__=str(original),__package__=None,__spec__=None)
    exec(compile(original.read_bytes(),str(original),'exec'),namespace)


if __name__=='__main__':main()
