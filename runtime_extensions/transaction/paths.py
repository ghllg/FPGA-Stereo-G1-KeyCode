from pathlib import Path
import sys

TASK = Path('C:\\Users\\developer\\Documents\\Codex\\2026-10-23\\new-chat')
PC = Path(r'C:\Users\developer\Documents\Codex\2026-10-20\iq\work\full_pipeline_exec_20261022')
BRIDGE = Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020\g1_stereo_bridge')
OLD = TASK/'outputs/offline_closure_20261024'
OUT = TASK/'outputs/remaining_failures_20261024/semantic'
sys.path[:0] = [str(PC), str(BRIDGE)]

