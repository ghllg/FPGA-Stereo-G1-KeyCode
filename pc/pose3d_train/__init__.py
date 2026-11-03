"""PC-only calibrated stereo pose learning; run modules from the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'windows_vision'))
