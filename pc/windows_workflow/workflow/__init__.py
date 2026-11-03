"""Windows PC live workflow; existing offline and acquisition tools remain reusable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'windows_vision'))
