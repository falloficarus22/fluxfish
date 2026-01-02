"""pytest configuration to fix import paths"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
python_dir = repo_root / "python"

if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
