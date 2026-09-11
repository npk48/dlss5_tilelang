"""Independent frozen reference and per-project compiler cache setup."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'reference'
MODEL=ROOT/'model'
os.environ.setdefault('TILELANG_CACHE_DIR',str(ROOT/'.cache/tilelang'))
sys.path.insert(0,str(REFERENCE))
