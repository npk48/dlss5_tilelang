"""Independent frozen reference and per-project compiler cache setup."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT/'reference'
os.environ.setdefault('TILELANG_CACHE_DIR',str(ROOT/'.cache/tilelang'))
sys.path.insert(0,str(REFERENCE))
