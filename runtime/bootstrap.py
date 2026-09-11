"""Independent frozen reference, model assets, and compiler-cache setup."""
from pathlib import Path
import json
import os
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reference'
MODEL = ROOT / 'model'
PRECOMPILED = ROOT / 'precompiled' / 'tilelang'

_cache_was_explicit = 'TILELANG_CACHE_DIR' in os.environ
CACHE = Path(os.environ.setdefault('TILELANG_CACHE_DIR', str(ROOT / '.cache/tilelang')))


def _seed_precompiled_cache():
    """Copy the qualified read-only bundle into the writable local cache once."""
    manifest = PRECOMPILED / 'bundle.json'
    if (_cache_was_explicit or sys.platform != 'win32' or not manifest.is_file()
            or platform.machine().lower() not in ('amd64', 'x86_64')):
        return
    info = json.loads(manifest.read_text(encoding='utf-8'))
    source = PRECOMPILED / info['tilelang_version']
    marker = CACHE / ('.seeded-' + info['bundle_id'])
    if marker.is_file() or not source.is_dir():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, CACHE / source.name, dirs_exist_ok=True)
    for old in CACHE.glob('.seeded-*'):
        old.unlink(missing_ok=True)
    marker.write_text(info['bundle_id'] + '\n', encoding='ascii')


_seed_precompiled_cache()
sys.path.insert(0, str(REFERENCE))
