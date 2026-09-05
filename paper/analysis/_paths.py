"""Locate the POLY-X tree without depending on where it was checked out.

Resolution order:

1. ``$POLYX_ROOT``, if set. Use this to run the analysis against a working copy
   of the platform rather than against the repository.
2. The repository root, found by walking up from this file. The repository
   carries ``poly_x/services`` and the training extract, so a bare checkout is
   enough for every step that does not need a third-party bulk dataset.
3. A platform working copy, if one happens to sit at the historical location.

``POLYX`` is the package directory, ``DATA`` the data directory, and
``SERVICES`` the directory the analysis loads prediction modules from by file
path (they carry no Django dependency of their own, but their package
``__init__`` chain does).
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve()


def _resolve() -> Path:
    env = os.environ.get('POLYX_ROOT')
    candidates = ([Path(env)] if env else []) + [
        _HERE.parents[2],                       # <repo>/paper/analysis/_paths.py
        _HERE.parents[3],
        Path(r'D:\myTools\Tool - InsilicoX Project\insilicox_web_app'),
    ]
    for c in candidates:
        if (c / 'poly_x' / 'services').is_dir():
            return c.resolve()
    raise SystemExit(
        'Could not locate a POLY-X tree containing poly_x/services. Set '
        'POLYX_ROOT to the directory that contains the poly_x package.')


ROOT = _resolve()
POLYX = ROOT / 'poly_x'
SERVICES = POLYX / 'services'
DATA = POLYX / 'data'
TRAINED = POLYX / 'trained_models'
DATA_CSV = DATA / 'tg_training_data.csv'
