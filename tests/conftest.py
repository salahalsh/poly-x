"""Shared fixtures.

The POLY-X service modules carry no Django dependency of their own, but they
live inside a Django app package whose ``__init__`` chain pulls in the app
registry. They are therefore loaded here by file path, which keeps the test
suite runnable in a bare environment (and therefore in CI) with nothing but
RDKit, NumPy, scikit-learn and pytest installed.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Repository root: overridable so the same suite runs from the paper's
# revision folder and from a checkout of the POLY-X repository.
def _find_repo() -> Path:
    """First location that actually contains ``poly_x/services``.

    The suite runs from three places: a checkout of the POLY-X repository, the
    paper's revision folder, and the working copy of the platform. Picking the
    first candidate that holds the modules keeps one conftest honest in all
    three, instead of silently skipping everything in two of them.
    """
    env = os.environ.get('POLYX_ROOT')
    here = Path(__file__).resolve()
    candidates = ([Path(env)] if env else []) + [
        here.parents[1],                       # checkout root: tests/ at top
        here.parents[2],                       # revision_R1/release/tests/
        Path(r'D:\myTools\Tool - InsilicoX Project\insilicox_web_app'),
    ]
    for c in candidates:
        if (c / 'poly_x' / 'services').is_dir():
            return c.resolve()
    return candidates[0].resolve()


REPO = _find_repo()

# A skipped test reads as a pass. In CI the suite must fail loudly if a module
# it is meant to exercise is absent, rather than reporting green having run
# nothing; set POLYX_STRICT=1 there.
STRICT = os.environ.get('POLYX_STRICT') == '1'


def _absent(msg):
    if STRICT:
        raise RuntimeError(f'POLYX_STRICT: {msg}')
    pytest.skip(msg)
SERVICES = REPO / 'poly_x' / 'services'


def load_service(name: str):
    path = SERVICES / f'{name}.py'
    if not path.exists():
        _absent(f'service module not found: {path}')
    spec = importlib.util.spec_from_file_location(f'polyx_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='session')
def gc_calc():
    return load_service('group_contribution').GroupContributionCalculator()


@pytest.fixture(scope='session')
def fingerprinter():
    return load_service('polymer_fingerprints').PolymerFingerprinter()


@pytest.fixture(scope='session')
def processability():
    return load_service('processability').ProcessabilityAssessor()


@pytest.fixture(scope='session')
def data_csv():
    p = REPO / 'poly_x' / 'data' / 'tg_training_data.csv'
    if not p.exists():
        # The collection is redistributed from Zenodo, not vendored here, so
        # this one may legitimately be unavailable even under POLYX_STRICT.
        pytest.skip(f'training data not found: {p}; fetch PolyMetriX from '
                    f'https://doi.org/10.5281/zenodo.14980914')
    return p


# The eight canonical homopolymers of the manuscript, with the Polymer
# Handbook / Van Krevelen reference values used in Table 1.
CANONICAL = {
    'PE':   ('[*]CC[*]', 195),
    'PP':   ('[*]CC(C)[*]', 253),
    'PS':   ('[*]CC(c1ccccc1)[*]', 373),
    'PVC':  ('[*]CC(Cl)[*]', 354),
    'PET':  ('[*]CCOC(=O)c1ccc(C(=O)O[*])cc1', 342),
    'PEO':  ('[*]CCO[*]', 206),
    'POM':  ('[*]CO[*]', 198),
    'PVOH': ('[*]CC(O)[*]', 358),
}
