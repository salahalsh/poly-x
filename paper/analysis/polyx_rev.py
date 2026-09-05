"""
Shared utilities for the POLY-X JCAMD revision (R1) re-analysis.

Every number reported in the revised manuscript must be produced by one of the
numbered scripts in this directory, all of which import from here. Nothing is
hard-coded and nothing is illustrative.

Design decisions taken in response to the referees:

* **One canonicalisation** (Reviewer 2, Major 7).  PSMILES are parsed by RDKit
  *with the wildcards intact* (``[*]`` becomes a dummy atom), so that a single
  canonical form exists independently of the downstream capping convention.
  Deduplication happens on that canonical form, not on the raw string.
* **Capping is an explicit, ablatable choice**, not an implicit one.  The three
  conventions actually present in the submitted code base (methyl cap, hydrogen
  cap, raw wildcard) are all available here and are compared in script 02.
* **Target-dependent preprocessing is fitted on training data only**
  (Reviewer 2, Major 7).  The 3-sigma outlier filter is applied per split.
* **The scaffold splitter is genuinely randomised** (Reviewer 2, Major 3 and 6).
  The submitted splitter shuffled the scaffold groups and then re-sorted them by
  ``(-size, min_index)``, which cancels the shuffle: every seed produced an
  identical split.  ``scaffold_split`` below shuffles *within* equal-size
  cohorts, so distinct seeds give genuinely distinct partitions.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
POLYX = Path(r"D:\myTools\Tool - InsilicoX Project\insilicox_web_app\poly_x")
DATA_CSV = POLYX / "data" / "tg_training_data.csv"
TG_MODELS = POLYX / "trained_models" / "tg"
PB_MODELS = POLYX / "trained_models" / "polybert"

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
OUT = REVISION / "outputs"
FIGS = REVISION / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 123, 456, 789, 2024)


# --------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------
def canonical_psmiles(psmiles: str) -> str | None:
    """Canonical PSMILES with wildcards preserved as dummy atoms.

    ``[*]CC[*]`` and ``[*]C([H])([H])C[*]``-style variants collapse onto one
    string; ``None`` is returned for anything RDKit cannot parse.
    """
    if not isinstance(psmiles, str) or not psmiles.strip():
        return None
    mol = Chem.MolFromSmiles(psmiles.strip())
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def to_mol(psmiles: str, cap: str = "methyl") -> Chem.Mol | None:
    """Convert a PSMILES to an RDKit molecule under an explicit capping rule.

    cap = "methyl"   -> ``[*]`` becomes ``[CH3]``  (Tier 1 convention)
    cap = "hydrogen" -> ``[*]`` becomes ``[H]``    (Tier 2 convention as shipped)
    cap = "wildcard" -> wildcards retained as dummy atoms (no substitution)
    """
    s = psmiles.strip()
    if cap == "methyl":
        s = s.replace('[*]', '[CH3]').replace('*', '[CH3]')
    elif cap == "hydrogen":
        s = s.replace('[*]', '[H]').replace('*', '[H]')
    elif cap != "wildcard":
        raise ValueError(f"unknown cap {cap!r}")
    return Chem.MolFromSmiles(s)


def murcko_scaffold(psmiles: str, cap: str = "hydrogen") -> str:
    """Bemis-Murcko scaffold of the capped repeat unit ('' for acyclic)."""
    mol = to_mol(psmiles, cap=cap)
    if mol is None:
        return "_unparseable"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return "_error"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class CleaningLog:
    """Per-stage row counts, so the dataset accounting is fully transparent
    (Reviewer 2, Minor 1)."""
    raw: int = 0
    after_nan: int = 0
    after_parse: int = 0
    after_canonical_dedup: int = 0
    removed_unparseable: tuple = ()
    removed_duplicates: tuple = ()

    def to_dict(self):
        return asdict(self)


def load_dataset(path: Path = DATA_CSV):
    """Load and clean the Tg dataset **without** any target-dependent step.

    The 3-sigma outlier filter of the submitted pipeline is deliberately NOT
    applied here: it is a target-dependent operation and is therefore deferred
    to the training partition only (see ``fit_outlier_filter``).
    """
    df = pd.read_csv(path)
    log = CleaningLog(raw=len(df))

    df = df.rename(columns={'PSMILES': 'smiles', 'Tg': 'target'})
    df = df[['smiles', 'target']].copy()
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['smiles', 'target'])
    log.after_nan = len(df)

    df['canonical'] = df['smiles'].apply(canonical_psmiles)
    bad = df[df['canonical'].isna()]
    log.removed_unparseable = tuple(bad['smiles'].tolist())
    df = df[df['canonical'].notna()].copy()
    log.after_parse = len(df)

    dup_mask = df.duplicated(subset=['canonical'], keep='first')
    log.removed_duplicates = tuple(
        df.loc[dup_mask, ['smiles', 'target']].itertuples(index=False, name=None))
    df = df[~dup_mask].copy()
    log.after_canonical_dedup = len(df)

    df = df.reset_index(drop=True)
    return df, log


def fit_outlier_filter(y_train: np.ndarray, n_sigma: float = 3.0):
    """Return (lo, hi) bounds estimated from the *training* target only."""
    mu, sd = float(np.mean(y_train)), float(np.std(y_train, ddof=1))
    return mu - n_sigma * sd, mu + n_sigma * sd


# --------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------
def scaffold_groups(df: pd.DataFrame, cap: str = "hydrogen") -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, smi in zip(df.index, df['smiles']):
        groups.setdefault(murcko_scaffold(smi, cap=cap), []).append(int(idx))
    return groups


def scaffold_split(df, seed=42, fracs=(0.8, 0.1, 0.1), cap="hydrogen",
                   groups=None):
    """Genuinely seed-dependent scaffold split.

    Groups are ordered by decreasing size; within each equal-size cohort the
    order is shuffled with the given seed. Groups are then filled sequentially
    into train, validation and test.
    """
    groups = groups if groups is not None else scaffold_groups(df, cap=cap)
    by_size: dict[int, list[list[int]]] = {}
    for g in groups.values():
        by_size.setdefault(len(g), []).append(sorted(g))

    rng = random.Random(seed)
    ordered: list[list[int]] = []
    for size in sorted(by_size, reverse=True):
        cohort = sorted(by_size[size], key=lambda g: g[0])
        rng.shuffle(cohort)
        ordered.extend(cohort)

    n = len(df)
    n_train, n_val = int(n * fracs[0]), int(n * fracs[1])
    tr: list[int] = []
    va: list[int] = []
    te: list[int] = []
    for g in ordered:
        if len(tr) < n_train:
            tr.extend(g)
        elif len(va) < n_train + n_val - len(tr) and len(va) < n_val:
            va.extend(g)
        else:
            te.extend(g)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def butina_clusters(df, cutoff=0.65, cap="hydrogen"):
    """Polymer-aware clustering on ECFP4 Tanimoto distance (Reviewer 2, Major 6).

    Returns a list of index lists. ``cutoff`` is the *distance* threshold.
    """
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in df['smiles']:
        m = to_mol(smi, cap=cap)
        fps.append(gen.GetFingerprint(m) if m is not None else None)
    valid = [i for i, f in enumerate(fps) if f is not None]
    fpl = [fps[i] for i in valid]

    dists = []
    for i in range(1, len(fpl)):
        sims = DataStructs.BulkTanimotoSimilarity(fpl[i], fpl[:i])
        dists.extend(1.0 - s for s in sims)
    raw = Butina.ClusterData(dists, len(fpl), cutoff, isDistData=True)
    return [[int(df.index[valid[j]]) for j in cluster] for cluster in raw]


def cluster_split(df, clusters, seed=42, fracs=(0.8, 0.1, 0.1)):
    """Leave-clusters-out split from a precomputed clustering."""
    by_size: dict[int, list[list[int]]] = {}
    for c in clusters:
        by_size.setdefault(len(c), []).append(sorted(c))
    rng = random.Random(seed)
    ordered: list[list[int]] = []
    for size in sorted(by_size, reverse=True):
        cohort = sorted(by_size[size], key=lambda g: g[0])
        rng.shuffle(cohort)
        ordered.extend(cohort)
    n = sum(len(c) for c in clusters)
    n_train, n_val = int(n * fracs[0]), int(n * fracs[1])
    tr, va, te = [], [], []
    for g in ordered:
        if len(tr) < n_train:
            tr.extend(g)
        elif len(va) < n_val:
            va.extend(g)
        else:
            te.extend(g)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def random_split(df, seed=42, fracs=(0.8, 0.1, 0.1)):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train, n_val = int(len(idx) * fracs[0]), int(len(idx) * fracs[1])
    return (np.sort(idx[:n_train]),
            np.sort(idx[n_train:n_train + n_val]),
            np.sort(idx[n_train + n_val:]))


# --------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------
def morgan_matrix(smiles, radius=2, n_bits=2048, counts=False, cap="hydrogen"):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = to_mol(smi, cap=cap)
        if m is None:
            continue
        out[i] = (gen.GetCountFingerprintAsNumPy(m) if counts
                  else gen.GetFingerprintAsNumPy(m)).astype(np.float32)
    return out


_DESC = [d for d in Descriptors._descList
         if not d[0].startswith('fr_')]  # drop 1-hot fragment counts


def rdkit_descriptor_matrix(smiles, cap="hydrogen"):
    """2-D RDKit descriptors: a physically-interpretable non-fingerprint
    baseline standing in for the PolyMetriX hierarchical descriptors."""
    out = np.zeros((len(smiles), len(_DESC)), dtype=np.float64)
    for i, smi in enumerate(smiles):
        m = to_mol(smi, cap=cap)
        if m is None:
            continue
        for j, (_, fn) in enumerate(_DESC):
            try:
                v = fn(m)
                out[i, j] = v if np.isfinite(v) else 0.0
            except Exception:
                out[i, j] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # A few RDKit descriptors (notably Ipc) grow factorially and overflow a
    # float32 cast inside scikit-learn, which would silently corrupt this
    # baseline. Clip to a range that float32 represents exactly.
    return np.clip(out, -1e30, 1e30)


DESC_NAMES = [n for n, _ in _DESC]


# --------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def optimize_weight(rf_val, gb_val, y_val, grid=None):
    """Grid search of the RF/GB blend weight on the validation split."""
    grid = np.arange(0.0, 1.0001, 0.05) if grid is None else grid
    blends = grid[:, None] * rf_val[None, :] + (1 - grid)[:, None] * gb_val[None, :]
    sse = ((blends - y_val[None, :]) ** 2).sum(axis=1)
    return float(grid[int(np.argmin(sse))])


def paired_bootstrap_delta(y, pred_a, pred_b, metric='r2', n_boot=10000, seed=0):
    """Paired bootstrap over test polymers for metric(A) - metric(B).

    Returns (delta_observed, lo95, hi95, p_two_sided).
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    fn = r2_score if metric == 'r2' else mean_absolute_error
    obs = fn(y, pred_a) - fn(y, pred_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = fn(y[idx], pred_a[idx]) - fn(y[idx], pred_b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(p, 1.0))


def dump(name: str, obj) -> Path:
    path = OUT / name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[write] {path}")
    return path


def scaffold_split_as_published(df, seed=42, fracs=(0.8, 0.1, 0.1),
                                cap="hydrogen", groups=None):
    """Faithful replica of the splitter used for the submitted manuscript.

    Retained so that the published numbers can be reproduced exactly and so
    that the effect of the defect can be quantified. The defect: the shuffle on
    line 3 below is immediately cancelled by the re-sort on line 4, because
    ``(-len(g), min(g))`` is a total order on the groups. Every seed therefore
    yields an identical partition, and same-size groups (2,398 of which are
    singletons) are assigned in dataset row order rather than at random.
    """
    groups = groups if groups is not None else scaffold_groups(df, cap=cap)
    sets = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    rng = random.Random(seed)
    rng.shuffle(sets)                                     # shuffled ...
    sets = sorted(sets, key=lambda g: (-len(g), min(g)))  # ... and un-shuffled

    n = len(df)
    n_train, n_val = int(n * fracs[0]), int(n * fracs[1])
    tr, va, te = [], [], []
    for g in sets:
        if len(tr) < n_train:
            tr.extend(g)
        elif len(va) < n_val:
            va.extend(g)
        else:
            te.extend(g)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))
