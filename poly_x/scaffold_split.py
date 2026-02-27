"""
Scaffold-based train/val/test splitting for polymers.

Uses Murcko scaffold of the repeat unit (after [*] -> [H] conversion)
to ensure structurally similar polymers stay in the same fold.

Reference:
    Bemis, G.W.; Murcko, M.A. The properties of known drugs.
    1. Molecular frameworks. J. Med. Chem. 1996, 39, 2887-2893.

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""

import logging
import random
from collections import defaultdict
from typing import Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


class PolymerScaffoldSplitter:
    """
    Scaffold-based train/val/test split for polymers.

    Uses Murcko scaffold of the repeat unit to ensure structurally
    similar polymers stay in the same fold.
    """

    def __init__(self, train_frac: float = 0.8, val_frac: float = 0.1,
                 test_frac: float = 0.1, random_seed: int = 42):
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = random_seed

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by Murcko scaffold.

        Args:
            df: DataFrame with 'smiles' and 'target' columns

        Returns:
            (train_df, val_df, test_df)
        """
        logger.info("Performing scaffold split on %d polymers...", len(df))

        # Compute scaffolds
        scaffolds = {}
        for idx, row in df.iterrows():
            try:
                clean = row['smiles'].replace('[*]', '[H]')
                mol = Chem.MolFromSmiles(clean)
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False)
                else:
                    scaffold = f'_invalid_{idx}'
            except Exception:
                scaffold = f'_error_{idx}'

            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(idx)

        # Group scaffolds by size, then shuffle within each size-group
        rng = random.Random(self.seed)
        size_groups = defaultdict(list)
        for group in scaffolds.values():
            size_groups[len(group)].append(group)

        scaffold_sets = []
        for size in sorted(size_groups.keys(), reverse=True):
            bucket = sorted(size_groups[size], key=lambda g: min(g))
            rng.shuffle(bucket)
            scaffold_sets.extend(bucket)

        n_total = len(df)
        n_train = int(n_total * self.train_frac)
        n_val = int(n_total * self.val_frac)

        train_idx, val_idx, test_idx = [], [], []
        for group in scaffold_sets:
            if len(train_idx) < n_train:
                train_idx.extend(group)
            elif len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        # Post-split assertions
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        assert not (train_set & val_set), "Scaffold leak: train/val overlap"
        assert not (train_set & test_set), "Scaffold leak: train/test overlap"
        assert not (val_set & test_set), "Scaffold leak: val/test overlap"
        assert len(train_set) + len(val_set) + len(test_set) == n_total, \
            f"Split lost samples: {len(train_set)}+{len(val_set)}+{len(test_set)} != {n_total}"

        # Verify scaffold-level disjointness
        train_scaffolds = {s for s, indices in scaffolds.items()
                          if any(i in train_set for i in indices)}
        val_scaffolds = {s for s, indices in scaffolds.items()
                        if any(i in val_set for i in indices)}
        test_scaffolds = {s for s, indices in scaffolds.items()
                         if any(i in test_set for i in indices)}
        assert not (train_scaffolds & test_scaffolds), \
            "Same scaffold in both train and test"
        assert not (train_scaffolds & val_scaffolds), \
            "Same scaffold in both train and val"

        train_df = df.loc[df.index.isin(train_idx)].reset_index(drop=True)
        val_df = df.loc[df.index.isin(val_idx)].reset_index(drop=True)
        test_df = df.loc[df.index.isin(test_idx)].reset_index(drop=True)

        logger.info("Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
                     len(train_df), 100 * len(train_df) / n_total,
                     len(val_df), 100 * len(val_df) / n_total,
                     len(test_df), 100 * len(test_df) / n_total)

        return train_df, val_df, test_df
