"""
Dataset Loaders for Polymer Property Training

Supports any CSV of PSMILES + property columns. The loader class is named
``PI1MLoader`` for historical reasons (an early PI1M prototype); the shipped
Tg models are trained on the PolyMetriX curated experimental Tg collection
(7,367 polymers, Zenodo 10.5281/zenodo.14980914), not on PI1M. The submitted
manuscript reported 7,365 after a global three-sigma filter that the revision
removed.

Includes data cleaning, outlier removal, and scaffold splitting.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PI1MLoader:
    """
    Load and preprocess a polymer property CSV.

    Named for an early PI1M prototype; the shipped Tg models use PolyMetriX.

    Expected CSV format:
        PSMILES, Tg (K), [optional: Tm, Td, density, ...]

    Cleaning steps:
        1. Drop rows with NaN target
        2. Validate PSMILES parses with RDKit
        3. Remove 3-sigma outliers on target property
        4. Deduplicate by canonical PSMILES
    """

    def __init__(self, data_path: str, target_column: str = 'Tg',
                 smiles_column: str = 'PSMILES'):
        self.data_path = data_path
        self.target_column = target_column
        self.smiles_column = smiles_column

    def load(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load and clean the dataset.

        Returns:
            DataFrame with columns ['smiles', 'target'] (cleaned)
        """
        logger.info("Loading dataset from %s ...", self.data_path)

        # Detect file format
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(self.data_path)
        elif ext in ('.xls', '.xlsx'):
            df = pd.read_excel(self.data_path)
        elif ext == '.tsv':
            df = pd.read_csv(self.data_path, sep='\t')
        else:
            df = pd.read_csv(self.data_path)

        logger.info("Raw dataset: %d rows, columns: %s",
                     len(df), list(df.columns))

        # Find SMILES and target columns (flexible matching)
        smiles_col = self._find_column(df, self.smiles_column,
                                        ['PSMILES', 'psmiles', 'smiles', 'SMILES',
                                         'polymer_smiles', 'Polymer_SMILES'])
        target_col = self._find_column(df, self.target_column,
                                        [self.target_column, self.target_column.lower(),
                                         f'{self.target_column}_K', f'{self.target_column}(K)'])

        if smiles_col is None:
            raise ValueError(f"SMILES column not found. Available: {list(df.columns)}")
        if target_col is None:
            raise ValueError(f"Target column '{self.target_column}' not found. "
                             f"Available: {list(df.columns)}")

        # Rename to standard columns
        df = df.rename(columns={smiles_col: 'smiles', target_col: 'target'})
        df = df[['smiles', 'target']].copy()

        # 1. Drop NaN
        n_before = len(df)
        df = df.dropna(subset=['smiles', 'target'])
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        logger.info("After NaN removal: %d → %d", n_before, len(df))

        # 2. Validate PSMILES
        n_before = len(df)
        df['valid'] = df['smiles'].apply(self._validate_psmiles)
        df = df[df['valid']].drop(columns=['valid'])
        logger.info("After PSMILES validation: %d → %d", n_before, len(df))

        # 3. Remove 3-sigma outliers
        n_before = len(df)
        mean = df['target'].mean()
        std = df['target'].std()
        df = df[(df['target'] >= mean - 3 * std) & (df['target'] <= mean + 3 * std)]
        logger.info("After outlier removal (3σ): %d → %d (mean=%.1f, std=%.1f)",
                     n_before, len(df), mean, std)

        # 4. Deduplicate
        n_before = len(df)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        logger.info("After deduplication: %d → %d", n_before, len(df))

        # Optional limit
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info("Sampled %d rows", max_samples)

        df = df.reset_index(drop=True)
        logger.info("Final dataset: %d polymers, target range [%.1f, %.1f]",
                     len(df), df['target'].min(), df['target'].max())

        return df

    def _find_column(self, df: pd.DataFrame, primary: str,
                     alternatives: List[str]) -> Optional[str]:
        """Find column by name, trying alternatives."""
        for name in [primary] + alternatives:
            if name in df.columns:
                return name
        # Case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        for name in [primary] + alternatives:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    @staticmethod
    def _validate_psmiles(smiles: str) -> bool:
        """Check if PSMILES can be parsed by RDKit."""
        try:
            from rdkit import Chem
            if not isinstance(smiles, str) or len(smiles.strip()) == 0:
                return False
            # Replace [*] with [H] for RDKit parsing
            clean = smiles.replace('[*]', '[H]')
            mol = Chem.MolFromSmiles(clean)
            return mol is not None
        except Exception:
            return False


class PolymerScaffoldSplitter:
    """
    Scaffold-based train/val/test split for polymers.

    Uses Murcko scaffold of the repeat unit (after [*] → [H] conversion)
    to ensure structurally similar polymers stay in the same fold.
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

        Returns:
            (train_df, val_df, test_df)
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

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

        # Sort scaffolds by (size DESC, min-index ASC) for fully deterministic splits.
        # Secondary sort by minimum row index breaks ties when two scaffolds
        # have the same size, ensuring reproducibility across Python versions.
        scaffold_sets = sorted(
            scaffolds.values(),
            key=lambda g: (-len(g), min(g)),
        )

        # Shuffle same-size scaffold groups using the seed for reproducibility
        import random
        rng = random.Random(self.seed)
        rng.shuffle(scaffold_sets)
        # Re-sort after shuffle: guarantees deterministic result for a given seed
        scaffold_sets = sorted(
            scaffold_sets,
            key=lambda g: (-len(g), min(g)),
        )

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

        train_df = df.loc[df.index.isin(train_idx)].reset_index(drop=True)
        val_df = df.loc[df.index.isin(val_idx)].reset_index(drop=True)
        test_df = df.loc[df.index.isin(test_idx)].reset_index(drop=True)

        logger.info("Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
                     len(train_df), 100 * len(train_df) / n_total,
                     len(val_df), 100 * len(val_df) / n_total,
                     len(test_df), 100 * len(test_df) / n_total)

        return train_df, val_df, test_df
