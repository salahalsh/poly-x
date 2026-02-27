"""
Tg Model Trainer -- Train RF + GB ensemble on polymer Tg dataset.

Standalone training pipeline for POLY-X Tier 2 (ECFP4 ML ensemble).

Produces:
    <output_dir>/
    ├── checkpoints/tg_rf.joblib, tg_gb.joblib
    ├── metadata.json
    ├── metrics.json
    ├── training_fingerprints.npy
    └── training_smiles.txt

Usage:
    python -m poly_x.train_tg --data-path tg_data.csv --output-dir results/

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats
import joblib

from .fingerprints import PolymerFingerprinter
from .scaffold_split import PolymerScaffoldSplitter

logger = logging.getLogger(__name__)


def _bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    return float(point), float(np.percentile(scores, 100 * alpha)), \
        float(np.percentile(scores, 100 * (1 - alpha)))


def _optimize_ensemble_weight(rf_pred, gb_pred, y_true):
    """Find optimal RF weight w in [0, 1] for weighted ensemble."""
    best_w, best_r2 = 0.5, -np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        ens = w * rf_pred + (1 - w) * gb_pred
        r2 = r2_score(y_true, ens)
        if r2 > best_r2:
            best_w, best_r2 = w, r2
    return round(float(best_w), 2)


def load_and_clean(data_path: str, target_column: str = 'Tg',
                   smiles_column: str = 'PSMILES',
                   max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load and clean polymer Tg dataset.

    Steps: drop NaN, validate PSMILES, remove 3-sigma outliers, deduplicate.
    """
    from rdkit import Chem

    logger.info("Loading dataset from %s ...", data_path)
    ext = os.path.splitext(data_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(data_path)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Find columns (flexible matching)
    smiles_col = None
    for name in [smiles_column, 'PSMILES', 'psmiles', 'smiles', 'SMILES']:
        if name in df.columns:
            smiles_col = name
            break
    if smiles_col is None:
        for c in df.columns:
            if 'smiles' in c.lower():
                smiles_col = c
                break

    target_col = None
    for name in [target_column, target_column.lower(), f'{target_column}_K']:
        if name in df.columns:
            target_col = name
            break

    if smiles_col is None:
        raise ValueError(f"SMILES column not found. Available: {list(df.columns)}")
    if target_col is None:
        raise ValueError(f"Target column '{target_column}' not found. "
                         f"Available: {list(df.columns)}")

    df = df.rename(columns={smiles_col: 'smiles', target_col: 'target'})
    df = df[['smiles', 'target']].copy()

    # Drop NaN
    n_before = len(df)
    df = df.dropna(subset=['smiles', 'target'])
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])
    logger.info("After NaN removal: %d -> %d", n_before, len(df))

    # Validate PSMILES
    n_before = len(df)
    def _valid(s):
        if not isinstance(s, str) or len(s.strip()) == 0:
            return False
        clean = s.replace('[*]', '[H]')
        return Chem.MolFromSmiles(clean) is not None

    df = df[df['smiles'].apply(_valid)]
    logger.info("After PSMILES validation: %d -> %d", n_before, len(df))

    # Remove 3-sigma outliers
    n_before = len(df)
    mean, std = df['target'].mean(), df['target'].std()
    lower, upper = mean - 3 * std, mean + 3 * std
    df = df[(df['target'] >= lower) & (df['target'] <= upper)]
    logger.info("After outlier removal: %d -> %d", n_before, len(df))

    # Deduplicate
    n_before = len(df)
    df = df.drop_duplicates(subset=['smiles'], keep='first')
    logger.info("After deduplication: %d -> %d", n_before, len(df))

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    df = df.reset_index(drop=True)
    logger.info("Final dataset: %d polymers, target range [%.1f, %.1f]",
                 len(df), df['target'].min(), df['target'].max())
    return df


class TgTrainer:
    """
    Train RF + GB ensemble for Tg prediction.

    Pipeline:
        1. Load CSV -> clean & filter
        2. ECFP4 2048-bit featurization
        3. Scaffold split (80/10/10)
        4. Train RF + GB, optimize ensemble weight on val set
        5. Evaluate on test set with bootstrap CIs
        6. Save models, metadata, training fingerprints
    """

    def __init__(self, data_path: str, output_dir: str = 'results',
                 max_samples: Optional[int] = None,
                 rf_n_estimators: int = 500,
                 gb_n_estimators: int = 200):
        self.data_path = data_path
        self.max_samples = max_samples
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    def train(self) -> Dict:
        """Run the full training pipeline."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Tg Model Training -- Starting")
        logger.info("=" * 60)

        # 1. Load data
        df = load_and_clean(self.data_path, max_samples=self.max_samples)

        # 2. Featurize
        fingerprinter = PolymerFingerprinter(radius=2, n_bits=2048)
        X = fingerprinter.featurize_batch(df['smiles'].tolist())

        # Remove failed featurizations
        valid_mask = X.sum(axis=1) > 0
        X = X[valid_mask]
        y = df.loc[valid_mask, 'target'].values
        smiles = df.loc[valid_mask, 'smiles'].values

        clean_df = pd.DataFrame({'smiles': smiles, 'target': y})
        clean_df = clean_df.reset_index(drop=True)
        logger.info("After filtering: %d valid samples", len(y))

        # 3. Scaffold split
        splitter = PolymerScaffoldSplitter(
            train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_df, val_df, test_df = splitter.split(clean_df)

        logger.info("Splits -- Train: %d, Val: %d, Test: %d",
                     len(train_df), len(val_df), len(test_df))

        # Featurize each split
        X_train = fingerprinter.featurize_batch(train_df['smiles'].tolist())
        y_train = train_df['target'].values
        X_val = fingerprinter.featurize_batch(val_df['smiles'].tolist())
        y_val = val_df['target'].values
        X_test = fingerprinter.featurize_batch(test_df['smiles'].tolist())
        y_test = test_df['target'].values

        # 4. Train RF
        rf_params = {
            'n_estimators': self.rf_n_estimators,
            'max_depth': None,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42,
        }
        logger.info("Training Random Forest (n_estimators=%d)...",
                     rf_params['n_estimators'])
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        rf_val_pred = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)

        # 5. Train GB
        gb_params = {
            'n_estimators': self.gb_n_estimators,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_leaf': 5,
            'random_state': 42,
        }
        logger.info("Training Gradient Boosting (n_estimators=%d)...",
                     gb_params['n_estimators'])
        gb = GradientBoostingRegressor(**gb_params)
        gb.fit(X_train, y_train)
        gb_val_pred = gb.predict(X_val)
        gb_test_pred = gb.predict(X_test)

        # 6. Optimized weighted ensemble
        rf_weight = _optimize_ensemble_weight(rf_val_pred, gb_val_pred, y_val)
        ensemble_val = rf_weight * rf_val_pred + (1 - rf_weight) * gb_val_pred
        ensemble_test = rf_weight * rf_test_pred + (1 - rf_weight) * gb_test_pred
        logger.info("Ensemble weight: RF=%.2f, GB=%.2f", rf_weight, 1 - rf_weight)

        # 7. Compute metrics with bootstrap CIs
        metrics = self._compute_metrics(
            y_val, ensemble_val, rf_val_pred, gb_val_pred,
            y_test, ensemble_test, rf_test_pred, gb_test_pred,
            rf_weight)

        logger.info("=" * 60)
        logger.info("ENSEMBLE -- Val: R2=%.4f, MAE=%.2f K, RMSE=%.2f K",
                     metrics['val']['r2'], metrics['val']['mae'],
                     metrics['val']['rmse'])
        logger.info("ENSEMBLE -- Test: R2=%.4f [%.4f, %.4f], MAE=%.2f K, RMSE=%.2f K",
                     metrics['test']['r2'],
                     metrics['test'].get('r2_ci_lower', 0),
                     metrics['test'].get('r2_ci_upper', 0),
                     metrics['test']['mae'],
                     metrics['test']['rmse'])
        logger.info("=" * 60)

        # 8. Save everything
        self._save(rf, gb, metrics, X_train,
                   train_df['smiles'].tolist(),
                   len(clean_df), start_time, rf_params, gb_params,
                   rf_weight)

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds", elapsed)
        return metrics

    def _compute_metrics(self, y_val, ens_val, rf_val, gb_val,
                         y_test, ens_test, rf_test, gb_test,
                         rf_weight):
        """Compute full metrics with bootstrap CIs."""
        def _r2(y, p): return r2_score(y, p)
        def _mae(y, p): return mean_absolute_error(y, p)
        def _rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

        def _eval(y_true, ens_pred, rf_pred, gb_pred, compute_ci=False):
            m = {
                'r2': round(float(_r2(y_true, ens_pred)), 4),
                'mae': round(float(_mae(y_true, ens_pred)), 2),
                'rmse': round(float(_rmse(y_true, ens_pred)), 2),
            }
            rho, pval = scipy_stats.spearmanr(y_true, ens_pred)
            m['spearman_rho'] = round(float(rho), 4)
            m['spearman_pvalue'] = float(pval)

            if compute_ci:
                _, r2_lo, r2_hi = _bootstrap_ci(y_true, ens_pred, _r2)
                _, mae_lo, mae_hi = _bootstrap_ci(y_true, ens_pred, _mae)
                m['r2_ci_lower'] = round(r2_lo, 4)
                m['r2_ci_upper'] = round(r2_hi, 4)
                m['mae_ci_lower'] = round(mae_lo, 2)
                m['mae_ci_upper'] = round(mae_hi, 2)

            return m

        return {
            'val': _eval(y_val, ens_val, rf_val, gb_val),
            'test': _eval(y_test, ens_test, rf_test, gb_test, compute_ci=True),
            'rf_test_r2': round(float(_r2(y_test, rf_test)), 4),
            'gb_test_r2': round(float(_r2(y_test, gb_test)), 4),
            'ensemble_test_r2': round(float(_r2(y_test, ens_test)), 4),
            'ensemble_rf_weight': rf_weight,
        }

    def _save(self, rf, gb, metrics, train_fps, train_smiles,
              n_total, start_time, rf_params, gb_params, rf_weight):
        """Save models, metadata, and training data."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        joblib.dump(rf, os.path.join(self.checkpoint_dir, 'tg_rf.joblib'),
                    compress=3)
        joblib.dump(gb, os.path.join(self.checkpoint_dir, 'tg_gb.joblib'),
                    compress=3)
        logger.info("Models saved to %s", self.checkpoint_dir)

        np.save(os.path.join(self.output_dir, 'training_fingerprints.npy'),
                train_fps)
        with open(os.path.join(self.output_dir, 'training_smiles.txt'), 'w',
                  encoding='utf-8') as f:
            for smi in train_smiles:
                f.write(smi + '\n')

        import sklearn
        import rdkit
        metadata = {
            'property': 'tg',
            'property_name': 'Glass Transition Temperature',
            'property_unit': 'K',
            'model_type': 'Weighted RF+GB Ensemble',
            'fingerprint_type': 'ECFP4 (Morgan)',
            'fingerprint_bits': 2048,
            'fingerprint_radius': 2,
            'n_total': n_total,
            'n_training': len(train_smiles),
            'scaffold_split': '80/10/10',
            'scaffold_split_seed': 42,
            'rf_params': {k: v for k, v in rf_params.items()
                         if k != 'n_jobs'},
            'gb_params': gb_params,
            'ensemble_rf_weight': rf_weight,
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse'],
            'dependency_versions': {
                'python': __import__('sys').version.split()[0],
                'sklearn': sklearn.__version__,
                'rdkit': rdkit.__version__,
                'numpy': np.__version__,
            },
            'trained_date': datetime.now().isoformat(),
            'training_time_seconds': round(time.time() - start_time, 1),
        }

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info("Metadata saved to %s", self.output_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train RF+GB ensemble for polymer Tg prediction')
    parser.add_argument('--data-path', required=True,
                        help='Path to CSV with PSMILES and Tg columns')
    parser.add_argument('--output-dir', default='results',
                        help='Directory to save models and metrics')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit dataset size (for testing)')
    parser.add_argument('--rf-estimators', type=int, default=500,
                        help='Number of RF trees')
    parser.add_argument('--gb-estimators', type=int, default=200,
                        help='Number of GB trees')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s')

    trainer = TgTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        rf_n_estimators=args.rf_estimators,
        gb_n_estimators=args.gb_estimators,
    )
    metrics = trainer.train()
    print(f"\nTest R2: {metrics['test']['r2']:.4f}")
    print(f"Test MAE: {metrics['test']['mae']:.2f} K")
    print(f"Test RMSE: {metrics['test']['rmse']:.2f} K")


if __name__ == '__main__':
    main()
