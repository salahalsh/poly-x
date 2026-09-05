"""
Tg Model Trainer - Train RF + GB ensemble on the PolyMetriX Tg dataset.

Produces:
    poly_x/trained_models/tg/
    ├── checkpoints/tg_rf.joblib, tg_gb.joblib
    ├── metadata.json
    ├── metrics.json
    ├── config.json
    ├── training_fingerprints.npy   (for AD assessment)
    └── training_smiles.txt         (for AD assessment)

Usage:
    trainer = TgTrainer(data_path='data/pi1m_tg.csv')
    metrics = trainer.train()
    print(metrics)  # {'r2': 0.75, 'mae': 28.3, 'rmse': 41.7}
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

logger = logging.getLogger(__name__)

#: Resolution of the ensemble-weight search. 101 points over [0, 1] is finer
#: than the validation noise on a few-hundred-row scaffold split, so a tighter
#: grid buys precision the data does not support.
_WEIGHT_GRID = np.linspace(0.0, 1.0, 101)


def _optimize_ensemble_weight(rf_pred, gb_pred, y_true):
    """Pick w for ``w * rf_pred + (1 - w) * gb_pred`` on a HELD-OUT split.

    Chosen by exhaustive search over `_WEIGHT_GRID` rather than a closed-form
    least-squares blend. Least squares would be one line, but it is unbounded:
    when RF and GB predictions are strongly collinear - which they are, being
    two tree ensembles on the same fingerprints - the optimum runs off to large
    positive and negative weights that extrapolate badly on the test set. A
    bounded grid cannot produce a weight outside [0, 1], so the ensemble is
    always a genuine interpolation between the two models.

    Minimises squared error, which for a fixed ``y_true`` has the same argmax
    as R2, so the choice of metric here does not change the selected weight.

    MUST be called with validation predictions, never test: this fits one adjustable
    parameter, and fitting it on test leaks the test set into the reported
    score. `TgTrainer.train` deliberately does NOT use this - it ships a fixed
    50/50 average, and the published Tg checkpoint metrics reflect that.

    Returns a plain float in [0, 1]: RF's share of the blend.
    """
    rf_pred = np.asarray(rf_pred, dtype=float)
    gb_pred = np.asarray(gb_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if rf_pred.shape != gb_pred.shape or rf_pred.shape != y_true.shape:
        raise ValueError(
            f'shape mismatch: rf={rf_pred.shape}, gb={gb_pred.shape}, '
            f'y={y_true.shape}')
    if y_true.size == 0:
        raise ValueError('cannot optimise an ensemble weight on an empty split')

    # (n_weights, n_samples) blend, then SSE per candidate weight.
    blends = (_WEIGHT_GRID[:, None] * rf_pred[None, :]
              + (1.0 - _WEIGHT_GRID)[:, None] * gb_pred[None, :])
    sse = ((blends - y_true[None, :]) ** 2).sum(axis=1)
    return float(_WEIGHT_GRID[int(np.argmin(sse))])


class TgTrainer:
    """
    Train RF + GB ensemble for Tg prediction.

    Pipeline:
        1. Load the property CSV (PolyMetriX for the shipped Tg model) → clean & filter
        2. ECFP4 2048-bit featurization
        3. Scaffold split (80/10/10)
        4. Train RF(500) + GB(200) independently
        5. Evaluate on test set (R2, MAE, RMSE)
        6. Save models, metadata, training fingerprints
    """

    def __init__(self, data_path: str, output_dir: Optional[str] = None,
                 max_samples: Optional[int] = None,
                 rf_n_estimators: int = 500,
                 gb_n_estimators: int = 200):
        self.data_path = data_path
        self.max_samples = max_samples
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators

        if output_dir is None:
            from django.conf import settings
            output_dir = str(
                Path(settings.BASE_DIR) / 'poly_x' / 'trained_models' / 'tg')
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    def train(self) -> Dict:
        """
        Run the full training pipeline.

        Returns:
            Dict with metrics (r2, mae, rmse for val and test sets)
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Tg Model Training - Starting")
        logger.info("=" * 60)

        # 1. Load data
        from .dataset_loaders import PI1MLoader, PolymerScaffoldSplitter
        loader = PI1MLoader(self.data_path, target_column='Tg')
        df = loader.load(max_samples=self.max_samples)

        # 2. Featurize
        from .featurizers import TrainingFeaturizer
        featurizer = TrainingFeaturizer(n_bits=2048, radius=2)
        X = featurizer.featurize(df['smiles'].tolist())

        # Remove failed featurizations
        valid_mask = featurizer.get_valid_mask(X)
        X = X[valid_mask]
        y = df.loc[valid_mask, 'target'].values
        smiles = df.loc[valid_mask, 'smiles'].values
        logger.info("After filtering: %d valid samples", len(y))

        # Create a clean DataFrame for scaffold split
        import pandas as pd
        clean_df = pd.DataFrame({'smiles': smiles, 'target': y})
        clean_df = clean_df.reset_index(drop=True)

        # 3. Scaffold split
        splitter = PolymerScaffoldSplitter(
            train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_df, val_df, test_df = splitter.split(clean_df)

        logger.info("Splits - Train: %d, Val: %d, Test: %d",
                     len(train_df), len(val_df), len(test_df))

        # Featurize each split separately (avoids index mapping bugs)
        logger.info("Featurizing splits...")
        X_train = featurizer.featurize(train_df['smiles'].tolist())
        y_train = train_df['target'].values
        X_val = featurizer.featurize(val_df['smiles'].tolist())
        y_val = val_df['target'].values
        X_test = featurizer.featurize(test_df['smiles'].tolist())
        y_test = test_df['target'].values

        # 4. Train RF
        logger.info("Training Random Forest (n_estimators=%d)...",
                     self.rf_n_estimators)
        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_val_pred = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)
        logger.info("RF - Val R²=%.4f, Test R²=%.4f",
                     r2_score(y_val, rf_val_pred),
                     r2_score(y_test, rf_test_pred))

        # 5. Train GB
        logger.info("Training Gradient Boosting (n_estimators=%d)...",
                     self.gb_n_estimators)
        gb = GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        gb.fit(X_train, y_train)
        gb_val_pred = gb.predict(X_val)
        gb_test_pred = gb.predict(X_test)
        logger.info("GB - Val R²=%.4f, Test R²=%.4f",
                     r2_score(y_val, gb_val_pred),
                     r2_score(y_test, gb_test_pred))

        # 6. Ensemble (average)
        ensemble_val = (rf_val_pred + gb_val_pred) / 2.0
        ensemble_test = (rf_test_pred + gb_test_pred) / 2.0

        metrics = {
            'val': {
                'r2': float(r2_score(y_val, ensemble_val)),
                'mae': float(mean_absolute_error(y_val, ensemble_val)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, ensemble_val))),
            },
            'test': {
                'r2': float(r2_score(y_test, ensemble_test)),
                'mae': float(mean_absolute_error(y_test, ensemble_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, ensemble_test))),
            },
            # Individual model metrics (for reproducibility verification)
            'rf_val': {
                'r2': float(r2_score(y_val, rf_val_pred)),
                'mae': float(mean_absolute_error(y_val, rf_val_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, rf_val_pred))),
            },
            'rf_test': {
                'r2': float(r2_score(y_test, rf_test_pred)),
                'mae': float(mean_absolute_error(y_test, rf_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, rf_test_pred))),
            },
            'gb_val': {
                'r2': float(r2_score(y_val, gb_val_pred)),
                'mae': float(mean_absolute_error(y_val, gb_val_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, gb_val_pred))),
            },
            'gb_test': {
                'r2': float(r2_score(y_test, gb_test_pred)),
                'mae': float(mean_absolute_error(y_test, gb_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, gb_test_pred))),
            },
            'rf_test_r2': float(r2_score(y_test, rf_test_pred)),
            'gb_test_r2': float(r2_score(y_test, gb_test_pred)),
            'ensemble_test_r2': float(r2_score(y_test, ensemble_test)),
        }

        logger.info("=" * 60)
        logger.info("ENSEMBLE - Val: R²=%.4f, MAE=%.2f K, RMSE=%.2f K",
                     metrics['val']['r2'], metrics['val']['mae'],
                     metrics['val']['rmse'])
        logger.info("ENSEMBLE - Test: R²=%.4f, MAE=%.2f K, RMSE=%.2f K",
                     metrics['test']['r2'], metrics['test']['mae'],
                     metrics['test']['rmse'])
        logger.info("=" * 60)

        # 7. Save everything
        self._save(rf, gb, metrics, X_train,
                   train_df['smiles'].tolist(),
                   len(clean_df), start_time)

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds", elapsed)

        return metrics

    def _save(self, rf, gb, metrics: Dict,
              train_fps: np.ndarray, train_smiles: list,
              n_total: int, start_time: float):
        """Save models, metadata, and training data for AD."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save models
        joblib.dump(rf, os.path.join(self.checkpoint_dir, 'tg_rf.joblib'))
        joblib.dump(gb, os.path.join(self.checkpoint_dir, 'tg_gb.joblib'))
        logger.info("Models saved to %s", self.checkpoint_dir)

        # Save training fingerprints (for AD assessment)
        np.save(os.path.join(self.output_dir, 'training_fingerprints.npy'),
                train_fps)
        with open(os.path.join(self.output_dir, 'training_smiles.txt'), 'w',
                  encoding='utf-8') as f:
            for smi in train_smiles:
                f.write(smi + '\n')
        logger.info("Saved %d training fingerprints for AD", len(train_fps))

        # Capture dependency versions for reproducibility
        import sklearn
        import rdkit
        dep_versions = {
            'python': __import__('sys').version.split()[0],
            'sklearn': sklearn.__version__,
            'rdkit': rdkit.__version__,
            'numpy': np.__version__,
            'joblib': joblib.__version__,
        }

        # Save metadata (complete for reproducibility)
        metadata = {
            'property': 'tg',
            'property_name': 'Glass Transition Temperature',
            'property_unit': 'K',
            'model_type': 'RF+GB Ensemble',
            'version': '1.0',
            # Dataset
            'n_total_before_cleaning': n_total,
            'n_training': len(train_smiles),
            'n_training_fingerprints': len(train_fps),
            # Fingerprint
            'fingerprint_type': 'ECFP4 (Morgan)',
            'fingerprint_bits': 2048,
            'fingerprint_radius': 2,
            # Model hyperparameters
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': None,
            'rf_min_samples_leaf': 2,
            'gb_n_estimators': self.gb_n_estimators,
            'gb_max_depth': 5,
            'gb_learning_rate': 0.1,
            'gb_subsample': 0.8,
            'gb_min_samples_leaf': 5,
            'random_state': 42,
            # Split
            'scaffold_split': '80/10/10',
            'scaffold_split_seed': 42,
            # Metrics
            'r2_score': metrics['test']['r2'],
            'mae': metrics['test']['mae'],
            'rmse': metrics['test']['rmse'],
            'rf_test_r2': metrics['rf_test_r2'],
            'gb_test_r2': metrics['gb_test_r2'],
            # Environment
            'dependency_versions': dep_versions,
            'trained_date': datetime.now().isoformat(),
            'training_time_seconds': round(time.time() - start_time, 1),
        }

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        config = {
            'property': 'tg',
            'fingerprint': {'type': 'Morgan', 'radius': 2, 'n_bits': 2048},
            'models': {
                'rf': {'type': 'RandomForestRegressor',
                       'n_estimators': self.rf_n_estimators},
                'gb': {'type': 'GradientBoostingRegressor',
                       'n_estimators': self.gb_n_estimators},
            },
            'split': {'method': 'scaffold', 'train': 0.8,
                      'val': 0.1, 'test': 0.1},
        }

        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("Metadata saved to %s", self.output_dir)
