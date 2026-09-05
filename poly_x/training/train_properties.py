"""
Generic Property Trainer - Train RF + GB ensemble for any polymer property.

Reuses the same architecture as TgTrainer but parameterized by property name.
Supports: Tm, Td, density, solubility_param (and any other numeric property).

Usage:
    trainer = PropertyTrainer(
        property_name='tm',
        property_display='Melting Temperature',
        property_unit='K',
        data_path='data/pi1m_tm.csv',
        target_column='Tm',
    )
    metrics = trainer.train()
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


class PropertyTrainer:
    """
    Generic trainer for polymer property regression.

    Same RF(500) + GB(200) architecture as TgTrainer,
    parameterized by property name and target column.
    """

    def __init__(self, property_name: str, data_path: str,
                 target_column: str,
                 property_display: Optional[str] = None,
                 property_unit: str = '',
                 output_dir: Optional[str] = None,
                 max_samples: Optional[int] = None,
                 rf_n_estimators: int = 500,
                 gb_n_estimators: int = 200):
        self.property_name = property_name
        self.property_display = property_display or property_name.upper()
        self.property_unit = property_unit
        self.data_path = data_path
        self.target_column = target_column
        self.max_samples = max_samples
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators

        if output_dir is None:
            from django.conf import settings
            output_dir = str(
                Path(settings.BASE_DIR) / 'poly_x' / 'trained_models' / property_name)
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    def train(self) -> Dict:
        """Run the full training pipeline. Returns metrics dict."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("%s Model Training - Starting", self.property_display)
        logger.info("=" * 60)

        # 1. Load & clean
        from .dataset_loaders import PI1MLoader, PolymerScaffoldSplitter
        loader = PI1MLoader(
            self.data_path,
            target_column=self.target_column,
        )
        df = loader.load(max_samples=self.max_samples)

        # 2. Featurize
        from .featurizers import TrainingFeaturizer
        featurizer = TrainingFeaturizer(n_bits=2048, radius=2)
        X = featurizer.featurize(df['smiles'].tolist())

        # Filter invalid
        valid_mask = featurizer.get_valid_mask(X)
        X = X[valid_mask]
        y = df.loc[valid_mask, 'target'].values
        smiles = df.loc[valid_mask, 'smiles'].values

        import pandas as pd
        clean_df = pd.DataFrame({'smiles': smiles, 'target': y})
        clean_df = clean_df.reset_index(drop=True)

        # 3. Scaffold split
        splitter = PolymerScaffoldSplitter()
        train_df, val_df, test_df = splitter.split(clean_df)

        # Featurize each split independently to avoid index mapping bugs
        # (scaffold split resets indices, so X_all[split_df.index] would
        #  grab wrong rows - fingerprints and labels would be misaligned)
        logger.info("Featurizing splits...")
        X_train = featurizer.featurize(train_df['smiles'].tolist())
        y_train = train_df['target'].values
        X_val = featurizer.featurize(val_df['smiles'].tolist())
        y_val = val_df['target'].values
        X_test = featurizer.featurize(test_df['smiles'].tolist())
        y_test = test_df['target'].values

        logger.info("Train=%d, Val=%d, Test=%d",
                     len(y_train), len(y_val), len(y_test))

        # 4. Train RF
        logger.info("Training RF (n=%d)...", self.rf_n_estimators)
        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )
        rf.fit(X_train, y_train)

        # 5. Train GB
        logger.info("Training GB (n=%d)...", self.gb_n_estimators)
        gb = GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators,
            max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5,
            random_state=42,
        )
        gb.fit(X_train, y_train)

        # 6. Ensemble
        ensemble_test = (rf.predict(X_test) + gb.predict(X_test)) / 2.0
        ensemble_val = (rf.predict(X_val) + gb.predict(X_val)) / 2.0

        rf_val_pred = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)
        gb_val_pred = gb.predict(X_val)
        gb_test_pred = gb.predict(X_test)

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
            'rf_test': {
                'r2': float(r2_score(y_test, rf_test_pred)),
                'mae': float(mean_absolute_error(y_test, rf_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, rf_test_pred))),
            },
            'gb_test': {
                'r2': float(r2_score(y_test, gb_test_pred)),
                'mae': float(mean_absolute_error(y_test, gb_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, gb_test_pred))),
            },
        }

        logger.info("ENSEMBLE Test: R²=%.4f, MAE=%.2f %s, RMSE=%.2f %s",
                     metrics['test']['r2'],
                     metrics['test']['mae'], self.property_unit,
                     metrics['test']['rmse'], self.property_unit)

        # 7. Save
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        joblib.dump(rf, os.path.join(
            self.checkpoint_dir, f'{self.property_name}_rf.joblib'))
        joblib.dump(gb, os.path.join(
            self.checkpoint_dir, f'{self.property_name}_gb.joblib'))

        # Save training data for AD
        np.save(os.path.join(self.output_dir, 'training_fingerprints.npy'),
                X_train)
        with open(os.path.join(self.output_dir, 'training_smiles.txt'), 'w',
                  encoding='utf-8') as f:
            for smi in train_df['smiles'].tolist():
                f.write(smi + '\n')

        # Dependency versions for reproducibility
        import sklearn
        import rdkit
        dep_versions = {
            'python': __import__('sys').version.split()[0],
            'sklearn': sklearn.__version__,
            'rdkit': rdkit.__version__,
            'numpy': np.__version__,
        }

        # Metadata (complete for reproducibility)
        metadata = {
            'property': self.property_name,
            'property_name': self.property_display,
            'property_unit': self.property_unit,
            'model_type': 'RF+GB Ensemble',
            'version': '1.0',
            'n_total_before_cleaning': len(clean_df) if 'clean_df' in dir() else len(y_train),
            'n_training': len(y_train),
            'n_validation': len(y_val),
            'n_test': len(y_test),
            'fingerprint_type': 'ECFP4 (Morgan)',
            'fingerprint_bits': 2048,
            'fingerprint_radius': 2,
            'rf_n_estimators': self.rf_n_estimators,
            'gb_n_estimators': self.gb_n_estimators,
            'random_state': 42,
            'scaffold_split': '80/10/10',
            'r2_score': metrics['test']['r2'],
            'mae': metrics['test']['mae'],
            'rmse': metrics['test']['rmse'],
            'rf_test_r2': metrics['rf_test']['r2'],
            'gb_test_r2': metrics['gb_test']['r2'],
            'dependency_versions': dep_versions,
            'trained_date': datetime.now().isoformat(),
        }
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds", elapsed)
        return metrics
