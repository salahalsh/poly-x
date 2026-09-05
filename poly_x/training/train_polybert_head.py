"""
Train lightweight prediction heads on polyBERT embeddings.

Pipeline:
1. Load FULL PolyMetriX dataset + clean
2. Scaffold split (80/10/10, seed=42) - SAME split as Tier 2
3. Extract 600-D polyBERT CLS embeddings (batched, with caching)
4. Train RF/GB ensemble on train embeddings
5. Evaluate on val AND test embeddings (fair cross-tier comparison)
6. Save head as joblib

Usage via management command:
    python manage.py train_polymer_models --property tg_polybert
"""

import logging
import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PolyBERTHeadTrainer:
    """Train sklearn prediction heads on top of polyBERT embeddings."""

    def __init__(self,
                 training_data_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 batch_size: int = 32,
                 max_samples: Optional[int] = None):
        """
        Args:
            training_data_dir: Dir with training_smiles.txt and
                               tg_training_data.csv (from Tier 2 training)
            output_dir: Where to save embeddings and heads
            batch_size: Batch size for embedding extraction
            max_samples: Limit for quick testing
        """
        base = Path(__file__).resolve().parent.parent

        if training_data_dir is None:
            training_data_dir = str(base / 'trained_models' / 'tg')
        if output_dir is None:
            output_dir = str(base / 'trained_models' / 'polybert')

        self.training_data_dir = training_data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_samples = max_samples

    def train(self) -> dict:
        """Full training pipeline with scaffold split (matching Tier 2).

        Uses the SAME dataset loading, cleaning, and scaffold splitting as
        Tier 2 (PI1MLoader + PolymerScaffoldSplitter with seed=42), ensuring
        identical train/val/test partitions for fair cross-tier comparison.
        """
        import warnings
        warnings.filterwarnings('ignore')
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'heads'), exist_ok=True)

        start_time = time.time()

        logger.info("=" * 60)
        logger.info("polyBERT Head Training (Scaffold Split)")
        logger.info("=" * 60)

        # 1. Load and clean the FULL dataset - same pipeline as Tier 2
        data_path = os.path.join(
            Path(self.training_data_dir).parent.parent, 'data', 'tg_training_data.csv')

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Training data not found: {data_path}\n"
                "Place tg_training_data.csv in poly_x/data/")

        from .dataset_loaders import PI1MLoader, PolymerScaffoldSplitter

        loader = PI1MLoader(data_path, target_column='Tg')
        df = loader.load(max_samples=self.max_samples)
        logger.info("Cleaned dataset: %d polymers", len(df))

        # 2. Scaffold split (80/10/10, seed=42) - IDENTICAL to Tier 2
        splitter = PolymerScaffoldSplitter(
            train_frac=0.8, val_frac=0.1, test_frac=0.1, random_seed=42)
        train_df, val_df, test_df = splitter.split(df)

        logger.info("Scaffold split - Train: %d, Val: %d, Test: %d",
                     len(train_df), len(val_df), len(test_df))

        # 3. Extract embeddings for ALL three partitions
        all_smiles = (train_df['smiles'].tolist() +
                      val_df['smiles'].tolist() +
                      test_df['smiles'].tolist())
        all_embeddings = self._extract_embeddings(all_smiles)

        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)

        X_train = all_embeddings[:n_train]
        y_train = train_df['target'].values
        X_val = all_embeddings[n_train:n_train + n_val]
        y_val = val_df['target'].values
        X_test = all_embeddings[n_train + n_val:]
        y_test = test_df['target'].values

        logger.info("Embedding shapes - Train: %s, Val: %s, Test: %s",
                     X_train.shape, X_val.shape, X_test.shape)

        # 4. Train RF + GB ensemble on train, evaluate on val AND test
        metrics = self._train_heads(X_train, y_train, X_val, y_val,
                                    X_test, y_test)

        # 5. Save metadata (complete for reproducibility)
        import sklearn
        import rdkit
        metadata = {
            'property': 'tg',
            'property_name': 'Glass Transition Temperature',
            'property_unit': 'K',
            'embedding_model': 'kuelumbus/polyBERT',
            'embedding_dim': 600,
            'split_method': 'scaffold',
            'split_fractions': '80/10/10',
            'split_seed': 42,
            'n_training': n_train,
            'n_validation': n_val,
            'n_test': n_test,
            'n_total_polymers': len(df),
            'metrics': metrics,
            'batch_size': self.batch_size,
            'rf_n_estimators': 300,
            'rf_min_samples_leaf': 3,
            'gb_n_estimators': 200,
            'gb_max_depth': 5,
            'gb_learning_rate': 0.1,
            'gb_subsample': 0.8,
            'gb_min_samples_leaf': 5,
            'random_state': 42,
            'dependency_versions': {
                'python': __import__('sys').version.split()[0],
                'sklearn': sklearn.__version__,
                'rdkit': rdkit.__version__,
                'numpy': np.__version__,
            },
            'training_time_seconds': round(time.time() - start_time, 1),
        }
        meta_path = os.path.join(self.output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Metadata saved: %s", meta_path)
        return metadata

    def _extract_embeddings(self, smiles_list: list) -> np.ndarray:
        """Extract polyBERT CLS embeddings, with caching.

        Cache is keyed on the exact SMILES list: if the list changes
        (e.g., due to switching from random to scaffold split), the
        cache is invalidated and embeddings are re-extracted.
        """
        cache_path = os.path.join(self.output_dir, 'all_embeddings_scaffold.npy')
        cache_smiles_path = os.path.join(self.output_dir, 'all_smiles_scaffold_order.txt')

        # Check cache
        if os.path.exists(cache_path) and os.path.exists(cache_smiles_path):
            with open(cache_smiles_path) as f:
                cached_smiles = [l.strip() for l in f if l.strip()]
            if cached_smiles == smiles_list:
                logger.info("Loading cached embeddings: %s", cache_path)
                return np.load(cache_path)
            else:
                logger.info("Cache mismatch (different SMILES), re-extracting")

        # Load polyBERT
        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info("Loading polyBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
        model = AutoModel.from_pretrained('kuelumbus/polyBERT')
        model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = model.cuda()
        logger.info("polyBERT on %s", device.upper())

        # Batch extraction
        n = len(smiles_list)
        embeddings = np.zeros((n, 600), dtype=np.float32)
        failed = 0
        t0 = time.time()

        logger.info("Extracting embeddings for %d polymers (batch_size=%d)...",
                     n, self.batch_size)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = smiles_list[start:end]

            try:
                inputs = tokenizer(
                    batch, return_tensors='pt',
                    padding=True, truncation=True, max_length=512)
                if device == 'cuda':
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings[start:end] = cls
            except Exception as e:
                logger.warning("Batch %d-%d failed: %s", start, end, e)
                failed += (end - start)

            # Progress logging
            done = end
            if done % 500 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                logger.info("  %d/%d (%.1f/s, ETA %.0fs)", done, n, rate, eta)

        elapsed = time.time() - t0
        logger.info("Embedding extraction: %.1fs (%.2f polymers/s), %d failed",
                     elapsed, n / elapsed, failed)

        # Save cache
        np.save(cache_path, embeddings)
        with open(cache_smiles_path, 'w') as f:
            f.write('\n'.join(smiles_list) + '\n')
        logger.info("Embeddings cached: %s (%.1f MB)",
                     cache_path, os.path.getsize(cache_path) / 1e6)

        return embeddings

    def _train_heads(self, X_train, y_train, X_val, y_val,
                     X_test=None, y_test=None) -> dict:
        """Train RF + GB ensemble on polyBERT embeddings.

        Evaluates on both validation and test sets when provided,
        matching Tier 2's reporting format for fair comparison.
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import joblib

        def _eval(y_true, y_pred):
            return {
                'r2': round(float(r2_score(y_true, y_pred)), 4),
                'mae_K': round(float(mean_absolute_error(y_true, y_pred)), 2),
                'rmse_K': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            }

        logger.info("Training RF on polyBERT embeddings...")
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_leaf=3, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        rf_time = time.time() - t0

        logger.info("Training GB on polyBERT embeddings...")
        t1 = time.time()
        gb = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_time = time.time() - t1

        # Validation predictions
        rf_val = rf.predict(X_val)
        gb_val = gb.predict(X_val)
        ens_val = (rf_val + gb_val) / 2

        metrics = {
            'rf_val': {**_eval(y_val, rf_val), 'train_time_s': round(rf_time, 1)},
            'gb_val': {**_eval(y_val, gb_val), 'train_time_s': round(gb_time, 1)},
            'val': _eval(y_val, ens_val),
        }

        # Test predictions (scaffold-disjoint)
        if X_test is not None and y_test is not None:
            rf_test = rf.predict(X_test)
            gb_test = gb.predict(X_test)
            ens_test = (rf_test + gb_test) / 2

            metrics['rf_test'] = _eval(y_test, rf_test)
            metrics['gb_test'] = _eval(y_test, gb_test)
            metrics['test'] = _eval(y_test, ens_test)

        logger.info("=" * 60)
        logger.info("polyBERT Head Results (Scaffold Split):")
        logger.info("-" * 60)
        for split_name in ['val', 'test']:
            m = metrics.get(split_name)
            if m:
                logger.info("  ENSEMBLE %s - R²=%.4f, MAE=%.2f K, RMSE=%.2f K",
                             split_name.upper(), m['r2'], m['mae_K'], m['rmse_K'])
        logger.info("-" * 60)
        for name in ['rf_val', 'gb_val', 'rf_test', 'gb_test']:
            m = metrics.get(name)
            if m:
                logger.info("  %s - R²=%.4f, MAE=%.2f K", name, m['r2'], m['mae_K'])
        logger.info("=" * 60)

        # Save heads
        heads_dir = os.path.join(self.output_dir, 'heads')
        rf_path = os.path.join(heads_dir, 'tg_head.joblib')
        gb_path = os.path.join(heads_dir, 'tg_gb_head.joblib')

        # Save ensemble as the primary head (combines both)
        # The predictor loads tg_head.joblib - we'll save an EnsembleHead wrapper
        joblib.dump(rf, os.path.join(heads_dir, 'tg_rf_head.joblib'))
        joblib.dump(gb, gb_path)

        # Create the primary combined head
        ensemble_head = _EnsembleHead(rf, gb)
        joblib.dump(ensemble_head, rf_path)

        logger.info("Heads saved:")
        logger.info("  RF: %s (%.1f MB)", os.path.basename(rf_path),
                     os.path.getsize(os.path.join(heads_dir, 'tg_rf_head.joblib')) / 1e6)
        logger.info("  GB: %s (%.1f KB)", os.path.basename(gb_path),
                     os.path.getsize(gb_path) / 1e3)
        logger.info("  Ensemble: %s", os.path.basename(rf_path))

        return metrics


class _EnsembleHead:
    """Lightweight wrapper for RF+GB ensemble prediction."""

    def __init__(self, rf, gb):
        self.rf = rf
        self.gb = gb

    def predict(self, X):
        rf_pred = self.rf.predict(X)
        gb_pred = self.gb.predict(X)
        return (rf_pred + gb_pred) / 2

    def predict_with_std(self, X):
        rf_pred = self.rf.predict(X)
        gb_pred = self.gb.predict(X)
        mean = (rf_pred + gb_pred) / 2
        std = np.abs(rf_pred - gb_pred) / 2
        return mean, std
