"""
ML Ensemble Predictor for Polymer Properties (Tier 2)

Loads trained RF + GB models from poly_x/trained_models/{property}/
and produces ensemble predictions with uncertainty quantification.

Properties supported: Tg, Tm, Td, density, solubility_param
Each property directory contains:
    checkpoints/{prop}_rf.joblib, {prop}_gb.joblib
    metadata.json, metrics.json
"""

import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolymerMLPrediction:
    """ML ensemble prediction result."""
    success: bool = True
    polymer_smiles: str = ''
    polymer_id: Optional[str] = None

    # ML predictions (ensemble average of RF + GB)
    tg_ml: Optional[float] = None
    tg_ml_std: Optional[float] = None
    tm_ml: Optional[float] = None
    tm_ml_std: Optional[float] = None
    td_ml: Optional[float] = None
    td_ml_std: Optional[float] = None
    density_ml: Optional[float] = None
    density_ml_std: Optional[float] = None
    solubility_param_ml: Optional[float] = None
    solubility_param_ml_std: Optional[float] = None

    # Metadata
    models_used: List[str] = field(default_factory=list)
    prediction_engine: str = 'ML Ensemble (RF+GB)'
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'success': self.success,
            'prediction_engine': self.prediction_engine,
            'ml_models_used': self.models_used,
        }
        for prop in ['tg', 'tm', 'td', 'density', 'solubility_param']:
            val = getattr(self, f'{prop}_ml', None)
            std = getattr(self, f'{prop}_ml_std', None)
            if val is not None:
                d[f'{prop}_ml'] = round(float(val), 2)
            if std is not None:
                d[f'{prop}_ml_std'] = round(float(std), 3)
        if self.warnings:
            d['ml_warnings'] = self.warnings
        return d


class PolymerMLPredictor:
    """
    Load and serve trained polymer property models.

    Scans poly_x/trained_models/ for property directories,
    each containing RF and GB joblib checkpoints.
    """

    # Properties we support
    PROPERTY_NAMES = ['tg', 'tm', 'td', 'density', 'solubility_param']

    def __init__(self, model_dir: Optional[str] = None):
        self._models = {}          # {prop: {'rf': model, 'gb': model}}
        self._metadata = {}        # {prop: metadata dict}
        self._fingerprinter = None

        if model_dir is None:
            from django.conf import settings
            model_dir = str(Path(settings.BASE_DIR) / 'poly_x' / 'trained_models')

        self._model_dir = model_dir
        self._discover_models()

    def _discover_models(self):
        """Scan trained_models/ for available property models."""
        if not os.path.isdir(self._model_dir):
            logger.info("ML model directory not found: %s", self._model_dir)
            return

        for prop in self.PROPERTY_NAMES:
            prop_dir = os.path.join(self._model_dir, prop)
            cp_dir = os.path.join(prop_dir, 'checkpoints')

            rf_path = os.path.join(cp_dir, f'{prop}_rf.joblib')
            gb_path = os.path.join(cp_dir, f'{prop}_gb.joblib')

            if os.path.exists(rf_path) and os.path.exists(gb_path):
                try:
                    from core_utils.safe_pickle import safe_joblib_load
                    rf_model = safe_joblib_load(rf_path)
                    gb_model = safe_joblib_load(gb_path)
                    self._models[prop] = {'rf': rf_model, 'gb': gb_model}

                    # Load metadata if available
                    meta_path = os.path.join(prop_dir, 'metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            self._metadata[prop] = json.load(f)

                    logger.info("Loaded ML models for %s (RF+GB)", prop)
                except Exception as e:
                    logger.warning("Failed to load models for %s: %s", prop, e)

        logger.info("ML Predictor: %d/%d properties available",
                     len(self._models), len(self.PROPERTY_NAMES))

    def clear_cache(self):
        """Release all loaded models to free C-level memory (joblib objects)."""
        self._models.clear()
        self._metadata.clear()
        self._fingerprinter = None
        logger.info("PolymerMLPredictor cache cleared")

    def is_available(self) -> bool:
        """True if at least one property model is loaded."""
        return len(self._models) > 0

    def get_available_properties(self) -> List[str]:
        """Return list of property names with loaded models."""
        return list(self._models.keys())

    def predict(self, polymer_smiles: str,
                polymer_id: Optional[str] = None) -> PolymerMLPrediction:
        """
        Predict all available properties for a polymer.

        Pipeline:
        1. Compute ECFP4 fingerprint (2048-bit)
        2. For each available property: RF prediction + GB prediction
        3. Ensemble = average(RF, GB); Uncertainty = |RF - GB|
        """
        result = PolymerMLPrediction(
            polymer_smiles=polymer_smiles,
            polymer_id=polymer_id,
        )

        if not self._models:
            result.success = False
            result.warnings.append('No trained models available')
            return result

        try:
            # Featurize
            from .polymer_fingerprints import PolymerFingerprinter
            if self._fingerprinter is None:
                self._fingerprinter = PolymerFingerprinter()

            fp = self._fingerprinter.featurize(polymer_smiles)
            if fp is None or fp.sum() == 0:
                result.success = False
                result.warnings.append('Failed to featurize polymer')
                return result

            fp_2d = fp.reshape(1, -1)

            # Predict each property
            for prop, models in self._models.items():
                try:
                    rf_pred = float(models['rf'].predict(fp_2d)[0])
                    gb_pred = float(models['gb'].predict(fp_2d)[0])

                    ensemble_avg = (rf_pred + gb_pred) / 2.0
                    ensemble_std = abs(rf_pred - gb_pred) / 2.0

                    setattr(result, f'{prop}_ml', ensemble_avg)
                    setattr(result, f'{prop}_ml_std', ensemble_std)
                    result.models_used.append(prop)

                except Exception as e:
                    logger.warning("ML prediction failed for %s: %s", prop, e)
                    result.warnings.append(f'{prop}: {e}')

        except Exception as e:
            logger.warning("ML prediction error: %s", e)
            result.success = False
            result.warnings.append(str(e))

        return result
