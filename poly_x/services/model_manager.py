"""
Polymer Model Manager - Singleton for Model Discovery and Caching

Discovers trained models in poly_x/trained_models/ and provides
unified access for the ML predictor and training pipeline.
"""

import logging
import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class PolymerModelManager:
    """
    Discover and manage trained polymer property models.

    Thread-safe singleton. Scans trained_models/ for metadata.json files
    and provides model info without loading the actual joblib weights.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            from django.conf import settings
            model_dir = str(
                Path(settings.BASE_DIR) / 'poly_x' / 'trained_models')

        self._model_dir = model_dir
        self._model_info = {}   # {property_name: metadata dict}
        self._discover()

    def _discover(self):
        """Scan for trained models by finding metadata.json files."""
        if not os.path.isdir(self._model_dir):
            logger.info("No model directory found at %s", self._model_dir)
            return

        for entry in os.listdir(self._model_dir):
            prop_dir = os.path.join(self._model_dir, entry)
            meta_path = os.path.join(prop_dir, 'metadata.json')

            if os.path.isdir(prop_dir) and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    meta['_dir'] = prop_dir
                    self._model_info[entry] = meta
                    logger.info("Discovered model: %s (v%s)",
                                entry, meta.get('version', '?'))
                except Exception as e:
                    logger.warning("Failed to read metadata for %s: %s",
                                   entry, e)

        logger.info("ModelManager: %d model(s) discovered", len(self._model_info))

    def get_available_properties(self) -> List[str]:
        """Return property names with trained models."""
        return list(self._model_info.keys())

    def get_model_info(self, property_name: str) -> Optional[Dict]:
        """Return metadata for a specific property model."""
        return self._model_info.get(property_name)

    def get_all_info(self) -> Dict[str, Any]:
        """Return summary of all available models."""
        summary = {}
        for prop, meta in self._model_info.items():
            summary[prop] = {
                'version': meta.get('version', 'unknown'),
                'model_type': meta.get('model_type', 'RF+GB Ensemble'),
                'n_training': meta.get('n_training', 0),
                'r2_score': meta.get('r2_score'),
                'mae': meta.get('mae'),
                'rmse': meta.get('rmse'),
                'trained_date': meta.get('trained_date'),
            }
        return summary

    def has_model(self, property_name: str) -> bool:
        """Check if a model exists for a property."""
        return property_name in self._model_info

    def get_model_dir(self, property_name: str) -> Optional[str]:
        """Return directory path for a property model."""
        info = self._model_info.get(property_name)
        if info:
            return info.get('_dir')
        return None

    def refresh(self):
        """Re-scan model directory (after training new models)."""
        self._model_info.clear()
        self._discover()
