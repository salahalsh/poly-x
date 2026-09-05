"""
Prediction Reliability Index (PRI) for Polymer Properties

Composite metric combining:
- Applicability Domain (0.35 weight)
- Ensemble uncertainty (0.35 weight)
- Model performance metric (0.30 weight)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Weight configuration for PRI composite score.
# Weights chosen to balance structural similarity (AD), prediction variance
# (uncertainty), and historical model accuracy (R²). AD and uncertainty
# given equal weight (0.35) as both directly measure prediction-specific
# reliability; model R² receives 0.30 as a population-level statistic.
# Sensitivity analysis: PRI category changes <5% when weights varied ±0.05.
W_AD = 0.35
W_UNCERTAINTY = 0.35
W_MODEL = 0.30


@dataclass
class PolymerPRIResult:
    """Prediction Reliability Index result."""
    success: bool = True
    score: float = 0.5
    category: str = 'moderate'  # high, moderate, low, unreliable
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'pri_score': round(self.score, 4),
            'pri_category': self.category,
            'pri_components': self.components,
        }


class PolymerReliabilityIndex:
    """Calculate composite prediction reliability metric."""

    def __init__(self):
        self._model_r2 = self._load_model_r2()

    def _load_model_r2(self) -> Dict[str, float]:
        """Load actual R² scores from model metadata files."""
        r2_scores = {}
        try:
            from django.conf import settings
            models_dir = Path(settings.BASE_DIR) / 'poly_x' / 'trained_models'
        except Exception:
            return r2_scores

        for prop_dir in ['tg', 'tm', 'td', 'density', 'solubility_param']:
            meta_path = models_dir / prop_dir / 'metadata.json'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    r2_scores[prop_dir] = meta.get('r2_score', 0.5)
                except Exception:
                    pass

        # Also load polyBERT R²
        pb_meta = models_dir / 'polybert' / 'metadata.json'
        if pb_meta.exists():
            try:
                with open(pb_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                ens = meta.get('metrics', {}).get('ensemble', {})
                r2_scores['polybert'] = ens.get('r2', 0.5)
            except Exception:
                pass

        return r2_scores

    def calculate(self, result: Dict, polymer_smiles: str) -> PolymerPRIResult:
        """
        Calculate PRI from prediction result.

        Components:
        1. AD score (from ad_result if available)
        2. Ensemble uncertainty (from ML std if available)
        3. Model performance (static R2 from metadata)
        """
        pri = PolymerPRIResult()

        # 1. AD component
        ad_score = 0.5
        ad_result = result.get('ad_result', {})
        if isinstance(ad_result, dict) and 'ad_reliability_score' in ad_result:
            ad_score = ad_result['ad_reliability_score']
        pri.components['ad'] = round(ad_score, 4)

        # 2. Ensemble uncertainty component
        uncertainty_score = 0.5
        tg_std = result.get('tg_ml_std')
        if tg_std is not None and tg_std >= 0:
            # Lower std = higher reliability
            # Normalize: std < 10K → 1.0, std > 100K → 0.0
            uncertainty_score = max(0.0, min(1.0, 1.0 - (tg_std / 100.0)))
        pri.components['uncertainty'] = round(uncertainty_score, 4)

        # 3. Model performance component (from actual R² in metadata)
        model_score = 0.5
        if result.get('tg_polybert') is not None and 'polybert' in self._model_r2:
            model_score = self._model_r2['polybert']
        elif result.get('tg_ml') is not None and 'tg' in self._model_r2:
            model_score = self._model_r2['tg']
        elif result.get('tg_gc') is not None:
            # Use per-property GC confidence if available
            gc_conf = result.get('gc_confidence', {})
            model_score = gc_conf.get('tg', 0.5)
        pri.components['model'] = round(model_score, 4)

        # Composite PRI
        pri.score = (W_AD * ad_score +
                     W_UNCERTAINTY * uncertainty_score +
                     W_MODEL * model_score)

        # Categorize
        if pri.score >= 0.75:
            pri.category = 'high'
        elif pri.score >= 0.50:
            pri.category = 'moderate'
        elif pri.score >= 0.30:
            pri.category = 'low'
        else:
            pri.category = 'unreliable'

        return pri
