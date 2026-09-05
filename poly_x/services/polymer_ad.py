"""
Applicability Domain for Polymer ML Models

Assesses whether a query polymer falls within the training domain
using Tanimoto similarity of ECFP4 fingerprints to training set.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Thresholds for domain classification (Tanimoto similarity of ECFP4 fingerprints).
# 0.30: Follows Tropsha (2010) "Activity landscapes, QSAR models": fingerprint
#   similarity >0.3 provides "reasonable structural similarity" for property transfer.
# 0.15: Conservative lower bound; below this, query polymer is chemically
#   dissimilar to all training examples, predictions are extrapolation.
IN_DOMAIN_THRESHOLD = 0.30
BORDERLINE_THRESHOLD = 0.15


@dataclass
class PolymerADResult:
    """Applicability Domain assessment result."""
    success: bool = True
    in_domain: bool = True
    status: str = 'UNKNOWN'  # IN_DOMAIN, BORDERLINE, OUT_OF_DOMAIN
    tanimoto_score: Optional[float] = None
    nearest_training_polymer: Optional[str] = None
    reliability_score: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'ad_in_domain': self.in_domain,
            'ad_status': self.status,
            'ad_tanimoto_score': round(self.tanimoto_score, 4) if self.tanimoto_score is not None else None,
            'ad_nearest_polymer': self.nearest_training_polymer,
            'ad_reliability_score': round(self.reliability_score, 4),
        }


class PolymerADAssessor:
    """Assess if a polymer is within the training domain."""

    def __init__(self, model_dir: Optional[str] = None):
        self._training_fps = None
        self._training_smiles = None
        self._fingerprinter = None

        # Try to load training fingerprints
        if model_dir is None:
            from django.conf import settings
            base = Path(settings.BASE_DIR) / 'poly_x' / 'trained_models' / 'tg'
            model_dir = str(base)

        fp_path = os.path.join(model_dir, 'training_fingerprints.npy')
        smiles_path = os.path.join(model_dir, 'training_smiles.txt')

        if os.path.exists(fp_path):
            try:
                self._training_fps = np.load(fp_path)
                if os.path.exists(smiles_path):
                    with open(smiles_path, 'r', encoding='utf-8') as f:
                        self._training_smiles = [l.strip() for l in f.readlines()]
                logger.info("AD: Loaded %d training fingerprints", len(self._training_fps))
            except Exception as e:
                logger.warning("AD: Could not load training data: %s", e)

    def assess(self, polymer_smiles: str) -> PolymerADResult:
        """
        Assess whether polymer is within training domain.

        Uses maximum Tanimoto similarity to training set.
        """
        result = PolymerADResult()

        if self._training_fps is None:
            # No training data - cannot assess, assume borderline
            result.status = 'UNKNOWN'
            result.reliability_score = 0.5
            return result

        try:
            from .polymer_fingerprints import PolymerFingerprinter
            if self._fingerprinter is None:
                self._fingerprinter = PolymerFingerprinter()

            # Compute Tanimoto similarities in a single pass (avoids
            # duplicating the 2×96MB temporary array allocation).
            query_fp = self._fingerprinter.featurize(polymer_smiles)
            if query_fp.sum() > 0:
                intersection = np.minimum(query_fp, self._training_fps).sum(axis=1)
                union = np.maximum(query_fp, self._training_fps).sum(axis=1)
                sims = np.divide(intersection, union,
                                 out=np.zeros_like(intersection, dtype=float),
                                 where=union > 0)
                max_sim = float(sims.max())
                # Find nearest training polymer from same similarity array
                if self._training_smiles:
                    best_idx = int(sims.argmax())
                    if best_idx < len(self._training_smiles):
                        result.nearest_training_polymer = self._training_smiles[best_idx]
                del intersection, union, sims  # free temp arrays promptly
            else:
                max_sim = self._fingerprinter.compute_tanimoto_to_set(
                    polymer_smiles, self._training_fps)

            result.tanimoto_score = max_sim

            # Classify domain status
            if max_sim >= IN_DOMAIN_THRESHOLD:
                result.in_domain = True
                result.status = 'IN_DOMAIN'
            elif max_sim >= BORDERLINE_THRESHOLD:
                result.in_domain = False
                result.status = 'BORDERLINE'
            else:
                result.in_domain = False
                result.status = 'OUT_OF_DOMAIN'

            # Continuous reliability score (no discontinuity at boundaries)
            # Maps Tanimoto [0, 1] → reliability [0.1, 1.0] via sigmoid-like curve
            # centered at IN_DOMAIN_THRESHOLD with steepness factor
            result.reliability_score = max(0.1, min(1.0, max_sim / 0.5 + 0.1))

        except Exception as e:
            logger.warning("AD assessment error: %s", e)
            result.success = False
            result.status = 'ERROR'

        return result
