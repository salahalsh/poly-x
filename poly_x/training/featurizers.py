"""
Training Featurizer - Batch ECFP4 fingerprint generation.

Wraps PolymerFingerprinter for use in training pipelines,
with progress logging and parallel processing support.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingFeaturizer:
    """
    Batch featurizer for training pipelines.

    Uses ECFP4 Morgan fingerprints (2048-bit) from PolymerFingerprinter.
    Handles failures gracefully by replacing invalid molecules with zeros.
    """

    def __init__(self, n_bits: int = 2048, radius: int = 2):
        from poly_x.services.polymer_fingerprints import PolymerFingerprinter
        self._fp = PolymerFingerprinter(n_bits=n_bits, radius=radius)
        self.n_bits = n_bits
        self.radius = radius

    def featurize(self, smiles_list: List[str],
                  log_every: int = 10000) -> np.ndarray:
        """
        Featurize a list of PSMILES strings.

        Args:
            smiles_list: List of PSMILES strings
            log_every: Log progress every N molecules

        Returns:
            numpy array of shape (n_samples, n_bits)
        """
        n = len(smiles_list)
        fps = np.zeros((n, self.n_bits), dtype=np.float32)
        n_failed = 0

        for i, smi in enumerate(smiles_list):
            try:
                fp = self._fp.featurize(smi)
                if fp is not None and fp.sum() > 0:
                    fps[i] = fp
                else:
                    n_failed += 1
            except Exception:
                n_failed += 1

            if log_every > 0 and (i + 1) % log_every == 0:
                logger.info("Featurized %d/%d (%.1f%%, %d failed)",
                            i + 1, n, 100 * (i + 1) / n, n_failed)

        logger.info("Featurization complete: %d/%d succeeded (%.1f%%)",
                     n - n_failed, n, 100 * (n - n_failed) / n if n > 0 else 0)

        if n_failed > 0:
            logger.warning("%d molecules failed featurization", n_failed)

        return fps

    def get_valid_mask(self, fps: np.ndarray) -> np.ndarray:
        """Return boolean mask for rows that have non-zero fingerprints."""
        return fps.sum(axis=1) > 0
