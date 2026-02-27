"""
Polymer Fingerprinting (Tier 2 featurization)

Provides ECFP4 (Morgan) fingerprints from PSMILES for ML training and inference.
Handles [*] polymer endpoints by replacing with [H].

Reference:
    Rogers, D.; Hahn, M. Extended-connectivity fingerprints.
    J. Chem. Inf. Model. 2010, 50, 742-754.

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""

import logging
from typing import Optional, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

logger = logging.getLogger(__name__)


class PolymerFingerprinter:
    """Generate molecular fingerprints from PSMILES."""

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits)

    def _psmiles_to_mol(self, psmiles: str) -> Optional[Chem.Mol]:
        """Convert PSMILES to RDKit Mol by replacing [*] endpoints."""
        clean = psmiles.strip().replace('[*]', '[H]').replace('*', '[H]')
        mol = Chem.MolFromSmiles(clean)
        return mol

    def featurize(self, psmiles: str) -> np.ndarray:
        """
        Generate ECFP4 fingerprint from PSMILES.

        Returns:
            numpy array of shape (n_bits,) with binary fingerprint
        """
        mol = self._psmiles_to_mol(psmiles)
        if mol is None:
            logger.warning("Could not parse PSMILES: %s", psmiles[:40])
            return np.zeros(self.n_bits, dtype=np.float32)

        fp = self._gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)

    def featurize_batch(self, psmiles_list: List[str]) -> np.ndarray:
        """
        Generate fingerprints for a batch of PSMILES.

        Returns:
            numpy array of shape (n_polymers, n_bits)
        """
        return np.array([self.featurize(s) for s in psmiles_list])

    def compute_tanimoto(self, psmiles_a: str, psmiles_b: str) -> float:
        """Compute Tanimoto similarity between two polymers."""
        mol_a = self._psmiles_to_mol(psmiles_a)
        mol_b = self._psmiles_to_mol(psmiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = self._gen.GetFingerprint(mol_a)
        fp_b = self._gen.GetFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)

    def compute_tanimoto_to_set(self, psmiles: str,
                                 reference_fps: np.ndarray) -> float:
        """
        Compute maximum Tanimoto similarity to a set of reference fingerprints.

        Args:
            psmiles: Query polymer PSMILES
            reference_fps: numpy array of shape (n_ref, n_bits)

        Returns:
            Maximum Tanimoto similarity (0.0 to 1.0)
        """
        query_fp = self.featurize(psmiles)
        if query_fp.sum() == 0:
            return 0.0

        intersection = np.minimum(query_fp, reference_fps).sum(axis=1)
        union = np.maximum(query_fp, reference_fps).sum(axis=1)
        similarities = np.divide(
            intersection, union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0)
        return float(similarities.max()) if len(similarities) > 0 else 0.0
