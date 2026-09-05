"""POLY-X: multi-tier polymer property prediction.

Tier 1  Van Krevelen group contribution (physics-based)
Tier 2  RF + GB ensemble on Morgan fingerprints (data-driven)
Tier 3  frozen polyBERT embeddings with RF/GB heads

The implementations live in ``poly_x.services`` and are the same modules the
deployed platform runs. ``poly_x.training`` holds the code that produced the
Tier 2 and Tier 3 checkpoints, and ``paper/analysis`` holds the twelve numbered
scripts that produced every number in the manuscript.

Reference:
    Jebril, I.H.; Alshehade, S.A. POLY-X: A Multi-Tier Platform for Polymer
    Property Prediction. Journal of Computer-Aided Molecular Design, 2026.
"""

__version__ = "1.0.0"
