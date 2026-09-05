"""
POLY-X Services Package - Polymer Property Prediction

Three Tiers:
- Tier 1: Group Contribution (Van Krevelen, always available)
- Tier 2: ML Ensemble (RF+GB on PolyMetriX, requires trained models)
- Tier 3: polyBERT Embeddings (requires transformers + HuggingFace model)

Enhanced Features:
- Group Contribution Analysis (per-group breakdown)
- Applicability Domain Assessment
- Prediction Reliability Index (PRI)
- Comparative Polymer Profiling
- Processability Assessment

All imports are lazy to avoid loading heavy ML/transformer libraries
at Django startup.
"""


_IMPORT_MAP = {
    # Core Tiers
    'GroupContributionCalculator': '.group_contribution',
    'GroupContributionResult': '.group_contribution',
    'PolymerMLPredictor': '.trained_ml_models',
    'PolymerMLPrediction': '.trained_ml_models',
    'PolyBERTPredictor': '.polybert_predictor',
    'PolyBERTPrediction': '.polybert_predictor',
    # Aggregator
    'PolyPropertyAggregator': '.poly_aggregator',
    'get_aggregator': '.poly_aggregator',
    # Featurization
    'PolymerFingerprinter': '.polymer_fingerprints',
    # Enhanced Features
    'GroupAnalyzer': '.group_analysis',
    'GroupAnalysisResult': '.group_analysis',
    'PolymerADAssessor': '.polymer_ad',
    'PolymerADResult': '.polymer_ad',
    'PolymerReliabilityIndex': '.polymer_reliability',
    'PolymerPRIResult': '.polymer_reliability',
    'PolymerComparativeProfiler': '.polymer_comparison',
    'ComparativeResult': '.polymer_comparison',
    'ProcessabilityAssessor': '.processability',
    'ProcessabilityResult': '.processability',
    # Model Management
    'PolymerModelManager': '.model_manager',
}

# Cache resolved attributes to avoid repeated importlib calls
_CACHE = {}


def __getattr__(name):
    """Lazy import with caching - only load heavy modules when first accessed."""
    if name in _CACHE:
        return _CACHE[name]

    if name in _IMPORT_MAP:
        import importlib
        module = importlib.import_module(_IMPORT_MAP[name], package=__name__)
        attr = getattr(module, name)
        _CACHE[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core Tiers
    'GroupContributionCalculator',
    'GroupContributionResult',
    'PolymerMLPredictor',
    'PolymerMLPrediction',
    'PolyBERTPredictor',
    'PolyBERTPrediction',
    # Aggregator
    'PolyPropertyAggregator',
    'get_aggregator',
    # Featurization
    'PolymerFingerprinter',
    # Enhanced Features
    'GroupAnalyzer',
    'GroupAnalysisResult',
    'PolymerADAssessor',
    'PolymerADResult',
    'PolymerReliabilityIndex',
    'PolymerPRIResult',
    'PolymerComparativeProfiler',
    'ComparativeResult',
    'ProcessabilityAssessor',
    'ProcessabilityResult',
    # Model Management
    'PolymerModelManager',
]

# Mobile API entry point
from .mobile import submit_poly_x  # noqa: F401, E402
