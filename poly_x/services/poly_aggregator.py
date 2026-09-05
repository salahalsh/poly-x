"""
Polymer Property Aggregator - Central Orchestrator

Combines results from multiple prediction engines (3 tiers)
and adds enhanced features (5 services) sequentially.

Thread-safe singleton: use get_aggregator() for cached instance.
"""

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PolyPropertyAggregator:
    """
    Aggregate polymer property predictions across 3 tiers.

    Tier 1: Group Contribution (Van Krevelen) - always available
    Tier 2: ML Ensemble (RF+GB on PolyMetriX) - requires trained models
    Tier 3: polyBERT Embeddings - requires transformers library

    Thread-safe singleton for Django views. Use fresh instance for Celery tasks.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, skip_polybert: bool = True):
        """Thread-safe singleton accessor (web process, skips polyBERT)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(skip_polybert=skip_polybert)
        return cls._instance

    def __init__(self, skip_polybert: bool = False):
        logger.info("Initializing PolyPropertyAggregator...")

        # ── Tier 1: Group Contribution (always available) ──
        from .group_contribution import GroupContributionCalculator
        self.gc_calculator = GroupContributionCalculator()
        logger.info("Tier 1 (Group Contribution): Ready")

        # ── Tier 2: ML Ensemble (conditional) ──
        self.ml_predictor = None
        self.ml_available = False
        try:
            from .trained_ml_models import PolymerMLPredictor
            self.ml_predictor = PolymerMLPredictor()
            self.ml_available = self.ml_predictor.is_available()
            logger.info("Tier 2 (ML Ensemble): %s",
                        "Ready" if self.ml_available else "No trained models found")
        except Exception as e:
            logger.warning("Tier 2 (ML Ensemble) unavailable: %s", e)

        # ── Tier 3: polyBERT (conditional) ──
        # skip_polybert=True in the web process (512MB) to avoid OOM;
        # polyBERT loads the 2GB DeBERTa model and should only run in
        # the Celery worker (2GB RAM).
        self.polybert = None
        self.polybert_available = False
        if not skip_polybert:
            try:
                from .polybert_predictor import PolyBERTPredictor
                self.polybert = PolyBERTPredictor()
                self.polybert_available = self.polybert.is_available()
                logger.info("Tier 3 (polyBERT): %s",
                            "Ready" if self.polybert_available else "Model not loaded")
            except Exception as e:
                logger.warning("Tier 3 (polyBERT) unavailable: %s", e)
        else:
            logger.info("Tier 3 (polyBERT): Skipped (web process, memory-constrained)")

        # ── Enhanced Features ──
        self._init_enhanced_features()

        logger.info("PolyPropertyAggregator initialized. Engines: GC=True, ML=%s, polyBERT=%s",
                     self.ml_available, self.polybert_available)

    def _init_enhanced_features(self):
        """Initialize enhanced feature services with graceful degradation."""
        self.group_analyzer = None
        self.ad_assessor = None
        self.pri_calculator = None
        self.comparative_profiler = None
        self.processability_assessor = None

        try:
            from .group_analysis import GroupAnalyzer
            self.group_analyzer = GroupAnalyzer()
        except Exception as e:
            logger.warning("GroupAnalyzer unavailable: %s", e)

        try:
            from .polymer_ad import PolymerADAssessor
            self.ad_assessor = PolymerADAssessor()
        except Exception as e:
            logger.warning("PolymerADAssessor unavailable: %s", e)

        try:
            from .polymer_reliability import PolymerReliabilityIndex
            self.pri_calculator = PolymerReliabilityIndex()
        except Exception as e:
            logger.warning("PolymerReliabilityIndex unavailable: %s", e)

        try:
            from .polymer_comparison import PolymerComparativeProfiler
            self.comparative_profiler = PolymerComparativeProfiler()
        except Exception as e:
            logger.warning("PolymerComparativeProfiler unavailable: %s", e)

        try:
            from .processability import ProcessabilityAssessor
            self.processability_assessor = ProcessabilityAssessor()
        except Exception as e:
            logger.warning("ProcessabilityAssessor unavailable: %s", e)

    def calculate(self, polymer_smiles: str,
                  polymer_id: Optional[str] = None,
                  engine: str = 'GROUP_CONTRIBUTION',
                  include_comparison: bool = True,
                  include_processability: bool = True) -> Dict:
        """
        Calculate polymer properties using specified engine.

        Args:
            polymer_smiles: PSMILES with [*] endpoints
            polymer_id: Optional identifier
            engine: GROUP_CONTRIBUTION | ML | POLYBERT | ALL
            include_comparison: Include comparative profiling
            include_processability: Include processability assessment

        Returns:
            Dict with all predicted properties
        """
        try:
            if engine == 'GROUP_CONTRIBUTION':
                gc_result = self.gc_calculator.calculate(polymer_smiles, polymer_id)
                result = gc_result.to_dict()

            elif engine == 'ML':
                # Start with GC baseline, overlay ML predictions
                gc_result = self.gc_calculator.calculate(polymer_smiles, polymer_id)
                result = gc_result.to_dict()
                if self.ml_available and self.ml_predictor:
                    ml_result = self.ml_predictor.predict(polymer_smiles, polymer_id)
                    ml_dict = ml_result.to_dict()
                    ml_dict.pop('success', None)  # Don't overwrite GC success
                    result.update(ml_dict)
                    result['prediction_engine'] = 'Group Contribution + ML Ensemble'
                else:
                    existing = result.get('prediction_warnings') or ''
                    result['prediction_warnings'] = (
                        existing + '; ML models not available, using GC only'
                    ).strip('; ')

            elif engine == 'POLYBERT':
                gc_result = self.gc_calculator.calculate(polymer_smiles, polymer_id)
                result = gc_result.to_dict()
                if self.polybert_available and self.polybert:
                    pb_result = self.polybert.predict(polymer_smiles, polymer_id)
                    pb_dict = pb_result.to_dict()
                    pb_dict.pop('success', None)  # Don't overwrite GC success
                    result.update(pb_dict)
                    result['prediction_engine'] = 'Group Contribution + polyBERT'
                else:
                    existing = result.get('prediction_warnings') or ''
                    result['prediction_warnings'] = (
                        existing + '; polyBERT not available, using GC only'
                    ).strip('; ')

            elif engine == 'ALL':
                # Combine all available tiers
                gc_result = self.gc_calculator.calculate(polymer_smiles, polymer_id)
                result = gc_result.to_dict()
                engines_used = ['Group Contribution']

                if self.ml_available and self.ml_predictor:
                    try:
                        ml_result = self.ml_predictor.predict(polymer_smiles, polymer_id)
                        ml_dict = ml_result.to_dict()
                        ml_dict.pop('success', None)
                        result.update(ml_dict)
                        engines_used.append('ML Ensemble')
                    except Exception as e:
                        logger.warning("ML prediction failed: %s", e)

                if self.polybert_available and self.polybert:
                    try:
                        pb_result = self.polybert.predict(polymer_smiles, polymer_id)
                        pb_dict = pb_result.to_dict()
                        pb_dict.pop('success', None)
                        result.update(pb_dict)
                        engines_used.append('polyBERT')
                    except Exception as e:
                        logger.warning("polyBERT prediction failed: %s", e)

                result['prediction_engine'] = ' + '.join(engines_used)

            else:
                return {
                    'success': False,
                    'error': f"Unknown engine: {engine}",
                    'polymer_smiles': polymer_smiles,
                }

            # Add enhanced features
            result = self._add_enhanced_features(
                result, polymer_smiles,
                include_comparison=include_comparison,
                include_processability=include_processability,
            )

            return result

        except Exception as e:
            logger.exception("Calculation error for %s: %s", polymer_smiles[:40], e)
            return {
                'success': False,
                'error': str(e),
                'polymer_smiles': polymer_smiles,
                'polymer_id': polymer_id,
            }

    def _add_enhanced_features(self, result: Dict, polymer_smiles: str,
                                include_comparison: bool = True,
                                include_processability: bool = True) -> Dict:
        """
        Add enhanced features sequentially with try/except.
        Each feature is independent - failure of one does not affect others.
        """
        # 1. Group Contribution Analysis
        if self.group_analyzer:
            try:
                ga_result = self.group_analyzer.analyze(
                    polymer_smiles, result.get('groups_found', {}),
                    result.get('group_contributions', {}))
                result['group_analysis'] = ga_result.to_dict()
            except Exception as e:
                logger.warning("Group analysis failed: %s", e)

        # 2. Applicability Domain
        if self.ad_assessor:
            try:
                ad_result = self.ad_assessor.assess(polymer_smiles)
                result['ad_result'] = ad_result.to_dict()
            except Exception as e:
                logger.warning("AD assessment failed: %s", e)

        # 3. Prediction Reliability Index
        if self.pri_calculator:
            try:
                pri_result = self.pri_calculator.calculate(result, polymer_smiles)
                result['pri_result'] = pri_result.to_dict()
            except Exception as e:
                logger.warning("PRI calculation failed: %s", e)

        # 4. Comparative Profiling
        if include_comparison and self.comparative_profiler:
            try:
                comp_result = self.comparative_profiler.compare(
                    polymer_smiles, result)
                result['comparative_result'] = comp_result.to_dict()
            except Exception as e:
                logger.warning("Comparative profiling failed: %s", e)

        # 5. Processability Assessment
        if include_processability and self.processability_assessor:
            try:
                proc_result = self.processability_assessor.assess(result)
                result['processability_result'] = proc_result.to_dict()
                # Promote overall assessment to top level
                result['overall_processability'] = proc_result.category
                result['overall_thermal_stability'] = proc_result.thermal_stability
            except Exception as e:
                logger.warning("Processability assessment failed: %s", e)

        return result

    def batch_calculate(self, polymer_list: List[Dict],
                        engine: str = 'GROUP_CONTRIBUTION') -> List[Dict]:
        """
        Batch prediction for multiple polymers.

        Args:
            polymer_list: List of {'smiles': '...', 'polymer_id': '...', 'name': '...'}
            engine: Prediction engine

        Returns:
            List of result dicts
        """
        results = []
        for polymer in polymer_list:
            result = self.calculate(
                polymer.get('smiles', ''),
                polymer_id=polymer.get('polymer_id'),
                engine=engine,
            )
            result['polymer_name'] = polymer.get('name', '')
            results.append(result)
        return results

    def get_available_engines(self) -> Dict[str, bool]:
        """Return availability status of each prediction engine."""
        return {
            'GROUP_CONTRIBUTION': True,
            'ML': self.ml_available,
            'POLYBERT': self.polybert_available,
            'ALL': True,  # Always available (at least GC works)
        }


def get_aggregator(use_singleton: bool = True) -> PolyPropertyAggregator:
    """
    Get PolyPropertyAggregator instance.

    Args:
        use_singleton: True for Django views (cached, skips polyBERT),
                       False for Celery tasks (fresh, loads all tiers)
    """
    if use_singleton:
        return PolyPropertyAggregator.get_instance(skip_polybert=True)
    # Celery worker: load all tiers including polyBERT (2GB RAM available)
    return PolyPropertyAggregator(skip_polybert=False)
