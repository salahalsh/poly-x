"""
Processability Assessment

Evaluates polymer processability based on:
- Tg-Tm processing window
- Thermal stability (Td - Tm gap)
- Chain stiffness indicators
- Recommended processing methods

Scoring formula:
    Base score = 50  (neutral starting point on 0-100 scale)
    + Processing window:  >100K → +20,  >50K → +10,  <30K → -15
    + Thermal margin:     >100K → +15,  >50K → +5,   <30K → -10
    + Chain flexibility:  Flexible → +5, Rigid → -5
    Score is clamped to [0, 100].

Categories (after clamping):
    >= 75 → Excellent,  >= 55 → Good,  >= 35 → Moderate,  < 35 → Difficult

Processing method recommendations based on Tg ranges
(Tadmor & Gogos, Principles of Polymer Processing, 2006):
    Tg < 273K (0°C):   Extrusion, Film blowing
    Tg < 373K (100°C):  Injection molding, Thermoforming
    Tg < 473K (200°C):  Compression molding, Solution casting
    Tg >= 473K:         Compression molding (high-performance polymers)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessabilityResult:
    """Processability assessment result."""
    success: bool = True
    processing_window: Optional[float] = None       # Tm - Tg (K)
    thermal_stability_margin: Optional[float] = None # Td - Tm (K)
    processability_score: float = 50.0               # 0-100
    category: str = 'Moderate'                       # Excellent/Good/Moderate/Difficult
    thermal_stability: str = 'Unknown'               # Excellent/Good/Moderate/Poor
    recommended_methods: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'processing_window': round(self.processing_window, 1) if self.processing_window is not None else None,
            'thermal_stability_margin': round(self.thermal_stability_margin, 1) if self.thermal_stability_margin is not None else None,
            'processability_score': round(self.processability_score, 1),
            'processability_category': self.category,
            'thermal_stability': self.thermal_stability,
            'recommended_methods': self.recommended_methods,
            'processability_warnings': self.warnings,
        }


class ProcessabilityAssessor:
    """Assess polymer processability from predicted properties."""

    def assess(self, result: Dict) -> ProcessabilityResult:
        """
        Assess processability from predicted properties.

        Key criteria:
        - Processing window: Tm - Tg should be > 50K for good processability
        - Thermal margin: Td - Tm should be > 30K
        - Chain stiffness: Flexible chains are easier to process
        """
        proc = ProcessabilityResult()

        # Use explicit None checks - `or` treats 0.0 as falsy
        tg = result.get('tg_gc')
        if tg is None:
            tg = result.get('tg_ml')
        tm = result.get('tm_gc')
        if tm is None:
            tm = result.get('tm_ml')
        td = result.get('td_ml')
        stiffness = result.get('chain_stiffness', 'Unknown')

        score = 50.0  # Base score

        # Processing window: Tm - Tg
        if tg is not None and tm is not None and tm > tg:
            proc.processing_window = tm - tg

            if proc.processing_window > 100:
                score += 20
            elif proc.processing_window > 50:
                score += 10
            elif proc.processing_window < 30:
                score -= 15
                proc.warnings.append(
                    f"Narrow processing window ({proc.processing_window:.0f} K)")
        elif tg is not None and tm is None:
            # Amorphous polymer - no Tm, process above Tg
            score += 10  # Amorphous is generally easier
            proc.recommended_methods.append('Injection molding (above Tg)')

        # Thermal stability margin: Td - Tm
        if td is not None and tm is not None and td > tm:
            proc.thermal_stability_margin = td - tm

            if proc.thermal_stability_margin > 100:
                score += 15
                proc.thermal_stability = 'Excellent'
            elif proc.thermal_stability_margin > 50:
                score += 5
                proc.thermal_stability = 'Good'
            elif proc.thermal_stability_margin < 30:
                score -= 10
                proc.thermal_stability = 'Poor'
                proc.warnings.append(
                    f"Low thermal margin ({proc.thermal_stability_margin:.0f} K) - risk of degradation during processing")
            else:
                proc.thermal_stability = 'Moderate'
        elif td is not None:
            if td > 600:
                proc.thermal_stability = 'Excellent'
            elif td > 450:
                proc.thermal_stability = 'Good'
            else:
                proc.thermal_stability = 'Moderate'
        else:
            proc.thermal_stability = 'Unknown'

        # Chain stiffness impact
        if stiffness == 'Flexible':
            score += 10
        elif stiffness == 'Rigid':
            score -= 10
            proc.warnings.append("Rigid backbone - may require high processing temperatures")

        # Clamp score
        proc.processability_score = max(0.0, min(100.0, score))

        # Categorize
        if proc.processability_score >= 75:
            proc.category = 'Excellent'
        elif proc.processability_score >= 55:
            proc.category = 'Good'
        elif proc.processability_score >= 35:
            proc.category = 'Moderate'
        else:
            proc.category = 'Difficult'

        # Recommend processing methods based on Tg
        if tg is not None:
            if tg < 273:  # Below 0°C
                proc.recommended_methods.append('Extrusion')
                proc.recommended_methods.append('Film blowing')
            elif tg < 373:  # Below 100°C
                proc.recommended_methods.append('Injection molding')
                proc.recommended_methods.append('Extrusion')
                proc.recommended_methods.append('Thermoforming')
            elif tg < 473:  # Below 200°C
                proc.recommended_methods.append('Injection molding (high temp)')
                proc.recommended_methods.append('Compression molding')
            else:  # Very high Tg
                proc.recommended_methods.append('Compression molding')
                proc.recommended_methods.append('Solution casting')

        if not proc.recommended_methods:
            proc.recommended_methods.append('Standard melt processing')

        return proc
