"""
Comparative Polymer Profiling

Compare query polymer against ~20 well-known commercial polymers
using Tanimoto similarity and property comparison.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Reference polymers with known experimental properties.
# All temperatures in Kelvin, density in g/cm³, solubility_param in MPa^0.5.
#
# Sources:
#   PH  = Brandrup, Immergut & Grulke, Polymer Handbook 4th Ed. (1999)
#   VK  = Van Krevelen & Te Nijenhuis, Properties of Polymers 4th Ed. (2009)
#   Mar = Mark, Physical Properties of Polymers Handbook 2nd Ed. (2007)
#
# Properties marked None = amorphous polymer (no Tm).
REFERENCE_POLYMERS = [
    {'name': 'Polyethylene (HDPE)',     'psmiles': '[*]CC[*]',             # PH VII/1
     'tg': 148, 'tm': 408, 'density': 0.96, 'solubility_param': 16.0},
    {'name': 'Polypropylene (iPP)',     'psmiles': '[*]CC(C)[*]',          # PH VII/7
     'tg': 253, 'tm': 449, 'density': 0.90, 'solubility_param': 15.5},
    {'name': 'Polystyrene (PS)',        'psmiles': '[*]CC(c1ccccc1)[*]',   # PH VII/15 (atactic)
     'tg': 373, 'tm': None, 'density': 1.05, 'solubility_param': 18.5},
    {'name': 'Poly(vinyl chloride)',    'psmiles': '[*]CC(Cl)[*]',         # PH VII/11
     'tg': 354, 'tm': None, 'density': 1.40, 'solubility_param': 19.5},
    {'name': 'PMMA',                    'psmiles': '[*]CC(C)(C(=O)OC)[*]', # PH VII/30 (atactic)
     'tg': 378, 'tm': None, 'density': 1.18, 'solubility_param': 18.6},
    {'name': 'PET',                     'psmiles': '[*]CCOC(=O)c1ccc(C(=O)O[*])cc1',  # VK T6.1
     'tg': 342, 'tm': 533, 'density': 1.38, 'solubility_param': 21.8},
    {'name': 'Nylon-6',                 'psmiles': '[*]CCCCCC(=O)N[*]',    # PH VII/100
     'tg': 323, 'tm': 496, 'density': 1.14, 'solubility_param': 22.5},
    {'name': 'Nylon-6,6',              'psmiles': '[*]CCCCCCNC(=O)CCCCC(=O)N[*]',  # PH VII/101
     'tg': 323, 'tm': 538, 'density': 1.14, 'solubility_param': 22.5},
    {'name': 'Polycarbonate (PC)',      'psmiles': '[*]Oc1ccc(C(C)(C)c2ccc(O[*])cc2)cc1',  # VK T6.1
     'tg': 423, 'tm': 500, 'density': 1.20, 'solubility_param': 19.4},
    {'name': 'Poly(lactic acid)',       'psmiles': '[*]CC(C)C(=O)O[*]',    # Mar Ch.16
     'tg': 333, 'tm': 453, 'density': 1.24, 'solubility_param': 20.0},
    {'name': 'PTFE',                    'psmiles': '[*]C(F)(F)C(F)(F)[*]', # PH VII/12
     'tg': 160, 'tm': 600, 'density': 2.15, 'solubility_param': 12.5},
    {'name': 'PEEK',                    'psmiles': '[*]Oc1ccc(Oc2ccc(C(=O)c3ccc([*])cc3)cc2)cc1',  # VK T6.1
     'tg': 416, 'tm': 616, 'density': 1.30, 'solubility_param': 22.0},
    {'name': 'Polyimide (Kapton)',      'psmiles': '[*]c1ccc2c(c1)C(=O)N(c1ccc(Oc3ccc([*])cc3)cc1)C2=O',  # VK T6.1
     'tg': 633, 'tm': None, 'density': 1.42, 'solubility_param': 22.0},
    {'name': 'PDMS',                    'psmiles': '[*][Si](C)(C)O[*]',    # VK Ch.6 p.138
     'tg': 150, 'tm': 233, 'density': 0.97, 'solubility_param': 15.0},
    {'name': 'Polyurethane (generic)',   'psmiles': '[*]CCCCCCOC(=O)NC(=O)O[*]',  # Mar Ch.21
     'tg': 213, 'tm': 463, 'density': 1.10, 'solubility_param': 20.0},
    {'name': 'POM (polyoxymethylene)',   'psmiles': '[*]CO[*]',            # PH VII/60
     'tg': 198, 'tm': 448, 'density': 1.42, 'solubility_param': 22.0},
    {'name': 'Poly(ethylene oxide)',    'psmiles': '[*]CCO[*]',            # PH VII/61
     'tg': 206, 'tm': 339, 'density': 1.13, 'solubility_param': 20.2},
    {'name': 'Poly(vinyl alcohol)',     'psmiles': '[*]CC(O)[*]',          # PH VII/19
     'tg': 358, 'tm': 503, 'density': 1.29, 'solubility_param': 25.8},
    {'name': 'Polysulfone (PSU)',       'psmiles': '[*]Oc1ccc(S(=O)(=O)c2ccc(O[*])cc2)cc1',  # VK T6.1
     'tg': 463, 'tm': None, 'density': 1.24, 'solubility_param': 21.0},
    {'name': 'ABS (approx.)',           'psmiles': '[*]CC(c1ccccc1)CC(C#N)[*]',  # Mar Ch.13
     'tg': 378, 'tm': None, 'density': 1.05, 'solubility_param': 18.8},
]


@dataclass
class ComparativeResult:
    """Comparative polymer profiling result."""
    success: bool = True
    most_similar_polymer: Optional[str] = None
    most_similar_score: float = 0.0
    comparisons: List[Dict] = field(default_factory=list)
    property_percentiles: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'most_similar_polymer': self.most_similar_polymer,
            'most_similar_score': round(self.most_similar_score, 4),
            'comparisons': self.comparisons[:5],  # Top 5
            'property_percentiles': self.property_percentiles,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
        }


class PolymerComparativeProfiler:
    """Compare query polymer against reference commercial polymers."""

    def __init__(self):
        self._fingerprinter = None

    def compare(self, polymer_smiles: str, result: Dict) -> ComparativeResult:
        """Compare polymer against reference set."""
        comp = ComparativeResult()

        try:
            from .polymer_fingerprints import PolymerFingerprinter
            if self._fingerprinter is None:
                self._fingerprinter = PolymerFingerprinter()

            # Compute similarity to each reference polymer
            comparisons = []
            for ref in REFERENCE_POLYMERS:
                sim = self._fingerprinter.compute_tanimoto(
                    polymer_smiles, ref['psmiles'])
                comparisons.append({
                    'name': ref['name'],
                    'psmiles': ref['psmiles'],
                    'tanimoto': round(sim, 4),
                    'tg': ref.get('tg'),
                    'tm': ref.get('tm'),
                    'density': ref.get('density'),
                })

            # Sort by similarity (most similar first)
            comparisons.sort(key=lambda x: x['tanimoto'], reverse=True)
            comp.comparisons = comparisons

            if comparisons:
                comp.most_similar_polymer = comparisons[0]['name']
                comp.most_similar_score = comparisons[0]['tanimoto']

            # Property percentiles (explicit None check - `or` treats 0.0 as falsy)
            query_tg = result.get('tg_gc')
            if query_tg is None:
                query_tg = result.get('tg_ml')
            query_density = result.get('density_gc')
            if query_density is None:
                query_density = result.get('density_ml')

            if query_tg is not None:
                ref_tgs = [r['tg'] for r in REFERENCE_POLYMERS if r.get('tg')]
                below = sum(1 for t in ref_tgs if t < query_tg)
                comp.property_percentiles['tg'] = round(
                    100.0 * below / len(ref_tgs), 1) if ref_tgs else 50.0

                # Strengths/weaknesses
                if query_tg > 400:
                    comp.strengths.append("High Tg (above 400 K) - good thermal resistance")
                elif query_tg < 200:
                    comp.weaknesses.append("Low Tg (below 200 K) - limited thermal resistance")

            if query_density is not None:
                ref_dens = [r['density'] for r in REFERENCE_POLYMERS if r.get('density')]
                below = sum(1 for d in ref_dens if d < query_density)
                comp.property_percentiles['density'] = round(
                    100.0 * below / len(ref_dens), 1) if ref_dens else 50.0

                if query_density < 1.0:
                    comp.strengths.append("Low density - lightweight material")
                elif query_density > 1.5:
                    comp.weaknesses.append("High density (> 1.5 g/cm3)")

        except Exception as e:
            logger.warning("Comparative profiling error: %s", e)
            comp.success = False

        return comp
