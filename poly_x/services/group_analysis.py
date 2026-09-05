"""
Group Contribution Analysis - Per-Group Property Breakdown

Shows which functional groups contribute positively/negatively
to each predicted property. Publication-grade visualization data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GroupAnalysisResult:
    """Per-group property contribution breakdown."""
    success: bool = True
    groups: List[Dict] = field(default_factory=list)
    dominant_tg_group: Optional[str] = None
    stiffness_contributors: List[str] = field(default_factory=list)
    flexibility_contributors: List[str] = field(default_factory=list)
    total_groups_matched: int = 0

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'groups': self.groups,
            'dominant_tg_group': self.dominant_tg_group,
            'stiffness_contributors': self.stiffness_contributors,
            'flexibility_contributors': self.flexibility_contributors,
            'total_groups_matched': self.total_groups_matched,
        }


# Groups known to increase/decrease stiffness
STIFFNESS_GROUPS = {
    'Phenylene (p-C6H4)', 'Amide (-CONH-)', 'Nitrile (-CN)',
    'Sulfone (-SO2-)', 'Pyridine ring',
}
FLEXIBILITY_GROUPS = {
    'Methylene (-CH2-)', 'Ether (-O-)', 'Dimethylsiloxane',
    'Methyl (-CH3)', 'Thioether (-S-)',
}


class GroupAnalyzer:
    """Analyze per-group contributions to polymer properties."""

    def analyze(self, polymer_smiles: str,
                groups_found: Dict[str, int],
                group_contributions: Dict[str, Dict]) -> GroupAnalysisResult:
        """
        Analyze group contributions from a GC calculation result.

        Args:
            polymer_smiles: Input PSMILES
            groups_found: {group_name: count} from GC calculator
            group_contributions: {group_name: {count, tg_contribution, ...}}
        """
        result = GroupAnalysisResult()

        if not group_contributions:
            result.success = False
            return result

        # Build per-group breakdown
        groups_list = []
        max_tg_contribution = float('-inf')
        dominant_group = None

        for name, details in group_contributions.items():
            count = details.get('count', 0)
            tg_contrib = details.get('tg_contribution', 0)
            ecoh_contrib = details.get('ecoh_contribution', 0)
            vw_contrib = details.get('vw_contribution', 0)

            groups_list.append({
                'name': name,
                'count': count,
                'tg_contribution': tg_contrib,
                'ecoh_contribution': ecoh_contrib,
                'vw_contribution': vw_contrib,
                'effect_on_tg': 'increases' if tg_contrib > 0 else 'decreases',
            })

            if tg_contrib > max_tg_contribution:
                max_tg_contribution = tg_contrib
                dominant_group = name

        # Sort by absolute Tg contribution (most impactful first)
        groups_list.sort(key=lambda g: abs(g['tg_contribution']), reverse=True)

        result.groups = groups_list
        result.dominant_tg_group = dominant_group
        result.total_groups_matched = sum(
            g.get('count', 0) for g in groups_list)

        # Classify stiffness/flexibility contributors
        for name in groups_found:
            if name in STIFFNESS_GROUPS:
                result.stiffness_contributors.append(name)
            elif name in FLEXIBILITY_GROUPS:
                result.flexibility_contributors.append(name)

        return result
