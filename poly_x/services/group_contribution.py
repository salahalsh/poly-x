"""
Van Krevelen Group Contribution for Polymer Properties

Predicts: Tg, Tm, density, solubility parameter, CED, molar volume
from chemical structure of repeat unit using group additivity.

Uses atom-exclusive matching: composite groups (ester, amide, etc.)
are matched first and their atoms excluded from simpler groups.

References:
    Van Krevelen & Te Nijenhuis, Properties of Polymers, 4th Ed., 2009
    Bicerano, Prediction of Polymer Properties, 3rd Ed., 2002
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# Van Krevelen Group Contribution Table
#
# Each entry: (SMARTS, name, Yg, Ecoh, Vw, priority)
#   Yg   = molar Tg contribution (K·g/mol ÷ 1000 for our formula)
#   Ecoh = molar cohesive energy (J/mol)   - VK Table 7.2
#   Vw   = van der Waals volume (cm³/mol)  - VK Table 4.6
#   priority = matching order (higher = matched first, atoms consumed)
#
# Yg calibrated against experimental Tg (Polymer Handbook, 4th Ed.):
#   PE=195K, PP=253K, PS=373K, PVC=354K, PET=342K, PVOH=358K,
#   Nylon-6=323K, PEO=206K, POM=198K, PDMS=150K
#
# Key references (abbreviated as VK, Bic, PH):
#   VK  = Van Krevelen & Te Nijenhuis (2009) Properties of Polymers, 4th Ed.
#   Bic = Bicerano (2002) Prediction of Polymer Properties, 3rd Ed.
#   PH  = Brandrup, Immergut & Grulke (1999) Polymer Handbook, 4th Ed.
# ══════════════════════════════════════════════════════════════════════

# Priority 3: Composite groups (matched first, consume all their atoms)
# Priority 2: Ring systems
# Priority 1: Simple groups (matched last, only on unconsumed atoms)

GROUP_CONTRIBUTIONS = [
    # ── Priority 3: Composite groups (multi-atom, matched first) ──
    # SMARTS            Name                      Yg     Ecoh    Vw   Pri  Source
    ('[C](=O)[NH]',  'Amide (-CONH-)',         23.0,  33500,  19.5, 3),  # VK T6.2, T7.2, T4.6
    ('[C](=O)[OD2]', 'Ester (-COO-)',          14.0,  18000,  18.0, 3),  # VK T6.2, T7.2, T4.6
    ('S(=O)(=O)',     'Sulfone (-SO2-)',        25.8,  23400,  19.6, 3),  # VK T7.2; Yg from Bic T5.3
    ('[Si](C)(C)O',  'Dimethylsiloxane',        9.0,   6000,  55.0, 3),  # VK Ch.6 (PDMS Tg=150K cal.)
    ('C#N',           'Nitrile (-CN)',          25.0,  25500,  24.0, 3),  # VK T7.2, Bic T5.3
    ('[N+](=O)[O-]',  'Nitro (-NO2)',           6.0,  11500,  24.0, 3),  # VK T7.2

    # ── Priority 2: Ring systems ──
    ('c1ccccc1',      'Phenylene (p-C6H4)',    31.0,  31940,  52.4, 2),  # VK T6.2, T7.2, T4.6 p.88
    ('c1ccncc1',      'Pyridine ring',         33.0,  33400,  48.0, 2),  # Bic T5.3
    ('c1ccoc1',       'Furan ring',            28.0,  28000,  42.0, 2),  # Bic T5.3
    ('C1CCCCC1',      'Cyclohexyl',            26.0,  26000,  68.0, 2),  # VK T4.6 (Vw), Bic (Ecoh)
    ('C1CCCC1',       'Cyclopentyl',           22.0,  22000,  55.0, 2),  # Bic T5.3

    # ── Priority 1: Simple groups (on remaining atoms) ──
    ('[CH3]',         'Methyl (-CH3)',           2.4,   4710,  33.5, 1),  # VK T4.6 p.72, T7.2 p.195
    ('[CH2]',         'Methylene (-CH2-)',       2.7,   4940,  16.1, 1),  # VK T4.6 p.72, T7.2 p.195
    ('[CH1;A]',       'Methine (-CH<)',          5.5,   3430,   6.5, 1),  # VK T4.6, T7.2
    ('[CH0;A]',       'Quaternary C (-C<)',      8.7,   1470,  -1.0, 1),  # VK T4.6 (neg Vw = overlap)
    ('[OD2;!$(O=*)]', 'Ether (-O-)',            3.5,   3350,   3.8, 1),  # VK T7.2, T4.6
    ('[C;A](=O)',     'Carbonyl (C=O)',          8.0,  17370,  10.8, 1),  # VK T7.2, T4.6
    ('[OH]',          'Hydroxyl (-OH)',          7.6,  29800,  10.0, 1),  # VK T7.2 (H-bond corrected)
    ('[NH]',          'Secondary amine (-NH-)',  8.4,   8400,   4.5, 1),  # VK T7.2, T4.6
    ('[NH2]',         'Primary amine (-NH2)',    9.0,  12600,   9.0, 1),  # VK T7.2, T4.6
    ('[SD2]',         'Thioether (-S-)',         5.1,   8800,  12.0, 1),  # VK T7.2, T4.6
    ('[Cl]',          'Chloro (-Cl)',           14.2,  11550,  24.0, 1),  # VK T7.2 p.196, T4.6
    ('[F]',           'Fluoro (-F)',             1.0,   2300,   9.5, 1),  # VK T7.2 (low Yg = flexible)
    ('[Br]',          'Bromo (-Br)',            15.0,  15500,  30.0, 1),  # VK T7.2
    ('C=C',           'Vinyl (C=C)',             4.2,   4200,  12.5, 1),  # VK T4.6, Bic T5.3
]


@dataclass
class GroupContributionResult:
    """Result of Van Krevelen group contribution calculation."""
    success: bool
    polymer_smiles: str
    polymer_id: Optional[str] = None

    # Predicted properties
    tg: Optional[float] = None              # K
    tm: Optional[float] = None              # K
    density: Optional[float] = None         # g/cm3
    solubility_parameter: Optional[float] = None  # MPa^0.5
    ced: Optional[float] = None             # J/cm3
    molar_volume: Optional[float] = None    # cm3/mol

    # Group decomposition
    groups_found: Dict[str, int] = field(default_factory=dict)
    group_contributions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # RDKit descriptors on repeat unit
    repeat_unit_mw: Optional[float] = None
    repeat_unit_logp: Optional[float] = None
    repeat_unit_tpsa: Optional[float] = None
    repeat_unit_hbd: Optional[int] = None
    repeat_unit_hba: Optional[int] = None
    repeat_unit_rotatable_bonds: Optional[int] = None
    repeat_unit_aromatic_rings: Optional[int] = None

    # Backbone classification
    backbone_type: Optional[str] = None
    pendant_groups: List[str] = field(default_factory=list)
    chain_stiffness: Optional[str] = None

    # Structural corrections applied (for transparency)
    corrections_applied: List[str] = field(default_factory=list)

    # Per-property GC confidence (0.0-1.0)
    gc_confidence: Dict[str, float] = field(default_factory=dict)

    # Structural coverage of the group library over this repeat unit.
    # Reported with every prediction because accuracy depends strongly on it:
    # over the PolyMetriX collection the MAE is ~57 K where every heavy atom is
    # matched and ~226 K where a quarter or more is left unmatched. Without
    # this a partial decomposition returns an apparently precise number.
    atom_coverage: Optional[float] = None       # 0.0-1.0, heavy atoms matched
    n_unmatched_atoms: Optional[int] = None
    unmatched_elements: List[str] = field(default_factory=list)
    used_fallback: bool = False                 # Tg = 200 + 0.5M path

    prediction_engine: str = 'Group Contribution (Van Krevelen)'
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary for database storage."""
        result = {
            'success': self.success,
            'polymer_smiles': self.polymer_smiles,
            'polymer_id': self.polymer_id,
            'prediction_engine': self.prediction_engine,
            'tg_gc': round(self.tg, 1) if self.tg is not None else None,
            'tm_gc': round(self.tm, 1) if self.tm is not None else None,
            'density_gc': round(self.density, 4) if self.density is not None else None,
            'solubility_param_gc': round(self.solubility_parameter, 2) if self.solubility_parameter is not None else None,
            'ced_gc': round(self.ced, 1) if self.ced is not None else None,
            'molar_volume_gc': round(self.molar_volume, 2) if self.molar_volume is not None else None,
            'repeat_unit_mw': round(self.repeat_unit_mw, 2) if self.repeat_unit_mw is not None else None,
            'repeat_unit_logp': round(self.repeat_unit_logp, 2) if self.repeat_unit_logp is not None else None,
            'repeat_unit_tpsa': round(self.repeat_unit_tpsa, 1) if self.repeat_unit_tpsa is not None else None,
            'repeat_unit_hbd': self.repeat_unit_hbd,
            'repeat_unit_hba': self.repeat_unit_hba,
            'repeat_unit_rotatable_bonds': self.repeat_unit_rotatable_bonds,
            'repeat_unit_aromatic_rings': self.repeat_unit_aromatic_rings,
            'backbone_type': self.backbone_type,
            'pendant_groups': self.pendant_groups,
            'chain_stiffness': self.chain_stiffness,
            'groups_found': self.groups_found,
            'group_contributions': self.group_contributions,
            'corrections_applied': self.corrections_applied,
            'gc_confidence': self.gc_confidence,
            'gc_atom_coverage': (round(self.atom_coverage, 3)
                                 if self.atom_coverage is not None else None),
            'gc_n_unmatched_atoms': self.n_unmatched_atoms,
            'gc_unmatched_elements': self.unmatched_elements,
            'gc_used_fallback': self.used_fallback,
        }
        if self.warnings:
            result['prediction_warnings'] = '; '.join(self.warnings)
        return result


class GroupContributionCalculator:
    """
    Van Krevelen group contribution property calculator.

    Uses atom-exclusive matching: composite groups are matched first
    and their atoms are excluded from subsequent simpler group matching.
    """

    def __init__(self):
        # Compile and sort patterns by priority (high first)
        self._group_patterns = []
        for entry in GROUP_CONTRIBUTIONS:
            smarts, name, yg, ecoh, vw, priority = entry
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                self._group_patterns.append(
                    (pattern, name, yg, ecoh, vw, priority))
            else:
                logger.warning("Invalid SMARTS: %s (%s)", smarts, name)
        # Sort by priority descending (composite first)
        self._group_patterns.sort(key=lambda x: x[5], reverse=True)

    def calculate(self, polymer_smiles: str,
                  polymer_id: Optional[str] = None) -> GroupContributionResult:
        """Calculate polymer properties via Van Krevelen group contribution."""
        result = GroupContributionResult(
            success=False,
            polymer_smiles=polymer_smiles,
            polymer_id=polymer_id,
        )

        try:
            # 1. Parse PSMILES → RDKit molecule (methyl-capped)
            mol, n_caps = self._parse_psmiles(polymer_smiles)
            if mol is None:
                result.warnings.append("Could not parse PSMILES")
                return result

            # 2. RDKit descriptors on the capped molecule
            self._compute_rdkit_descriptors(mol, result)

            # 3. Correct MW: subtract methyl caps
            mw_capped = result.repeat_unit_mw or Descriptors.MolWt(mol)
            mw = mw_capped - n_caps * 15.035
            if mw <= 0:
                mw = mw_capped
            result.repeat_unit_mw = round(mw, 2)

            if mw <= 0:
                result.warnings.append("Invalid molecular weight")
                return result

            # 4. Atom-exclusive group matching
            groups = self._match_groups_exclusive(mol, n_caps)
            result.groups_found = {name: count for name, count, *_ in groups}

            # 4b. Structural coverage, reported with every prediction.
            self._compute_coverage(mol, result)

            # 5. Calculate properties
            self._calculate_properties(groups, mw, result)

            # 6. Classify structure
            self._classify_structure(mol, result)

            # 7. Apply structural Tg corrections (post-hoc)
            self._apply_structural_corrections(mol, result)

            # 8. Calculate Tm from corrected Tg (multi-level Boyer-Beaman)
            # Skip if Tm was already set by the fallback path (no groups matched)
            if result.tg is not None and groups:
                tm_ratio = self._estimate_tm_tg_ratio(groups, mol)
                result.tm = result.tg * tm_ratio

            # 9. Estimate per-property GC confidence
            self._estimate_gc_confidence(mol, result)

            # 9. Per-group breakdown
            self._build_group_contributions(groups, mw, result)

            result.success = True

        except Exception as e:
            logger.exception("Group contribution error for %s: %s",
                             polymer_smiles[:40], e)
            result.warnings.append(f"Calculation error: {e}")

        return result

    def _parse_psmiles(self, psmiles: str) -> Tuple[Optional[Chem.Mol], int]:
        """
        Convert PSMILES to RDKit Mol using methyl-cap strategy.

        Returns (mol, n_caps) where n_caps is the number of [*] replaced.
        """
        clean = psmiles.strip()
        n_caps = clean.count('[*]')
        if n_caps == 0:
            # Try bare * (handle both forms)
            n_caps = clean.count('*')
        clean = clean.replace('[*]', '[CH3]')
        clean = clean.replace('*', '[CH3]')

        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            mol = Chem.MolFromSmiles(clean, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return None, n_caps
        return mol, n_caps

    def _compute_rdkit_descriptors(self, mol: Chem.Mol,
                                    result: GroupContributionResult):
        """Compute RDKit descriptors on the repeat unit molecule."""
        try:
            result.repeat_unit_mw = Descriptors.MolWt(mol)
            result.repeat_unit_logp = Descriptors.MolLogP(mol)
            result.repeat_unit_tpsa = Descriptors.TPSA(mol)
            result.repeat_unit_hbd = Descriptors.NumHDonors(mol)
            result.repeat_unit_hba = Descriptors.NumHAcceptors(mol)
            result.repeat_unit_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            result.repeat_unit_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        except Exception as e:
            result.warnings.append(f"RDKit descriptor error: {e}")

    def _match_groups_exclusive(self, mol: Chem.Mol,
                                 n_caps: int) -> List[Tuple[str, int, float, float, float]]:
        """
        Atom-exclusive group matching.

        High-priority composite groups (ester, amide, etc.) are matched first
        and their atom indices are "consumed". Lower-priority groups only
        count matches on unconsumed atoms.
        """
        consumed_atoms: Set[int] = set()
        matched = []

        for pattern, name, yg, ecoh, vw, priority in self._group_patterns:
            # Sort matches by lowest atom index for cross-platform determinism
            # (RDKit GetSubstructMatches returns matches in arbitrary order)
            all_matches = sorted(mol.GetSubstructMatches(pattern),
                                 key=lambda m: m[0] if m else 0)
            count = 0

            for match_atoms in all_matches:
                atom_set = set(match_atoms)
                # Only count if none of these atoms already consumed
                if not atom_set & consumed_atoms:
                    count += 1
                    consumed_atoms.update(atom_set)

            # Subtract methyl caps
            if name == 'Methyl (-CH3)' and count > 0:
                count = max(0, count - n_caps)

            if count > 0:
                matched.append((name, count, yg, ecoh, vw))

        return matched

    def _compute_coverage(self, mol: Chem.Mol,
                          result: GroupContributionResult) -> None:
        """Record what fraction of the repeat unit the group library explains.

        Uses the same atom-exclusive priority walk as ``_match_groups_exclusive``
        so that the reported coverage corresponds exactly to the decomposition
        the prediction was built from.
        """
        consumed: Set[int] = set()
        for pattern, _name, *_rest in self._group_patterns:
            for match in sorted(mol.GetSubstructMatches(pattern),
                                key=lambda m: m[0] if m else 0):
                atoms = set(match)
                if not atoms & consumed:
                    consumed |= atoms

        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not heavy:
            return
        unmatched = [i for i in heavy if i not in consumed]
        result.atom_coverage = 1.0 - len(unmatched) / len(heavy)
        result.n_unmatched_atoms = len(unmatched)
        result.unmatched_elements = sorted(
            {mol.GetAtomWithIdx(i).GetSymbol() for i in unmatched})
        if unmatched:
            result.warnings.append(
                f"{len(unmatched)} of {len(heavy)} heavy atoms "
                f"({', '.join(result.unmatched_elements)}) are not covered by "
                f"the group library; treat this prediction with caution")

    def _calculate_properties(self, groups, mw, result: GroupContributionResult):
        """
        Apply Van Krevelen equations (VK 2009, Chapters 4, 6, 7).

        Equations (all per repeat unit of molar mass M):
            Tg (K) = (sum(Yg_i × n_i) / M) × 1000           [VK Eq.6.3]
            Vm (cm³/mol) = 1.3 × sum(Vw_i × n_i)            [VK Eq.4.7, amorphous]
            CED (J/cm³) = sum(Ecoh_i × n_i) / Vm             [VK Eq.7.3]
            delta (MPa^0.5) = sqrt(CED)                      [VK Eq.7.6]
            rho (g/cm³) = M / Vm                              [VK Eq.4.8]

        Fallback Tg (no groups matched) = 200 + 0.5×M:
            Empirical baseline from Bicerano (2002) Ch.5 for unrecognized
            structures; conservative estimate assuming moderate stiffness.
        """
        if not groups:
            result.warnings.append("No functional groups matched")
            result.used_fallback = True
            result.tg = 200.0 + mw * 0.5
            result.tm = result.tg * 1.5
            result.molar_volume = mw / 1.1
            result.density = mw / result.molar_volume if result.molar_volume > 0 else 1.0
            result.ced = 300.0
            result.solubility_parameter = math.sqrt(result.ced)
            return

        total_yg = sum(count * yg for _, count, yg, _, _ in groups)
        total_ecoh = sum(count * ecoh for _, count, _, ecoh, _ in groups)
        total_vw = sum(count * vw for _, count, _, _, vw in groups)

        if total_vw <= 0:
            total_vw = mw / 1.1
            result.warnings.append("Van der Waals volume fallback used")

        # Tg (K)
        result.tg = max(100.0, total_yg / mw * 1000.0)

        # Molar volume (cm³/mol) - amorphous packing: V ≈ 1.3 × Vw (Van Krevelen)
        result.molar_volume = 1.3 * total_vw

        # CED (J/cm³) - must use molar volume (not raw Vw) per Van Krevelen Ch.7
        result.ced = total_ecoh / result.molar_volume if result.molar_volume > 0 else 300.0

        # Solubility parameter (MPa^0.5)
        # 1 J/cm³ = 1 MPa, so delta = sqrt(CED in J/cm³) gives MPa^0.5
        result.solubility_parameter = math.sqrt(result.ced) if result.ced > 0 else 15.0

        # Density (g/cm³)
        result.density = mw / result.molar_volume if result.molar_volume > 0 else 1.0

        # Tm: deferred to after structural corrections (see calculate())

    def _estimate_tm_tg_ratio(self, groups, mol: Chem.Mol = None) -> float:
        """Estimate Tm/Tg ratio using multi-level Boyer-Beaman.

        Standard Boyer-Beaman uses binary symmetric (1.5) / asymmetric (2.0).
        This fails for many polymer classes. We use a tiered system:

        Tier A - Symmetric backbone, no pendants: ratio 2.0
            (PE, POM, PEO - high crystallizability gives high Tm)
        Tier B - Backbone linkages (ester, amide, ether + aromatic): ratio 1.5
            (PET, Nylon-6, PC - linkage groups moderate crystallizability)
        Tier C - Small polar pendants (OH, Cl, F): ratio 1.4
            (PVOH, PVC - small pendants allow close packing)
        Tier D - Bulky pendants (phenyl, ester, nitrile): ratio 2.0
            (PS, PMMA, PAN - bulky groups hinder crystallization, amorphous)
        Tier E - Perfluoro backbone: ratio 3.8
            (PTFE Tm/Tg=600/160=3.75 - strong helical crystallization)

        References:
            Van Krevelen & Te Nijenhuis (2009) Ch. 6, Table 6.5
            Wunderlich, Thermal Analysis of Polymeric Materials, 2005
        """
        # Special case: perfluoro polymers have anomalously high Tm/Tg
        if mol is not None:
            n_heavy = mol.GetNumHeavyAtoms()
            n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
            if n_heavy > 0 and n_f / n_heavy > 0.35:
                return 3.8  # PTFE: Tm/Tg ≈ 600/160

        group_names = {name for name, _, _, _, _ in groups}

        # Classify pendant groups
        backbone_only = {'Methylene (-CH2-)', 'Phenylene (p-C6H4)',
                         'Ether (-O-)', 'Ester (-COO-)',
                         'Amide (-CONH-)', 'Sulfone (-SO2-)',
                         'Dimethylsiloxane'}
        small_polar = {'Hydroxyl (-OH)', 'Chloro (-Cl)', 'Fluoro (-F)'}
        bulky_pendant = {'Methyl (-CH3)', 'Nitrile (-CN)',
                         'Cyclohexyl', 'Cyclopentyl'}

        pendant_names = group_names - backbone_only
        has_small_polar = bool(pendant_names & small_polar)
        has_bulky = bool(pendant_names & bulky_pendant)
        has_aromatic_pendant = bool(group_names & {'Phenylene (p-C6H4)'}) and has_bulky

        # Polar backbone linkages that moderate Tm/Tg ratio
        polar_linkages = {'Ester (-COO-)', 'Amide (-CONH-)', 'Sulfone (-SO2-)'}

        if not pendant_names:
            if group_names & polar_linkages:
                # Tier A2: polar-linked backbone (Nylon-6, PET)
                # Polar linkages moderate crystallization → Tm/Tg ≈ 1.5
                return 1.5
            else:
                # Tier A1: simple backbone (PE, POM, PEO)
                return 2.0
        elif has_bulky or has_aromatic_pendant:
            # Tier D: bulky pendants (PS, PMMA, PAN)
            return 2.0
        elif has_small_polar and not has_bulky:
            # Tier C: small polar pendants (PVOH, PVC)
            return 1.4
        else:
            # Tier B: backbone linkages dominate
            return 1.5

    def _classify_structure(self, mol: Chem.Mol, result: GroupContributionResult):
        """Classify polymer backbone type and chain stiffness."""
        n_aromatic = result.repeat_unit_aromatic_rings or 0
        n_rotatable = result.repeat_unit_rotatable_bonds or 0
        n_atoms = mol.GetNumHeavyAtoms()

        has_aromatic = n_aromatic > 0
        has_ester = mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[OD2]'))
        has_amide = mol.HasSubstructMatch(Chem.MolFromSmarts('[C](=O)[NH]'))
        has_ether = mol.HasSubstructMatch(Chem.MolFromSmarts('[OD2;!$(O=*)]'))
        has_siloxane = mol.HasSubstructMatch(Chem.MolFromSmarts('[Si]'))

        if has_siloxane:
            result.backbone_type = 'Siloxane'
        elif has_amide:
            result.backbone_type = 'Polyamide'
        elif has_ester and has_aromatic:
            result.backbone_type = 'Aromatic Polyester'
        elif has_ester:
            result.backbone_type = 'Aliphatic Polyester'
        elif has_aromatic and has_ether:
            result.backbone_type = 'Polyarylene Ether'
        elif has_aromatic:
            result.backbone_type = 'Aromatic Backbone'
        elif has_ether:
            result.backbone_type = 'Polyether'
        else:
            result.backbone_type = 'Polyolefin'

        # Chain stiffness
        if n_atoms > 0:
            aromatic_fraction = n_aromatic / max(1, n_atoms / 6)
            rotatable_fraction = n_rotatable / max(1, n_atoms)
            if aromatic_fraction > 0.5 or n_rotatable == 0:
                result.chain_stiffness = 'Rigid'
            elif aromatic_fraction > 0.2 or rotatable_fraction < 0.2:
                result.chain_stiffness = 'Moderate'
            else:
                result.chain_stiffness = 'Flexible'
        else:
            result.chain_stiffness = 'Unknown'

        # Pendant groups
        pendants = []
        pendant_patterns = [
            (Chem.MolFromSmarts('[CH3]'), 'Methyl'),
            (Chem.MolFromSmarts('c1ccccc1'), 'Phenyl'),
            (Chem.MolFromSmarts('[OH]'), 'Hydroxyl'),
            (Chem.MolFromSmarts('C#N'), 'Nitrile'),
            (Chem.MolFromSmarts('[Cl]'), 'Chloro'),
            (Chem.MolFromSmarts('[F]'), 'Fluoro'),
            (Chem.MolFromSmarts('C(=O)O'), 'Ester'),
        ]
        for pat, name in pendant_patterns:
            if pat and mol.HasSubstructMatch(pat):
                pendants.append(name)
        result.pendant_groups = pendants

    def _apply_structural_corrections(self, mol: Chem.Mol,
                                       result: GroupContributionResult):
        """Apply post-hoc corrections for known GC failure modes.

        These correct systematic errors where group additivity breaks down
        due to conformational, steric, or inter-chain interaction effects.

        References:
            Van Krevelen & Te Nijenhuis (2009) Ch. 6 (fluoropolymers)
            Bicerano (2002) Ch. 5 (steric effects on Tg)
        """
        if result.tg is None:
            return

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy == 0:
            return

        # Count fluorine atoms for fluoropolymer detection
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        f_fraction = n_f / n_heavy if n_heavy > 0 else 0
        is_fluoropolymer = f_fraction > 0.30

        # ── Correction 1: Perfluoro backbone (helical conformation) ──
        # PTFE's 13/6 helix has low-energy crankshaft motion → anomalously low Tg.
        # Detect via fluorine atom fraction (not SMARTS - avoids degree mismatch).
        if is_fluoropolymer and f_fraction > 0.35:
            # Scale correction by fluorination degree:
            # f_fraction=0.5 (PTFE) → factor 0.75, f_fraction=0.35 → factor ~0.93
            factor = 1.0 - 0.5 * (f_fraction - 0.35) / 0.15
            factor = max(0.70, min(1.0, factor))
            old_tg = result.tg
            result.tg = result.tg * factor
            result.corrections_applied.append(
                f"Perfluoro Tg correction: {old_tg:.0f}→{result.tg:.0f} K "
                f"(F fraction={f_fraction:.2f})")

        # ── Correction 2: Alpha-methyl steric effect ──
        # Quaternary backbone C with BULKY pendant restricts rotation → higher Tg.
        # Only applies when quaternary C has carbonyl or aromatic neighbor.
        # Skipped for fluoropolymers (where quaternary C is CF2, not steric).
        if not is_fluoropolymer:
            # Pattern A: alpha-methyl + carbonyl pendant (PMMA-like)
            pat_a = Chem.MolFromSmarts('[CH0;A]([CH3])([CX3]=O)')
            # Pattern B: alpha-methyl + aromatic pendant (poly-alpha-methylstyrene)
            pat_b = Chem.MolFromSmarts('[CH0;A]([CH3])(c)')
            # Count DISTINCT quaternary C atoms (not matches - avoids
            # double-counting from cap methyls bonded to same atom)
            quat_atoms = set()
            if pat_a:
                for m in mol.GetSubstructMatches(pat_a):
                    quat_atoms.add(m[0])  # m[0] is the [CH0] atom idx
            if pat_b:
                for m in mol.GetSubstructMatches(pat_b):
                    quat_atoms.add(m[0])
            quat_count = len(quat_atoms)

            if quat_count >= 1:
                # +25% Tg per alpha-methyl quaternary C (Bicerano Ch. 5)
                # Capped at 1.5x to avoid runaway for multi-substituted
                factor = min(1.50, 1.0 + 0.25 * quat_count)
                old_tg = result.tg
                result.tg = result.tg * factor
                result.corrections_applied.append(
                    f"Alpha-methyl steric correction: {old_tg:.0f}→{result.tg:.0f} K "
                    f"(n_quat={quat_count})")

        # Note: Tm is calculated AFTER this method returns,
        # using the corrected Tg. See calculate() step 8.

    def _estimate_gc_confidence(self, mol: Chem.Mol,
                                result: GroupContributionResult):
        """Estimate per-property GC reliability.

        Based on known limitations of additive group contribution:
        - Tg: good for standard polymers, lower for fluorinated/unusual
        - Tm: always lower (Boyer-Beaman is a rough approximation)
        - Density: moderate (Vw from mixed literature sources)
        - Solubility parameter: moderate (depends on CED accuracy)
        """
        conf = {}

        # Base Tg confidence: high for most polymers
        tg_conf = 0.80
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > 0 and n_f / n_heavy > 0.3:
            tg_conf -= 0.20  # Fluoropolymers: GC less reliable even after correction
        if result.groups_found and sum(result.groups_found.values()) <= 2:
            tg_conf -= 0.10  # Very few groups matched → higher uncertainty
        conf['tg'] = round(max(0.1, tg_conf), 2)

        # Tm confidence: always lower than Tg (Boyer-Beaman is crude)
        conf['tm'] = round(max(0.1, conf['tg'] - 0.30), 2)

        # Density confidence: moderate (Vw values from mixed sources)
        conf['density'] = 0.55

        # Solubility parameter: moderate (CED depends on Ecoh accuracy)
        conf['solubility_parameter'] = 0.60

        # CED: same as solubility parameter
        conf['ced'] = 0.60

        result.gc_confidence = conf

    def _build_group_contributions(self, groups, mw,
                                    result: GroupContributionResult):
        """Build per-group property contribution breakdown."""
        contributions = {}
        for name, count, yg, ecoh, vw in groups:
            contributions[name] = {
                'count': count,
                'tg_contribution': round(count * yg, 1),
                'ecoh_contribution': round(count * ecoh, 1),
                'vw_contribution': round(count * vw, 1),
            }
        result.group_contributions = contributions

    def validate_calibration(self, tolerance_k: float = 30.0) -> Dict:
        """Verify GC predictions against known experimental Tg values.

        Runs predictions for canonical polymers and compares against
        literature Tg values (Polymer Handbook, 4th Ed.).
        Tolerance default = 30 K (typical GC accuracy).

        Returns dict with 'passed', 'results', and any 'failures'.
        """
        # (PSMILES, name, experimental Tg in K, source)
        validation_set = [
            ('[*]CC[*]',                       'PE',    195, 'PH VII/1'),
            ('[*]CC(C)[*]',                    'PP',    253, 'PH VII/7'),
            ('[*]CC(c1ccccc1)[*]',             'PS',    373, 'PH VII/15'),
            ('[*]CC(Cl)[*]',                   'PVC',   354, 'PH VII/11'),
            ('[*]CCOC(=O)c1ccc(C(=O)O[*])cc1', 'PET',   342, 'VK T6.1'),
            ('[*]CCO[*]',                      'PEO',   206, 'PH VII/61'),
            ('[*]CO[*]',                       'POM',   198, 'PH VII/60'),
            ('[*]CC(O)[*]',                    'PVOH',  358, 'PH VII/19'),
        ]

        results = []
        failures = []
        for psmiles, name, exp_tg, source in validation_set:
            try:
                pred = self.calculate(psmiles)
                pred_tg = pred.tg
                error = abs(pred_tg - exp_tg) if pred_tg else float('inf')
                passed = error <= tolerance_k
                entry = {
                    'name': name,
                    'psmiles': psmiles,
                    'experimental_tg': exp_tg,
                    'predicted_tg': round(pred_tg, 1) if pred_tg else None,
                    'error_k': round(error, 1),
                    'tolerance_k': tolerance_k,
                    'passed': passed,
                    'source': source,
                }
                results.append(entry)
                if not passed:
                    failures.append(entry)
            except Exception as e:
                failures.append({
                    'name': name, 'error': str(e), 'passed': False})
                results.append({
                    'name': name, 'error': str(e), 'passed': False})

        n_passed = sum(1 for r in results if r.get('passed'))
        return {
            'passed': len(failures) == 0,
            'n_passed': n_passed,
            'n_total': len(validation_set),
            'tolerance_k': tolerance_k,
            'results': results,
            'failures': failures,
        }
