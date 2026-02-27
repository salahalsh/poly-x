"""
Van Krevelen Group Contribution for Polymer Properties (Tier 1)

Predicts: Tg, Tm, density, solubility parameter, CED, molar volume
from chemical structure of repeat unit using group additivity.

Uses atom-exclusive matching: composite groups (ester, amide, etc.)
are matched first and their atoms excluded from simpler groups.

References:
    Van Krevelen & Te Nijenhuis, Properties of Polymers, 4th Ed., 2009
    Bicerano, Prediction of Polymer Properties, 3rd Ed., 2002

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.

Usage:
    python -m poly_x.group_contribution "[*]CC(c1ccccc1)[*]"
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ======================================================================
# Van Krevelen Group Contribution Table
#
# Each entry: (SMARTS, name, Yg, Ecoh, Vw, priority)
#   Yg   = molar Tg contribution (K*g/mol / 1000 for our formula)
#   Ecoh = molar cohesive energy (J/mol)   -- VK Table 7.2
#   Vw   = van der Waals volume (cm3/mol)  -- VK Table 4.6
#   priority = matching order (higher = matched first, atoms consumed)
#
# Yg calibrated against experimental Tg (Polymer Handbook, 4th Ed.):
#   PE=195K, PP=253K, PS=373K, PVC=354K, PET=342K, PVOH=358K,
#   Nylon-6=323K, PEO=206K, POM=198K, PDMS=150K
#
# Key references (abbreviated as VK, Bic, PH):
#   VK  = Van Krevelen & Te Nijenhuis (2009)
#   Bic = Bicerano (2002)
#   PH  = Brandrup, Immergut & Grulke (1999) Polymer Handbook, 4th Ed.
# ======================================================================

# Priority 3: Composite groups (matched first, consume all their atoms)
# Priority 2: Ring systems
# Priority 1: Simple groups (matched last, only on unconsumed atoms)

GROUP_CONTRIBUTIONS = [
    # -- Priority 3: Composite groups (multi-atom, matched first) --
    # SMARTS            Name                      Yg     Ecoh    Vw   Pri
    ('[C](=O)[NH]',  'Amide (-CONH-)',         23.0,  33500,  19.5, 3),
    ('[C](=O)[OD2]', 'Ester (-COO-)',          14.0,  18000,  18.0, 3),
    ('S(=O)(=O)',     'Sulfone (-SO2-)',        25.8,  23400,  19.6, 3),
    ('[Si](C)(C)O',  'Dimethylsiloxane',        9.0,   6000,  55.0, 3),
    ('C#N',           'Nitrile (-CN)',          25.0,  25500,  24.0, 3),
    ('[N+](=O)[O-]',  'Nitro (-NO2)',           6.0,  11500,  24.0, 3),

    # -- Priority 2: Ring systems --
    ('c1ccccc1',      'Phenylene (p-C6H4)',    31.0,  31940,  52.4, 2),
    ('c1ccncc1',      'Pyridine ring',         33.0,  33400,  48.0, 2),
    ('c1ccoc1',       'Furan ring',            28.0,  28000,  42.0, 2),
    ('C1CCCCC1',      'Cyclohexyl',            26.0,  26000,  68.0, 2),
    ('C1CCCC1',       'Cyclopentyl',           22.0,  22000,  55.0, 2),

    # -- Priority 1: Simple groups (on remaining atoms) --
    ('[CH3]',         'Methyl (-CH3)',           2.4,   4710,  33.5, 1),
    ('[CH2]',         'Methylene (-CH2-)',       2.7,   4940,  16.1, 1),
    ('[CH1;A]',       'Methine (-CH<)',          5.5,   3430,   6.5, 1),
    ('[CH0;A]',       'Quaternary C (-C<)',      8.7,   1470,  -1.0, 1),
    ('[OD2;!$(O=*)]', 'Ether (-O-)',            3.5,   3350,   3.8, 1),
    ('[C;A](=O)',     'Carbonyl (C=O)',          8.0,  17370,  10.8, 1),
    ('[OH]',          'Hydroxyl (-OH)',          7.6,  29800,  10.0, 1),
    ('[NH]',          'Secondary amine (-NH-)',  8.4,   8400,   4.5, 1),
    ('[NH2]',         'Primary amine (-NH2)',    9.0,  12600,   9.0, 1),
    ('[SD2]',         'Thioether (-S-)',         5.1,   8800,  12.0, 1),
    ('[Cl]',          'Chloro (-Cl)',           14.2,  11550,  24.0, 1),
    ('[F]',           'Fluoro (-F)',             1.0,   2300,   9.5, 1),
    ('[Br]',          'Bromo (-Br)',            15.0,  15500,  30.0, 1),
    ('C=C',           'Vinyl (C=C)',             4.2,   4200,  12.5, 1),
]


@dataclass
class GroupContributionResult:
    """Result of Van Krevelen group contribution calculation."""
    success: bool
    polymer_smiles: str

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

    # Backbone classification
    backbone_type: Optional[str] = None
    chain_stiffness: Optional[str] = None

    # Structural corrections applied
    corrections_applied: List[str] = field(default_factory=list)

    # Per-property GC confidence (0.0-1.0)
    gc_confidence: Dict[str, float] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)


class GroupContributionCalculator:
    """
    Van Krevelen group contribution property calculator.

    Uses atom-exclusive matching: composite groups are matched first
    and their atoms are excluded from subsequent simpler group matching.
    """

    def __init__(self):
        self._group_patterns = []
        for entry in GROUP_CONTRIBUTIONS:
            smarts, name, yg, ecoh, vw, priority = entry
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                self._group_patterns.append(
                    (pattern, name, yg, ecoh, vw, priority))
            else:
                logger.warning("Invalid SMARTS: %s (%s)", smarts, name)
        self._group_patterns.sort(key=lambda x: x[5], reverse=True)

    def calculate(self, polymer_smiles: str) -> GroupContributionResult:
        """Calculate polymer properties via Van Krevelen group contribution."""
        result = GroupContributionResult(
            success=False,
            polymer_smiles=polymer_smiles,
        )

        try:
            mol, n_caps = self._parse_psmiles(polymer_smiles)
            if mol is None:
                result.warnings.append("Could not parse PSMILES")
                return result

            mw_capped = Descriptors.MolWt(mol)
            mw = mw_capped - n_caps * 15.035
            if mw <= 0:
                mw = mw_capped
            result.repeat_unit_mw = round(mw, 2)

            groups = self._match_groups_exclusive(mol, n_caps)
            result.groups_found = {name: count for name, count, *_ in groups}

            self._calculate_properties(groups, mw, result)
            self._classify_structure(mol, result)
            self._apply_structural_corrections(mol, result)

            if result.tg is not None and groups:
                tm_ratio = self._estimate_tm_tg_ratio(groups, mol)
                result.tm = result.tg * tm_ratio

            self._estimate_gc_confidence(mol, result)
            self._build_group_contributions(groups, mw, result)

            result.success = True

        except Exception as e:
            logger.exception("Group contribution error for %s: %s",
                             polymer_smiles[:40], e)
            result.warnings.append(f"Calculation error: {e}")

        return result

    def _parse_psmiles(self, psmiles: str) -> Tuple[Optional[Chem.Mol], int]:
        """Convert PSMILES to RDKit Mol using methyl-cap strategy."""
        clean = psmiles.strip()
        n_caps = clean.count('[*]')
        if n_caps == 0:
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

    def _match_groups_exclusive(self, mol: Chem.Mol,
                                 n_caps: int) -> List[Tuple[str, int, float, float, float]]:
        """Atom-exclusive group matching."""
        consumed_atoms: Set[int] = set()
        matched = []

        for pattern, name, yg, ecoh, vw, priority in self._group_patterns:
            all_matches = sorted(mol.GetSubstructMatches(pattern),
                                 key=lambda m: m[0] if m else 0)
            count = 0
            for match_atoms in all_matches:
                atom_set = set(match_atoms)
                if not atom_set & consumed_atoms:
                    count += 1
                    consumed_atoms.update(atom_set)

            if name == 'Methyl (-CH3)' and count > 0:
                count = max(0, count - n_caps)

            if count > 0:
                matched.append((name, count, yg, ecoh, vw))

        return matched

    def _calculate_properties(self, groups, mw, result: GroupContributionResult):
        """Apply Van Krevelen equations (VK 2009, Chapters 4, 6, 7)."""
        if not groups:
            result.warnings.append(
                "No functional groups matched -- using empirical fallback.")
            result.tg = 200.0 + mw * 0.5
            result.tm = result.tg * 1.5
            result.molar_volume = mw / 1.1
            result.density = mw / result.molar_volume if result.molar_volume > 0 else 1.0
            result.ced = 300.0
            result.solubility_parameter = math.sqrt(result.ced)
            result.corrections_applied.append("Fallback Tg formula (no groups matched)")
            return

        total_yg = sum(count * yg for _, count, yg, _, _ in groups)
        total_ecoh = sum(count * ecoh for _, count, _, ecoh, _ in groups)
        total_vw = sum(count * vw for _, count, _, _, vw in groups)

        if total_vw <= 0:
            total_vw = mw / 1.1
            result.warnings.append("Van der Waals volume fallback used")

        result.tg = max(100.0, total_yg / mw * 1000.0)
        result.molar_volume = 1.3 * total_vw
        result.ced = total_ecoh / result.molar_volume if result.molar_volume > 0 else 300.0
        result.solubility_parameter = math.sqrt(result.ced) if result.ced > 0 else 15.0
        result.density = mw / result.molar_volume if result.molar_volume > 0 else 1.0

    def _estimate_tm_tg_ratio(self, groups, mol: Chem.Mol = None) -> float:
        """Estimate Tm/Tg ratio using multi-level Boyer-Beaman."""
        if mol is not None:
            n_heavy = mol.GetNumHeavyAtoms()
            n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
            if n_heavy > 0 and n_f / n_heavy > 0.35:
                return 3.8  # PTFE: Tm/Tg ~ 600/160

        group_names = {name for name, _, _, _, _ in groups}

        backbone_only = {'Methylene (-CH2-)', 'Phenylene (p-C6H4)',
                         'Ether (-O-)', 'Ester (-COO-)',
                         'Amide (-CONH-)', 'Sulfone (-SO2-)',
                         'Dimethylsiloxane'}
        small_polar = {'Hydroxyl (-OH)', 'Chloro (-Cl)', 'Fluoro (-F)'}
        bulky_pendant = {'Methyl (-CH3)', 'Nitrile (-CN)',
                         'Cyclohexyl', 'Cyclopentyl'}
        polar_linkages = {'Ester (-COO-)', 'Amide (-CONH-)', 'Sulfone (-SO2-)'}

        pendant_names = group_names - backbone_only
        has_small_polar = bool(pendant_names & small_polar)
        has_bulky = bool(pendant_names & bulky_pendant)
        has_aromatic_pendant = bool(group_names & {'Phenylene (p-C6H4)'}) and has_bulky

        if not pendant_names:
            if group_names & polar_linkages:
                return 1.5
            else:
                return 2.0
        elif has_bulky or has_aromatic_pendant:
            return 2.0
        elif has_small_polar and not has_bulky:
            return 1.4
        else:
            return 1.5

    def _classify_structure(self, mol: Chem.Mol, result: GroupContributionResult):
        """Classify polymer backbone type and chain stiffness."""
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_rotatable = Descriptors.NumRotatableBonds(mol)
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

    def _apply_structural_corrections(self, mol: Chem.Mol,
                                       result: GroupContributionResult):
        """Apply post-hoc corrections for known GC failure modes."""
        if result.tg is None:
            return

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy == 0:
            return

        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        f_fraction = n_f / n_heavy if n_heavy > 0 else 0
        is_fluoropolymer = f_fraction > 0.30

        # Correction 1: Perfluoro backbone (helical conformation)
        if is_fluoropolymer and f_fraction > 0.30:
            factor = 1.0 - 0.25 * (f_fraction - 0.30) / 0.20
            factor = max(0.70, min(1.0, factor))
            old_tg = result.tg
            result.tg = result.tg * factor
            result.corrections_applied.append(
                f"Perfluoro Tg correction: {old_tg:.0f}->{result.tg:.0f} K "
                f"(F fraction={f_fraction:.2f})")

        # Correction 2: Alpha-methyl steric effect
        if not is_fluoropolymer:
            pat_a = Chem.MolFromSmarts('[CH0;A]([CH3])([CX3]=O)')
            pat_b = Chem.MolFromSmarts('[CH0;A]([CH3])(c)')
            quat_atoms = set()
            if pat_a:
                for m in mol.GetSubstructMatches(pat_a):
                    quat_atoms.add(m[0])
            if pat_b:
                for m in mol.GetSubstructMatches(pat_b):
                    quat_atoms.add(m[0])
            quat_count = len(quat_atoms)

            if quat_count >= 1:
                factor = min(1.50, 1.0 + 0.25 * quat_count)
                old_tg = result.tg
                result.tg = result.tg * factor
                result.corrections_applied.append(
                    f"Alpha-methyl steric correction: {old_tg:.0f}->{result.tg:.0f} K "
                    f"(n_quat={quat_count})")

    def _estimate_gc_confidence(self, mol: Chem.Mol,
                                result: GroupContributionResult):
        """Estimate per-property GC reliability."""
        conf = {}
        tg_conf = 0.80
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > 0 and n_f / n_heavy > 0.3:
            tg_conf -= 0.20
        if result.groups_found and sum(result.groups_found.values()) <= 2:
            tg_conf -= 0.10
        conf['tg'] = round(max(0.1, tg_conf), 2)
        conf['tm'] = round(max(0.1, conf['tg'] - 0.30), 2)
        conf['density'] = 0.55
        conf['solubility_parameter'] = 0.60
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
        """Verify GC predictions against known experimental Tg values."""
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
                    'passed': passed,
                    'source': source,
                }
                results.append(entry)
                if not passed:
                    failures.append(entry)
            except Exception as e:
                failures.append({'name': name, 'error': str(e), 'passed': False})
                results.append({'name': name, 'error': str(e), 'passed': False})

        n_passed = sum(1 for r in results if r.get('passed'))
        return {
            'passed': len(failures) == 0,
            'n_passed': n_passed,
            'n_total': len(validation_set),
            'results': results,
        }


def main():
    """CLI entry point for group contribution prediction."""
    if len(sys.argv) < 2:
        print("Usage: python -m poly_x.group_contribution <PSMILES>")
        print("Example: python -m poly_x.group_contribution \"[*]CC(c1ccccc1)[*]\"")
        sys.exit(1)

    psmiles = sys.argv[1]
    calc = GroupContributionCalculator()
    result = calc.calculate(psmiles)

    if result.success:
        print(f"\nPSMILES: {result.polymer_smiles}")
        print(f"Backbone: {result.backbone_type} ({result.chain_stiffness})")
        print(f"Repeat unit MW: {result.repeat_unit_mw:.2f} g/mol")
        print(f"\n--- Predicted Properties ---")
        print(f"  Tg:  {result.tg:.1f} K ({result.tg - 273.15:.1f} C)")
        if result.tm:
            print(f"  Tm:  {result.tm:.1f} K ({result.tm - 273.15:.1f} C)")
        if result.density:
            print(f"  Density: {result.density:.3f} g/cm3")
        if result.solubility_parameter:
            print(f"  Solubility parameter: {result.solubility_parameter:.2f} MPa^0.5")
        if result.ced:
            print(f"  CED: {result.ced:.1f} J/cm3")

        print(f"\n--- Group Decomposition ---")
        for name, count in result.groups_found.items():
            print(f"  {name}: {count}x")

        if result.corrections_applied:
            print(f"\n--- Corrections ---")
            for c in result.corrections_applied:
                print(f"  {c}")

        print(f"\n--- GC Confidence ---")
        for prop, conf in result.gc_confidence.items():
            print(f"  {prop}: {conf:.2f}")
    else:
        print(f"Prediction failed: {'; '.join(result.warnings)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
