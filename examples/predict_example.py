"""
POLY-X Quick Start Example: Group Contribution Predictions

Demonstrates Tier 1 (Van Krevelen) property prediction for common polymers.

Usage:
    python examples/predict_example.py

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Platform for
    Polymer Property Prediction. Journal of
    Computer-Aided Molecular Design, 2026.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poly_x.services.group_contribution import GroupContributionCalculator


def main():
    calc = GroupContributionCalculator()

    # Test polymers: (PSMILES, name, experimental Tg in K)
    polymers = [
        ('[*]CC(c1ccccc1)[*]',              'Polystyrene (PS)',         373),
        ('[*]CCOC(=O)c1ccc(C(=O)O[*])cc1',  'Poly(ethylene terephthalate) (PET)', 342),
        ('[*]CC(C)(C(=O)OC)[*]',             'Poly(methyl methacrylate) (PMMA)',  378),
        ('[*]CC[*]',                         'Polyethylene (PE)',        195),
        ('[*]CC(C)[*]',                      'Polypropylene (PP)',       253),
        ('[*]CC(O)[*]',                      'Poly(vinyl alcohol) (PVOH)', 358),
    ]

    print("=" * 70)
    print("POLY-X Tier 1: Van Krevelen Group Contribution Predictions")
    print("=" * 70)

    for psmiles, name, exp_tg in polymers:
        result = calc.calculate(psmiles)
        if result.success:
            error = abs(result.tg - exp_tg)
            print(f"\n{name}")
            print(f"  PSMILES:  {psmiles}")
            print(f"  Tg:       {result.tg:.1f} K  (exp: {exp_tg} K, error: {error:.1f} K)")
            if result.tm:
                print(f"  Tm:       {result.tm:.1f} K")
            if result.density:
                print(f"  Density:  {result.density:.3f} g/cm3")
            print(f"  Backbone: {result.backbone_type} ({result.chain_stiffness})")
            print(f"  Groups:   {result.groups_found}")
            if result.corrections_applied:
                print(f"  Corrections: {result.corrections_applied}")
        else:
            print(f"\n{name}: FAILED - {result.warnings}")

    # Run validation
    print("\n" + "=" * 70)
    print("Validation against 8 canonical homopolymers:")
    print("=" * 70)
    val = calc.validate_calibration(tolerance_k=30.0)
    print(f"Passed: {val['n_passed']}/{val['n_total']} (tolerance: 30 K)")
    for r in val['results']:
        status = 'PASS' if r.get('passed') else 'FAIL'
        print(f"  [{status}] {r['name']}: pred={r.get('predicted_tg')} K, "
              f"exp={r.get('experimental_tg')} K, error={r.get('error_k')} K")


if __name__ == '__main__':
    main()
