"""
Script 04 - Tier 1 group contribution evaluated at scale, with coverage.

Answers:
  * Reviewer 1 and Reviewer 2, Major 5 - the eight-polymer MAE of 3.9 K is not
    an independent validation. Here the frozen Tier 1 parameters are applied to
    every polymer in the dataset (none of which was used to fit them), and
    performance is reported by polymer family, by functional-group coverage,
    and by distance from the calibration domain, together with the failures.
  * Reviewer 2, Minor 4 and Minor 5 - atom-level coverage and the behaviour on
    unsupported structures are now reported for every query. The submitted
    implementation silently returns an apparently precise number for a repeat
    unit whose atoms are almost entirely unmatched, and falls back to the
    empirical Tg = 200 + 0.5M when nothing matches at all.
  * Reviewer 2, Minor 6 - which of the eight canonical polymers occur in the
    training data, and in which partition.

Writes: outputs/s04_gc_predictions.csv
        outputs/s04_gc_summary.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# Import the service module by file path so that no Django app registry is
# required: the calculator itself has no Django dependency.
import importlib.util  # noqa: E402
from _paths import SERVICES  # noqa: E402

_GC = SERVICES / "group_contribution.py"
_spec = importlib.util.spec_from_file_location("polyx_group_contribution", _GC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GroupContributionCalculator = _mod.GroupContributionCalculator

from polyx_rev import OUT, dump, load_dataset, to_mol  # noqa: E402
from s01_dataset_and_splits import CANONICAL, family  # noqa: E402

# Experimental Tg used in Table 1 of the submitted manuscript.
CANONICAL_EXP = {'PE': 195, 'PP': 253, 'PS': 373, 'PVC': 354, 'PET': 342,
                 'PEO': 206, 'POM': 198, 'PVOH': 358}


def atom_coverage(calc, psmiles):
    """Fraction of heavy atoms of the (methyl-capped) repeat unit that are
    consumed by a matched functional group, plus the unmatched elements."""
    mol, n_caps = calc._parse_psmiles(psmiles)
    if mol is None:
        return None, None, None
    consumed = set()
    for pattern, name, *_ in calc._group_patterns:
        for match in sorted(mol.GetSubstructMatches(pattern),
                            key=lambda m: m[0] if m else 0):
            s = set(match)
            if not s & consumed:
                consumed |= s
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return None, None, None
    unmatched = [i for i in heavy if i not in consumed]
    frac = 1.0 - len(unmatched) / len(heavy)
    elems = sorted({mol.GetAtomWithIdx(i).GetSymbol() for i in unmatched})
    return float(frac), len(unmatched), elems


def main():
    calc = GroupContributionCalculator()
    df, _ = load_dataset()
    df = df.copy()
    df['family'] = df['smiles'].apply(family)

    recs = []
    for i, (smi, y, fam) in enumerate(zip(df['smiles'], df['target'], df['family'])):
        r = calc.calculate(smi)
        cov, n_unmatched, elems = atom_coverage(calc, smi)
        recs.append({
            'smiles': smi,
            'exp_tg': float(y),
            'family': fam,
            'success': bool(r.success),
            'pred_tg': float(r.tg) if r.tg is not None else np.nan,
            'n_groups': int(sum(r.groups_found.values())) if r.groups_found else 0,
            'n_group_types': len(r.groups_found or {}),
            'atom_coverage': cov if cov is not None else np.nan,
            'n_unmatched_atoms': n_unmatched if n_unmatched is not None else -1,
            'unmatched_elements': '|'.join(elems) if elems else '',
            'fallback_used': bool(r.warnings and
                                  any('No functional groups matched' in w
                                      for w in r.warnings)),
            'corrections': '|'.join(r.corrections_applied or []),
            'mw': float(r.repeat_unit_mw) if r.repeat_unit_mw else np.nan,
        })
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(df)}', flush=True)

    out = pd.DataFrame(recs)
    out['abs_error'] = (out['pred_tg'] - out['exp_tg']).abs()
    out['signed_error'] = out['pred_tg'] - out['exp_tg']
    out.to_csv(OUT / 's04_gc_predictions.csv', index=False)
    print(f'[write] {OUT / "s04_gc_predictions.csv"}')

    ok = out[out['success'] & out['pred_tg'].notna()]

    def stats(frame):
        if len(frame) == 0:
            return None
        e = frame['abs_error'].values
        se = frame['signed_error'].values
        from sklearn.metrics import r2_score
        return {
            'n': int(len(frame)),
            'mae': float(np.mean(e)),
            'median_ae': float(np.median(e)),
            'rmse': float(np.sqrt(np.mean(se ** 2))),
            'bias': float(np.mean(se)),
            'r2': float(r2_score(frame['exp_tg'], frame['pred_tg']))
            if len(frame) > 2 else None,
            'pct_within_10K': float((e <= 10).mean() * 100),
            'pct_within_25K': float((e <= 25).mean() * 100),
            'pct_within_50K': float((e <= 50).mean() * 100),
        }

    cov_bins = [(-0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]
    by_cov = {}
    for lo, hi in cov_bins:
        sel = ok[(ok['atom_coverage'] > lo) & (ok['atom_coverage'] <= hi)]
        by_cov[f'{max(lo,0):.2f}-{min(hi,1):.2f}'] = stats(sel)

    # The fine bins below 0.75 hold very few polymers (as few as one), so the
    # headline statement uses three bands with adequate support. Note that the
    # relationship is NOT monotone across the fine bins: error is flat and high
    # below 0.75 coverage and then falls sharply. Reporting it as monotone
    # would overstate the result.
    bands = {
        'below_0.75': ok[ok['atom_coverage'] <= 0.75],
        '0.75_to_below_1.00': ok[(ok['atom_coverage'] > 0.75)
                                 & (ok['atom_coverage'] < 0.999)],
        'full_coverage': ok[ok['atom_coverage'] >= 0.999],
    }
    by_band = {k: stats(v) for k, v in bands.items()}
    fine = [v['mae'] for v in by_cov.values() if v]
    by_band['_monotone_across_fine_bins'] = bool(
        all(a >= b for a, b in zip(fine, fine[1:])))

    summary = {
        'overall_all_predictions': stats(ok),
        'overall_excluding_fallback': stats(ok[~ok['fallback_used']]),
        'fallback_only': stats(ok[ok['fallback_used']]),
        'n_fallback': int(out['fallback_used'].sum()),
        'fallback_fraction': float(out['fallback_used'].mean()),
        'coverage_distribution': {
            'mean': float(ok['atom_coverage'].mean()),
            'median': float(ok['atom_coverage'].median()),
            'pct_full_coverage': float((ok['atom_coverage'] >= 0.999).mean() * 100),
            'pct_below_half': float((ok['atom_coverage'] < 0.5).mean() * 100),
        },
        'by_atom_coverage': by_cov,
        'by_coverage_band': by_band,
        'by_family': {fam: stats(ok[ok['family'] == fam])
                      for fam in sorted(ok['family'].unique())},
        'worst_20': ok.nlargest(20, 'abs_error')[
            ['smiles', 'family', 'exp_tg', 'pred_tg', 'abs_error',
             'atom_coverage', 'unmatched_elements']].to_dict('records'),
        'most_common_unmatched_elements': (
            ok[ok['unmatched_elements'] != '']['unmatched_elements']
            .value_counts().head(15).to_dict()),
    }

    # ---- the eight canonical polymers, re-examined ------------------------
    canon = {}
    for name, psmi in CANONICAL.items():
        r = calc.calculate(psmi)
        cov, n_un, elems = atom_coverage(calc, psmi)
        hit = df.index[df['smiles'].apply(
            lambda s: Chem.CanonSmiles(s.replace('[*]', '[H]'))
            == Chem.CanonSmiles(psmi.replace('[*]', '[H]'))
            if Chem.MolFromSmiles(s.replace('[*]', '[H]')) else False)].tolist()
        canon[name] = {
            'psmiles': psmi,
            'manuscript_experimental_tg_K': CANONICAL_EXP[name],
            'gc_predicted_tg_K': round(float(r.tg), 2) if r.tg else None,
            'error_vs_manuscript_K': round(abs(float(r.tg) - CANONICAL_EXP[name]), 2)
            if r.tg else None,
            'atom_coverage': cov,
            'groups': r.groups_found,
            'corrections': r.corrections_applied,
            'in_training_dataset': bool(hit),
            'dataset_tg_K': float(df.loc[hit[0], 'target']) if hit else None,
            'dataset_vs_manuscript_delta_K': (
                round(float(df.loc[hit[0], 'target']) - CANONICAL_EXP[name], 2)
                if hit else None),
        }
    summary['canonical_eight'] = canon
    summary['canonical_eight_mae'] = float(np.mean(
        [v['error_vs_manuscript_K'] for v in canon.values()
         if v['error_vs_manuscript_K'] is not None]))

    dump('s04_gc_summary.json', summary)

    print('\n=== Tier 1 group contribution, full dataset ===')
    print('all:', summary['overall_all_predictions'])
    print('excl. fallback:', summary['overall_excluding_fallback'])
    print('fallback n =', summary['n_fallback'])
    print('coverage:', summary['coverage_distribution'])
    print('canonical-8 MAE:', summary['canonical_eight_mae'])


if __name__ == '__main__':
    main()
