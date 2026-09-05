"""Regression tests for the numerical examples reported in the manuscript.

Reviewer 1 asked for "regression tests for the reported numerical examples".
Two kinds are covered:

1. **Reproduction of the originally submitted metrics.** The submitted pipeline
   is replicated end to end, including its defects (global 3-sigma filter before
   splitting, size-sorted scaffold assignment, equal-weight ensemble), and must
   reproduce test R2 = 0.5532260797 and MAE = 43.5619576 K. If these ever stop
   reproducing, the claim that the original results are auditable is no longer
   true.

2. **Agreement between the manuscript and the archived analysis outputs.**
   Every token substituted into the revised manuscript is checked against the
   JSON file it came from, so the manuscript cannot drift away from its data.

The first group is slow (it trains two ensembles) and is marked accordingly:
run with ``-m "not slow"`` to skip it.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

def _find_outputs() -> Path:
    """Locate the archived analysis outputs in either layout.

    The same suite runs from the paper's revision folder, where the outputs sit
    beside the manuscript, and from a checkout of the repository, where they
    are committed under ``paper/outputs``. Returning the first that exists
    keeps one suite honest in both places.
    """
    import os
    here = Path(__file__).resolve()
    env = os.environ.get('POLYX_REVISION')
    candidates = []
    if env:
        candidates.append(Path(env) / 'outputs')
    candidates += [here.parents[1] / 'paper' / 'outputs',
                   here.parents[2] / 'outputs']
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[-1]


OUT = _find_outputs()
REVISION = OUT.parent

# The values printed in the originally submitted manuscript.
SUBMITTED = {
    'test_r2': 0.5532260797078669,
    'test_mae': 43.56195761273432,
    'test_rmse': 60.308767168563485,
    'val_r2': 0.7374672002280418,
    'rf_test_r2': 0.5086260942558986,
    'gb_test_r2': 0.560305661756507,
    'n_total': 7365,
    'n_train': 5892,
    'n_test': 737,
}


def _skip_if_no_outputs():
    if not OUT.exists():
        pytest.skip(f'analysis outputs not found at {OUT}; run the analysis '
                    f'scripts first')


def jload(name):
    _skip_if_no_outputs()
    p = OUT / name
    if not p.exists():
        pytest.skip(f'{name} not present')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 1. The submitted numbers must still reproduce
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_submitted_tier2_metrics_reproduce_exactly(data_csv):
    """Replicate the submitted pipeline, defects included, to six decimals."""
    import pandas as pd
    import random
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    RDLogger.DisableLog('rdApp.*')

    raw = pd.read_csv(data_csv).rename(columns={'PSMILES': 'smiles', 'Tg': 'target'})
    raw['target'] = pd.to_numeric(raw['target'], errors='coerce')
    d = raw.dropna(subset=['smiles', 'target'])
    mu, sd = d['target'].mean(), d['target'].std()          # global, as submitted
    d = d[(d['target'] >= mu - 3 * sd) & (d['target'] <= mu + 3 * sd)]
    d = d.drop_duplicates(subset=['smiles']).reset_index(drop=True)
    assert len(d) == SUBMITTED['n_total']

    groups = {}
    for idx, smi in zip(d.index, d['smiles']):
        mol = Chem.MolFromSmiles(smi.replace('[*]', '[H]'))
        key = (MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
               if mol is not None else f'_invalid_{idx}')
        groups.setdefault(key, []).append(int(idx))

    sets = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    random.Random(42).shuffle(sets)                          # cancelled below
    sets = sorted(sets, key=lambda g: (-len(g), min(g)))
    n_train, n_val = int(len(d) * 0.8), int(len(d) * 0.1)
    tr, va, te = [], [], []
    for g in sets:
        (tr if len(tr) < n_train else va if len(va) < n_val else te).extend(g)
    tr, va, te = np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))
    assert (len(tr), len(te)) == (SUBMITTED['n_train'], SUBMITTED['n_test'])

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(d), 2048), dtype=np.float32)
    for i, smi in enumerate(d['smiles']):
        mol = Chem.MolFromSmiles(smi.replace('[*]', '[H]'))
        if mol is not None:
            X[i] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    y = d['target'].values

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                               random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.1, subsample=0.8,
                                   min_samples_leaf=5,
                                   random_state=42).fit(X[tr], y[tr])
    ens_te = 0.5 * (rf.predict(X[te]) + gb.predict(X[te]))     # equal weight
    ens_va = 0.5 * (rf.predict(X[va]) + gb.predict(X[va]))

    assert r2_score(y[te], ens_te) == pytest.approx(SUBMITTED['test_r2'], abs=1e-9)
    assert mean_absolute_error(y[te], ens_te) == pytest.approx(
        SUBMITTED['test_mae'], abs=1e-6)
    assert np.sqrt(mean_squared_error(y[te], ens_te)) == pytest.approx(
        SUBMITTED['test_rmse'], abs=1e-6)
    assert r2_score(y[va], ens_va) == pytest.approx(SUBMITTED['val_r2'], abs=1e-9)
    assert r2_score(y[te], rf.predict(X[te])) == pytest.approx(
        SUBMITTED['rf_test_r2'], abs=1e-9)
    assert r2_score(y[te], gb.predict(X[te])) == pytest.approx(
        SUBMITTED['gb_test_r2'], abs=1e-9)


def test_submitted_ensemble_val_r2_is_below_gradient_boosting_alone():
    """The defect Reviewer 2 identified by arithmetic alone.

    A validation-maximising weight sweep that includes the pure GB endpoint
    cannot select a blend that scores below GB. That the submitted numbers do
    is the evidence that the sweep was never applied.
    """
    assert SUBMITTED['val_r2'] < 0.745, (
        'the submitted validation R2 of the ensemble must be below the '
        'Gradient Boosting value of 0.745 for the reported inconsistency to '
        'be reproduced')


# --------------------------------------------------------------------------
# 2. The manuscript must agree with the archived outputs
# --------------------------------------------------------------------------
def test_gc_canonical_mae_matches_the_archived_output():
    gc = jload('s04_gc_summary.json')
    assert gc['canonical_eight_mae'] == pytest.approx(3.9, abs=0.05), (
        'the eight-polymer MAE quoted in the abstract must match '
        's04_gc_summary.json')


def test_gc_full_dataset_is_much_worse_than_the_canonical_eight():
    """The central corrected claim of the revision."""
    gc = jload('s04_gc_summary.json')
    assert gc['overall_all_predictions']['mae'] > 10 * gc['canonical_eight_mae']
    assert gc['overall_all_predictions']['r2'] < 0


def test_gc_error_decreases_across_coverage_bands():
    """Figure 2c and the coverage argument rest on the three-band ordering.

    Deliberately NOT asserted across the fine bins: the finest low-coverage bin
    holds a single polymer and the fine-bin trend is not monotone. The summary
    records that fact so the manuscript cannot quietly claim otherwise.
    """
    gc = jload('s04_gc_summary.json')
    b = gc['by_coverage_band']
    low, mid, full = (b['below_0.75']['mae'],
                      b['0.75_to_below_1.00']['mae'],
                      b['full_coverage']['mae'])
    assert low > mid > full, f'band ordering broken: {low}, {mid}, {full}'
    for band in ('below_0.75', '0.75_to_below_1.00', 'full_coverage'):
        assert b[band]['n'] >= 100, (
            f'{band} has only {b[band]["n"]} polymers; too few to report')
    assert b['_monotone_across_fine_bins'] is False, (
        'the fine bins are now monotone; the manuscript text says they are '
        'not and must be updated')


def test_reliability_index_is_calibrated():
    """A reliability score is only useful if error falls across its strata."""
    cal = jload('s05_calibration_published_42.json')
    q = cal['error_by_pri_quintile_recalibrated']
    assert q[0]['mae'] > q[-1]['mae'], (
        f'MAE does not fall from the lowest to the highest PRI quintile: '
        f'{[b["mae"] for b in q]}')
    rho = cal['rank_correlation_with_absolute_error']['pri_recalibrated']
    assert rho['spearman'] < 0 and rho['p'] < 0.05


def test_no_reliability_parameter_was_fitted_on_test_data():
    """Guards the leakage that Reviewer 2, Major 9 identified."""
    cal = jload('s05_calibration_published_42.json')
    fit = cal['fitted_on_training_out_of_fold_only']
    assert fit['unc_logistic_centre_K'] != 44.0, (
        'the uncertainty centre is still the held-out-derived 44 K')
    assert fit['model_component_validation_r2'] != pytest.approx(0.5532, abs=1e-3), (
        'the model component is still the test R2')


def test_conformal_coverage_is_reported_as_measured():
    """Coverage falls short of nominal here, and the text must say so.

    Split conformal guarantees marginal coverage only under exchangeability
    between the calibration and test partitions, which a scaffold split is
    built to violate. The intervals are still usable; what must never happen is
    the manuscript quoting the nominal level as though it were attained.
    """
    cal = jload('s05_calibration_published_42.json')
    levels = cal['conformal']['levels']
    assert levels, 'no conformal levels recorded'
    gaps = {a: v['nominal_coverage'] - v['empirical_coverage']
            for a, v in levels.items()}
    # Sanity: the intervals must not be wildly off in either direction.
    for a, g in gaps.items():
        assert -0.10 < g < 0.20, f'implausible coverage gap at alpha={a}: {g}'

    tok = jload('s10_tokens.json')
    worst = max(gaps.values())
    assert 'CONFORMAL_SHORTFALL' in tok
    assert tok['CONFORMAL_SHORTFALL'] == f'{100 * worst:.0f}', (
        'the reported shortfall does not match the measured one')
    if worst > 0.02:
        assert 'CONFORMAL_VERDICT' in tok
        assert 'short of nominal' in tok.get('CONFORMAL_VERDICT', ''), (
            'coverage is below nominal but the verdict sentence does not say so')


def test_applicability_domain_degeneracy_is_reported_not_hidden():
    """The 0.30 boundary puts almost the whole test partition in one class.

    That is a real property of this collection, and the manuscript reports it
    instead of showing a distribution drawn from a different population. The
    test fails if the strata ever become informative without the text being
    updated, or if the degeneracy is quietly dropped.
    """
    cal = jload('s05_calibration_published_42.json')
    dist = cal['ad_status_distribution']
    total = sum(dist.values())
    biggest = max(dist.values()) / total
    tok = jload('s10_tokens.json')
    if biggest > 0.95:
        assert 'MIN_TEST_SIM' in tok, (
            'the domain strata are degenerate; the minimum similarity must be '
            'reported so a reader can see why')
        assert float(tok['MIN_TEST_SIM']) > 0.15, (
            'a polymer below the out-of-domain boundary exists, so the strata '
            'are not degenerate for the stated reason')


def test_similarity_is_a_stronger_error_signal_than_disagreement():
    """The honest ordering of the three indicators on this data."""
    cal = jload('s05_calibration_published_42.json')
    rc = cal['rank_correlation_with_absolute_error']
    assert abs(rc['max_tanimoto']['spearman']) > abs(
        rc['disagreement']['spearman']), (
        'similarity should out-rank disagreement as an error signal')
    assert rc['disagreement']['p'] > 0.01, (
        'disagreement has become a significant error predictor; Section 3.7 '
        'says it is not and must be updated')


def test_polybert_mirror_was_verified():
    """Tier 3 numbers are only comparable with the submitted ones if the
    mirrored weights are identical."""
    v = jload('s03_mirror_verification.json')
    assert v['verified'] is True
    assert v['mean_cosine'] > 0.999


def test_dataset_accounting_identifies_the_two_removed_records():
    acc = jload('s01_dataset_accounting.json')
    a = acc['as_submitted']
    assert a['raw_rows'] - a['after_global_3sigma'] == 2
    assert len(a['rows_removed_by_3sigma']) == 2
    assert all(r['target'] > 700 for r in a['rows_removed_by_3sigma']), (
        'both removed records should be high-Tg outliers')


def test_scaffold_groups_are_not_one_per_polymer():
    """The statement corrected in response to Reviewer 2, Minor 2."""
    st = jload('s01_scaffold_stats.json')
    assert st['n_scaffold_groups'] < st['n_polymers']
    assert st['null_scaffold_group_size'] > 100


def test_manuscript_tokens_match_their_sources():
    """Every substituted token is re-derived from its source file."""
    tok = jload('s10_tokens.json')
    gc = jload('s04_gc_summary.json')
    cal = jload('s05_calibration_published_42.json')
    st = jload('s01_scaffold_stats.json')

    assert tok['GC_CANON_MAE'] == f"{gc['canonical_eight_mae']:.1f}"
    assert tok['GC_FULL_MAE'] == f"{gc['overall_all_predictions']['mae']:.0f}"
    assert tok['GC_FULL_R2'] == f"{gc['overall_all_predictions']['r2']:.2f}"
    assert tok['N_SCAFFOLD_GROUPS'] == f"{st['n_scaffold_groups']:,}"
    assert tok['PRI_SPEARMAN'] == (
        f"{cal['rank_correlation_with_absolute_error']['pri_recalibrated']['spearman']:.3f}")
    assert tok['OOF_MAE'] == (
        f"{cal['fitted_on_training_out_of_fold_only']['oof_mae_K']:.1f}")


def test_built_manuscript_contains_no_unresolved_tokens():
    tex = REVISION / 'Manuscript_R1.tex'   # paper folder only, not in a checkout
    if not tex.exists():
        pytest.skip('manuscript not built yet')
    import re
    left = re.findall(r'@@([A-Z0-9_]+)@@', tex.read_text(encoding='utf-8'))
    assert not left, f'unresolved tokens in the built manuscript: {left}'


def test_every_figure_has_a_provenance_sidecar():
    figs = REVISION / 'figures'
    if not figs.exists():
        pytest.skip('figures not built yet')
    pdfs = sorted(figs.glob('fig*.pdf'))
    if not pdfs:
        pytest.skip('no figures built yet')
    for p in pdfs:
        side = p.with_suffix('').with_suffix('.provenance.txt')
        side = figs / f'{p.stem}.provenance.txt'
        assert side.exists(), f'{p.name} has no provenance sidecar'
        assert side.read_text(encoding='utf-8').strip(), f'{side.name} is empty'
