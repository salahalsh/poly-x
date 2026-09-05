"""
Script 08 - external validation and a prospective design case study.

Answers:
  * the submitted manuscript's own Limitation 6 ("No predictions on an
    independent external test set from a different source are reported"), and
    Reviewer 1's request for group-contribution failures on conjugated systems.
    An independent experimental set of conjugated polymers from Tao et al.
    (*Patterns* 2021, PolyInfo-derived, distributed with the Polymer_Tg_
    repository) is used; any structure also present in the training data is
    removed by canonical PSMILES before scoring.
  * Reviewer 2, Major 14 - a design problem in which the three tiers and the
    reliability indicators change the decision, with the selection validated
    against held-out measured values, plus a genuinely prospective screen of
    PI1M for which we state plainly that no experiment was performed.

Design problem
--------------
"Select amorphous polymers with Tg in [450, 500] K (a high-temperature
engineering window) that the platform believes it can predict."

The candidate pool is the held-out test partition, whose labels are never used
for fitting, selection or calibration. Precision-at-k of the selection is
therefore an honest simulation of a prospective campaign, and lets us ask the
question Reviewer 2 poses: do the reliability indicators separate dependable
from failed prospective predictions?

Writes: outputs/s08_external_conjugated.json / .csv
        outputs/s08_case_study.json
        outputs/s08_pi1m_candidates.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from polyx_rev import (OUT, canonical_psmiles, dump, fit_outlier_filter,
                       load_dataset, metrics, morgan_matrix)
from s01_dataset_and_splits import family
from s05_reliability_calibration import inv_logistic, logistic, max_tanimoto

RDLogger.DisableLog('rdApp.*')

from _paths import DATA as POLYX_DATA, SERVICES  # noqa: E402
CONJ = POLYX_DATA / "Polymer_Tg_" / "Data" / "32_Conjugate_Polymer.txt"
PI1M = POLYX_DATA / "PI1M" / "PI1M.csv"

_GC = SERVICES / "group_contribution.py"
_spec = importlib.util.spec_from_file_location("polyx_gc", _GC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GroupContributionCalculator = _mod.GroupContributionCalculator

RF_KW = dict(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
GB_KW = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
             min_samples_leaf=5, random_state=42)

TARGET_LO, TARGET_HI = 450.0, 500.0


def fit_tier2(X, y, tr):
    rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
    gb = GradientBoostingRegressor(**GB_KW).fit(X[tr], y[tr])
    return rf, gb


# --------------------------------------------------------------------------
def external_conjugated(rf, gb, X_train_fp, train_canon, calc):
    """Score the independent conjugated-polymer set."""
    raw = pd.read_csv(CONJ, sep='\t')
    raw.columns = [c.strip() for c in raw.columns]
    rows = []
    for tg_c, smi in zip(raw['Tg'], raw['Smiles']):
        canon = canonical_psmiles(smi)
        if canon is None:
            continue
        tg_k = float(tg_c) + 273.15          # source reports degrees Celsius
        fp = morgan_matrix([smi], radius=2, n_bits=2048)
        sim, nn = max_tanimoto(fp, X_train_fp)
        p_rf, p_gb = float(rf.predict(fp)[0]), float(gb.predict(fp)[0])
        gc = calc.calculate(smi)
        rows.append({
            'smiles': smi, 'canonical': canon,
            'exp_tg_K': tg_k,
            'in_training_set': canon in train_canon,
            'tier1_gc_tg_K': float(gc.tg) if gc.tg is not None else np.nan,
            'tier2_ml_tg_K': 0.5 * (p_rf + p_gb),
            'rf': p_rf, 'gb': p_gb,
            'disagreement_K': abs(p_rf - p_gb) / 2.0,
            'max_tanimoto': float(sim[0]),
        })
    d = pd.DataFrame(rows)
    unseen = d[~d['in_training_set']].copy()
    for col, tag in (('tier1_gc_tg_K', 'tier1_gc'), ('tier2_ml_tg_K', 'tier2_ml')):
        unseen[f'{tag}_abs_error'] = (unseen[col] - unseen['exp_tg_K']).abs()
    unseen.to_csv(OUT / 's08_external_conjugated.csv', index=False)

    res = {
        'source': ('Tao et al., Machine Learning Discovery of High-Temperature '
                   'Polymers, Patterns 2021; PolyInfo-derived conjugated '
                   'polymer set (32_Conjugate_Polymer.txt)'),
        'n_total': int(len(d)),
        'n_already_in_training': int(d['in_training_set'].sum()),
        'n_external_unseen': int(len(unseen)),
        'exp_tg_range_K': [float(unseen['exp_tg_K'].min()),
                           float(unseen['exp_tg_K'].max())],
        'mean_max_tanimoto_to_training': float(unseen['max_tanimoto'].mean()),
        'pct_out_of_domain': float((unseen['max_tanimoto'] < 0.15).mean() * 100),
        'pct_in_domain': float((unseen['max_tanimoto'] >= 0.30).mean() * 100),
    }
    for col, tag in (('tier1_gc_tg_K', 'tier1_gc'), ('tier2_ml_tg_K', 'tier2_ml')):
        sub = unseen[unseen[col].notna()]
        res[tag] = metrics(sub['exp_tg_K'].values, sub[col].values) if len(sub) > 2 else None
        if res[tag]:
            res[tag]['bias_K'] = float((sub[col] - sub['exp_tg_K']).mean())
            res[tag]['n'] = int(len(sub))
    dump('s08_external_conjugated.json', res)
    return res


# --------------------------------------------------------------------------
def case_study(rf, gb, X, y, tr, te, smiles, calib, calc):
    """Simulated prospective selection on the untouched test partition."""
    p_rf, p_gb = rf.predict(X[te]), gb.predict(X[te])
    pred = 0.5 * (p_rf + p_gb)
    dis = np.abs(p_rf - p_gb) / 2.0
    sim, _ = max_tanimoto(X[te], X[tr])
    r_ad = logistic(sim, calib['ad_logistic_k'], calib['ad_logistic_tau'])
    r_un = inv_logistic(dis, calib['unc_logistic_k_per_K'],
                        calib['unc_logistic_centre_K'])
    pri = 0.35 * r_ad + 0.35 * r_un + 0.30 * calib['model_component_validation_r2']

    gc_pred = np.array([
        (lambda r: float(r.tg) if r.tg is not None else np.nan)(
            calc.calculate(smiles[int(i)])) for i in te])

    truth = y[te]
    hit = (truth >= TARGET_LO) & (truth <= TARGET_HI)
    base_rate = float(hit.mean())

    def precision_at_k(score_desc, mask, k):
        """Rank candidates that pass `mask` by `score_desc`, take top k."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        order = idx[np.argsort(-score_desc[idx])]
        top = order[:k]
        return {'k': int(len(top)), 'n_eligible': int(len(idx)),
                'precision': float(hit[top].mean()),
                'mae_on_selected': float(np.abs(pred[top] - truth[top]).mean()),
                'enrichment_over_base_rate': float(hit[top].mean() / base_rate)
                if base_rate > 0 else None}

    # Selection strategies. "in window" = the tier's own prediction lands in
    # the target window; the score then ranks by how central it is.
    centre = 0.5 * (TARGET_LO + TARGET_HI)
    strategies = {}
    for name, p in (('tier1_gc', gc_pred), ('tier2_ml', pred)):
        m = np.isfinite(p) & (p >= TARGET_LO) & (p <= TARGET_HI)
        score = -np.abs(np.nan_to_num(p, nan=1e9) - centre)
        strategies[name] = {
            'n_selected_by_window': int(m.sum()),
            **{f'top{k}': precision_at_k(score, m, k) for k in (10, 25, 50)},
        }
    # Tier 2 with the reliability filter applied.
    m_ml = (pred >= TARGET_LO) & (pred <= TARGET_HI)
    for label, extra in (('tier2_ml_in_domain', sim >= 0.30),
                         ('tier2_ml_pri_high', pri >= np.quantile(pri, 0.75)),
                         ('tier2_ml_low_disagreement',
                          dis <= np.quantile(dis, 0.25))):
        m = m_ml & extra
        score = -np.abs(pred - centre)
        strategies[label] = {
            'n_selected_by_window': int(m.sum()),
            **{f'top{k}': precision_at_k(score, m, k) for k in (10, 25, 50)},
        }
    # Consensus: both tiers place the candidate in the window.
    m_cons = m_ml & np.isfinite(gc_pred) & (gc_pred >= TARGET_LO) & (gc_pred <= TARGET_HI)
    strategies['tier1_and_tier2_consensus'] = {
        'n_selected_by_window': int(m_cons.sum()),
        **{f'top{k}': precision_at_k(-np.abs(pred - centre), m_cons, k)
           for k in (10, 25, 50)},
    }

    # Does reliability separate successes from failures among the selected?
    sel = m_ml
    res_sel = {
        'n_selected': int(sel.sum()),
        'precision_all_selected': float(hit[sel].mean()) if sel.sum() else None,
        'pri_mean_when_correct': float(pri[sel & hit].mean()) if (sel & hit).sum() else None,
        'pri_mean_when_wrong': float(pri[sel & ~hit].mean()) if (sel & ~hit).sum() else None,
        'similarity_mean_when_correct': float(sim[sel & hit].mean()) if (sel & hit).sum() else None,
        'similarity_mean_when_wrong': float(sim[sel & ~hit].mean()) if (sel & ~hit).sum() else None,
    }
    if (sel & hit).sum() > 2 and (sel & ~hit).sum() > 2:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(pri[sel & hit], pri[sel & ~hit], alternative='greater')
        res_sel['mannwhitney_pri_correct_gt_wrong_p'] = float(p)
        u2, p2 = mannwhitneyu(sim[sel & hit], sim[sel & ~hit], alternative='greater')
        res_sel['mannwhitney_similarity_correct_gt_wrong_p'] = float(p2)

    out = {
        'design_target': f'Tg in [{TARGET_LO}, {TARGET_HI}] K',
        'candidate_pool': 'held-out scaffold-split test partition',
        'n_candidates': int(len(te)),
        'n_true_hits_in_pool': int(hit.sum()),
        'base_rate': base_rate,
        'strategies': strategies,
        'reliability_separates_success_from_failure': res_sel,
    }
    dump('s08_case_study.json', out)
    return out


# --------------------------------------------------------------------------
def pi1m_screen(rf, gb, X_train_fp, train_canon, calib, calc, n_sample=40000):
    """A genuinely prospective screen. No experimental validation is claimed."""
    if not PI1M.exists():
        print('[pi1m] not available; skipped')
        return None
    df = pd.read_csv(PI1M)
    col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    cand = df[col].iloc[idx].astype(str).tolist()

    X = morgan_matrix(cand, radius=2, n_bits=2048)
    keep = X.sum(1) > 0
    cand = [c for c, k in zip(cand, keep) if k]
    X = X[keep]
    print(f'[pi1m] scoring {len(cand)} candidates')

    p_rf, p_gb = rf.predict(X), gb.predict(X)
    pred = 0.5 * (p_rf + p_gb)
    dis = np.abs(p_rf - p_gb) / 2.0
    sim, _ = max_tanimoto(X, X_train_fp)
    r_ad = logistic(sim, calib['ad_logistic_k'], calib['ad_logistic_tau'])
    r_un = inv_logistic(dis, calib['unc_logistic_k_per_K'],
                        calib['unc_logistic_centre_K'])
    pri = 0.35 * r_ad + 0.35 * r_un + 0.30 * calib['model_component_validation_r2']

    m = (pred >= TARGET_LO) & (pred <= TARGET_HI) & (sim >= 0.30)
    sel = np.where(m)[0]
    sel = sel[np.argsort(-pri[sel])][:50]
    rows = []
    for i in sel:
        gc = calc.calculate(cand[int(i)])
        canon = canonical_psmiles(cand[int(i)])
        rows.append({
            'psmiles': cand[int(i)],
            'tier2_ml_tg_K': float(pred[i]),
            'tier1_gc_tg_K': float(gc.tg) if gc.tg is not None else np.nan,
            'disagreement_K': float(dis[i]),
            'max_tanimoto_to_training': float(sim[i]),
            'pri': float(pri[i]),
            'already_in_training_set': bool(canon in train_canon),
            'family': family(cand[int(i)]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 's08_pi1m_candidates.csv', index=False)
    print(f'[pi1m] wrote {len(out)} candidates')
    return {'n_screened': int(len(cand)),
            'n_passing_window_and_domain': int(m.sum()),
            'n_reported': int(len(out)),
            'note': ('These are computational proposals only. No experimental '
                     'synthesis or measurement was performed in this work.')}


def main():
    import json
    df, _ = load_dataset()
    y = df['target'].values.astype(float)
    smiles = df['smiles'].tolist()
    X = morgan_matrix(smiles, radius=2, n_bits=2048)

    z = np.load(OUT / 's02_splits.npz')
    tr, te = z['published_42_train'], z['published_42_test']
    lo, hi = fit_outlier_filter(y[tr])
    tr = tr[(y[tr] >= lo) & (y[tr] <= hi)]

    calib_path = OUT / 's05_calibration_published_42.json'
    if calib_path.exists():
        with open(calib_path, encoding='utf-8') as f:
            calib = json.load(f)['fitted_on_training_out_of_fold_only']
    else:
        raise SystemExit('run s05_reliability_calibration.py first')

    print('[fit] Tier 2 on the training partition ...')
    rf, gb = fit_tier2(X, y, tr)
    calc = GroupContributionCalculator()
    train_canon = set(df['canonical'].iloc[tr])

    print('[external] conjugated polymers ...')
    ext = external_conjugated(rf, gb, X[tr], train_canon, calc)
    print(json.dumps({k: v for k, v in ext.items() if k != 'source'}, indent=2))

    print('[case study] simulated prospective selection ...')
    cs = case_study(rf, gb, X, y, tr, te, smiles, calib, calc)
    print(json.dumps(cs['reliability_separates_success_from_failure'], indent=2))

    print('[pi1m] prospective screen ...')
    pi = pi1m_screen(rf, gb, X[tr], train_canon, calib, calc)
    if pi:
        cs['pi1m_prospective_screen'] = pi
        dump('s08_case_study.json', cs)


if __name__ == '__main__':
    main()
