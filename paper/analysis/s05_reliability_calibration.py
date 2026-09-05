"""
Script 05 - calibration of the applicability domain, the reliability index and
the uncertainty estimate.

Answers:
  * Reviewer 1 - "A reliability score is only useful if high-score predictions
    are demonstrably more accurate than low-score predictions." Calibration
    plots, error by PRI bin and coverage by confidence are produced here.
  * Reviewer 2, Major 9 - every component is now estimated rather than
    asserted, and nothing is fitted on test labels. The logistic slopes, the
    domain threshold and the error normalisation constant are all fitted on
    out-of-fold predictions over the TRAINING partition only. The model
    component uses the validation R-squared, not the test R-squared.
  * Reviewer 2, Major 10 - RF/GB disagreement is tested as an error predictor
    (Spearman rank correlation, error by disagreement decile) and is compared
    against a bootstrap ensemble and against split-conformal intervals, whose
    empirical coverage is measured.

Nothing here reads a test label until the final evaluation step.

Usage: python s05_reliability_calibration.py [protocol] [seed]

Writes: outputs/s05_calibration_<protocol>_<seed>.json
        outputs/s05_perpolymer_<protocol>_<seed>.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from polyx_rev import (OUT, dump, fit_outlier_filter, load_dataset,
                       morgan_matrix)
from s01_dataset_and_splits import family

RF_KW = dict(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
GB_KW = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
             min_samples_leaf=5, random_state=42)


# --------------------------------------------------------------------------
def max_tanimoto(query_fp, ref_fps, exclude_self=False):
    """Max Tanimoto of each query row against the reference set."""
    q = query_fp.astype(bool)
    r = ref_fps.astype(bool)
    inter = q @ r.T.astype(np.float32)
    qs = q.sum(1)[:, None].astype(np.float32)
    rs = r.sum(1)[None, :].astype(np.float32)
    union = qs + rs - inter
    sims = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    return sims.max(axis=1), sims.argmax(axis=1)


def logistic(x, k, tau):
    return 1.0 / (1.0 + np.exp(-k * (x - tau)))


def inv_logistic(x, k, centre):
    return 1.0 / (1.0 + np.exp(k * (x - centre)))


# --------------------------------------------------------------------------
def main():
    proto = sys.argv[1] if len(sys.argv) > 1 else 'published'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    df, _ = load_dataset()
    y = df['target'].values.astype(float)
    smiles = df['smiles'].tolist()
    X = morgan_matrix(smiles, radius=2, n_bits=2048)

    z = np.load(OUT / 's02_splits.npz')
    tr = z[f'{proto}_{seed}_train']
    va = z[f'{proto}_{seed}_val']
    te = z[f'{proto}_{seed}_test']
    lo, hi = fit_outlier_filter(y[tr])
    tr = tr[(y[tr] >= lo) & (y[tr] <= hi)]
    print(f'[split] {proto}/{seed}: train {len(tr)} val {len(va)} test {len(te)}')

    # ---------------------------------------------------------------- STEP 1
    # Out-of-fold predictions on the TRAINING partition. Everything that has to
    # be calibrated is calibrated from these, so no test label is ever used.
    print('[oof] 5-fold out-of-fold predictions over the training partition ...')
    oof_rf = np.zeros(len(tr))
    oof_gb = np.zeros(len(tr))
    # Similarity of each held-out fold member to the folds it was NOT trained
    # on. This is the geometry a real query sees. Leave-one-out similarity
    # inside the whole training partition is not: a dense training set almost
    # always contains a near neighbour, which flattens the fitted curve.
    oof_sim = np.zeros(len(tr))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for f, (a, b) in enumerate(kf.split(tr), 1):
        ia, ib = tr[a], tr[b]
        rf = RandomForestRegressor(**RF_KW).fit(X[ia], y[ia])
        gb = GradientBoostingRegressor(**GB_KW).fit(X[ia], y[ia])
        oof_rf[b] = rf.predict(X[ib])
        oof_gb[b] = gb.predict(X[ib])
        oof_sim[b], _ = max_tanimoto(X[ib], X[ia])
        print(f'  fold {f}/5 done', flush=True)

    oof_ens = 0.5 * (oof_rf + oof_gb)
    oof_disagree = np.abs(oof_rf - oof_gb) / 2.0
    oof_err = np.abs(oof_ens - y[tr])
    oof_mae = float(oof_err.mean())
    print(f'[oof] training out-of-fold MAE = {oof_mae:.2f} K')

    # Persist the out-of-fold arrays so the calibration can be refitted without
    # repeating the cross-validation.
    np.savez_compressed(
        OUT / f's05_oof_{proto}_{seed}.npz',
        train_idx=tr, oof_rf=oof_rf, oof_gb=oof_gb, oof_sim=oof_sim,
        y_train=y[tr])

    # ---------------------------------------------------------------- STEP 2
    # Fit the two logistic components on the out-of-fold data.
    # Target: a bounded "accuracy score" that is 1 when the error is zero and
    # decays with the out-of-fold MAE as the natural scale.
    acc = np.exp(-oof_err / oof_mae)

    try:
        (k_ad, tau_ad), _ = curve_fit(logistic, oof_sim, acc,
                                      p0=[15.0, 0.30],
                                      bounds=([0.1, 0.0], [100.0, 1.0]),
                                      maxfev=20000)
    except Exception:
        k_ad, tau_ad = 15.0, 0.30
    try:
        (k_un, c_un), _ = curve_fit(inv_logistic, oof_disagree, acc,
                                    p0=[0.08, oof_mae],
                                    bounds=([0.001, 1.0], [1.0, 400.0]),
                                    maxfev=20000)
    except Exception:
        k_un, c_un = 0.08, oof_mae

    print(f'[fit] AD logistic:  k = {k_ad:.3f}, tau = {tau_ad:.3f} '
          f'(submitted: k = 15, tau = 0.30 in the text; '
          f'clipped-linear in the code)')
    print(f'[fit] unc logistic: k = {k_un:.4f}/K, centre = {c_un:.1f} K '
          f'(submitted: k = 0.08/K, centre = 44 K taken from held-out '
          f'performance)')

    # Model component: validation R-squared, computed with a model that never
    # saw the validation split, NOT the test R-squared used previously.
    rf_full = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
    gb_full = GradientBoostingRegressor(**GB_KW).fit(X[tr], y[tr])
    from sklearn.metrics import r2_score
    val_pred = 0.5 * (rf_full.predict(X[va]) + gb_full.predict(X[va]))
    model_component = float(max(0.0, r2_score(y[va], val_pred)))
    print(f'[fit] model component (validation R2) = {model_component:.4f}')

    # ---------------------------------------------------------------- STEP 3
    # Split-conformal prediction intervals, normalised by the disagreement.
    # Calibrated on the validation partition; evaluated on test.
    va_rf, va_gb = rf_full.predict(X[va]), gb_full.predict(X[va])
    va_ens = 0.5 * (va_rf + va_gb)
    va_dis = np.abs(va_rf - va_gb) / 2.0
    va_res = np.abs(va_ens - y[va])
    kappa = float(np.median(va_dis)) + 1e-9        # stabiliser
    nonconf = va_res / (va_dis + kappa)
    q_levels = {}
    for alpha in (0.10, 0.20, 0.32):
        n = len(nonconf)
        q = float(np.quantile(nonconf, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
        q_levels[str(alpha)] = q

    # ---------------------------------------------------------------- STEP 4
    # Apply everything to the test partition, then and only then look at the
    # test labels.
    te_rf, te_gb = rf_full.predict(X[te]), gb_full.predict(X[te])
    te_ens = 0.5 * (te_rf + te_gb)
    te_dis = np.abs(te_rf - te_gb) / 2.0
    te_err = np.abs(te_ens - y[te])
    te_sim, te_nn = max_tanimoto(X[te], X[tr])

    r_ad_new = logistic(te_sim, k_ad, tau_ad)
    r_un_new = inv_logistic(te_dis, k_un, c_un)
    pri_new = 0.35 * r_ad_new + 0.35 * r_un_new + 0.30 * model_component

    # The submitted scheme, for side-by-side comparison. The manuscript wrote
    # logistics; the deployed code used clipped linear maps and the test R2.
    r_ad_paper = logistic(te_sim, 15.0, 0.30)
    r_un_paper = inv_logistic(te_dis, 0.08, 44.0)
    pri_paper = 0.35 * r_ad_paper + 0.35 * r_un_paper + 0.30 * 0.5532
    r_ad_code = np.clip(te_sim / 0.5 + 0.1, 0.1, 1.0)
    r_un_code = np.clip(1.0 - te_dis / 100.0, 0.0, 1.0)
    pri_code = 0.35 * r_ad_code + 0.35 * r_un_code + 0.30 * 0.5532

    def categorise(p):
        return np.where(p >= 0.75, 'high',
                        np.where(p >= 0.50, 'moderate',
                                 np.where(p >= 0.30, 'low', 'unreliable')))

    fams = [family(smiles[int(i)]) for i in te]
    per = pd.DataFrame({
        'smiles': [smiles[int(i)] for i in te],
        'family': fams,
        'y_true': y[te], 'y_pred': te_ens, 'abs_error': te_err,
        'rf_pred': te_rf, 'gb_pred': te_gb, 'disagreement': te_dis,
        'max_tanimoto': te_sim,
        'nearest_train_smiles': [smiles[int(tr[j])] for j in te_nn],
        'ad_status': np.where(te_sim >= 0.30, 'IN_DOMAIN',
                              np.where(te_sim >= 0.15, 'BORDERLINE',
                                       'OUT_OF_DOMAIN')),
        'pri_recalibrated': pri_new, 'pri_category': categorise(pri_new),
        'pri_as_published_equations': pri_paper,
        'pri_as_deployed_code': pri_code,
    })
    for alpha, q in q_levels.items():
        w = q * (te_dis + kappa)
        per[f'conformal_halfwidth_a{alpha}'] = w
        per[f'conformal_covered_a{alpha}'] = (te_err <= w)
    per.to_csv(OUT / f's05_perpolymer_{proto}_{seed}.csv', index=False)

    # ---------------------------------------------------------------- STEP 5
    def by_bin(values, n_bins=5, labels=None):
        qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9
        out = []
        for i in range(n_bins):
            m = (values > qs[i]) & (values <= qs[i + 1])
            if m.sum() == 0:
                continue
            out.append({
                'bin': labels[i] if labels else
                       f'{qs[i]:.3f}-{qs[i+1]:.3f}',
                'n': int(m.sum()),
                'mae': float(te_err[m].mean()),
                'median_ae': float(np.median(te_err[m])),
                'rmse': float(np.sqrt((te_err[m] ** 2).mean())),
                'r2': float(r2_score(y[te][m], te_ens[m])) if m.sum() > 2 else None,
            })
        return out

    def by_category(cats, order):
        out = {}
        for c in order:
            m = cats == c
            if m.sum() == 0:
                out[c] = None
                continue
            out[c] = {'n': int(m.sum()), 'fraction': float(m.mean()),
                      'mae': float(te_err[m].mean()),
                      'median_ae': float(np.median(te_err[m])),
                      'rmse': float(np.sqrt((te_err[m] ** 2).mean()))}
        return out

    rho_dis, p_dis = spearmanr(te_dis, te_err)
    rho_sim, p_sim = spearmanr(te_sim, te_err)
    rho_pri, p_pri = spearmanr(pri_new, te_err)
    rho_pri_pub, p_pri_pub = spearmanr(pri_paper, te_err)

    # Weight sensitivity: the submitted manuscript asserted "<5% category
    # change under +/-0.05 weight perturbation" but no such function existed.
    # It is implemented and executed here.
    base_cat = categorise(pri_new)
    sens = {}
    for name, delta in [('ad', 0.05), ('ad', -0.05), ('unc', 0.05),
                        ('unc', -0.05), ('model', 0.05), ('model', -0.05)]:
        w = {'ad': 0.35, 'unc': 0.35, 'model': 0.30}
        w[name] = max(0.0, w[name] + delta)
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
        p = w['ad'] * r_ad_new + w['unc'] * r_un_new + w['model'] * model_component
        changed = float((categorise(p) != base_cat).mean() * 100)
        sens[f'{name}{delta:+.2f}'] = {'weights': w, 'pct_category_changed': changed}

    summary = {
        'protocol': proto, 'seed': seed,
        'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
        'fitted_on_training_out_of_fold_only': {
            'oof_mae_K': oof_mae,
            'ad_logistic_k': float(k_ad), 'ad_logistic_tau': float(tau_ad),
            'unc_logistic_k_per_K': float(k_un), 'unc_logistic_centre_K': float(c_un),
            'model_component_validation_r2': model_component,
        },
        'submitted_parameters_for_comparison': {
            'ad_logistic_k': 15.0, 'ad_logistic_tau': 0.30,
            'unc_logistic_k_per_K': 0.08, 'unc_logistic_centre_K': 44.0,
            'model_component': 0.5532,
            'note': ('44 K and 0.5532 both derive from held-out performance; '
                     'the deployed code additionally used clipped linear maps '
                     'rather than these logistics.'),
        },
        'test_overall': {
            'mae': float(te_err.mean()),
            'rmse': float(np.sqrt((te_err ** 2).mean())),
            'r2': float(r2_score(y[te], te_ens)),
        },
        'rank_correlation_with_absolute_error': {
            'disagreement': {'spearman': float(rho_dis), 'p': float(p_dis)},
            'max_tanimoto': {'spearman': float(rho_sim), 'p': float(p_sim)},
            'pri_recalibrated': {'spearman': float(rho_pri), 'p': float(p_pri)},
            'pri_as_published': {'spearman': float(rho_pri_pub),
                                 'p': float(p_pri_pub)},
        },
        'error_by_pri_quintile_recalibrated': by_bin(pri_new),
        'pri_quintile_monotone': bool(all(
            a >= b for a, b in zip([x['mae'] for x in by_bin(pri_new)],
                                   [x['mae'] for x in by_bin(pri_new)][1:]))),
        'error_by_pri_quintile_as_published': by_bin(pri_paper),
        'error_by_disagreement_decile': by_bin(te_dis, n_bins=10),
        'error_by_similarity_quintile': by_bin(te_sim),
        'error_by_pri_category': by_category(
            categorise(pri_new), ['high', 'moderate', 'low', 'unreliable']),
        'error_by_ad_status': by_category(
            per['ad_status'].values, ['IN_DOMAIN', 'BORDERLINE', 'OUT_OF_DOMAIN']),
        'ad_status_distribution': {
            k: int(v) for k, v in per['ad_status'].value_counts().items()},
        'pri_distribution_recalibrated': {
            'median': float(np.median(pri_new)),
            'q25': float(np.quantile(pri_new, 0.25)),
            'q75': float(np.quantile(pri_new, 0.75)),
            'pct_high': float((pri_new >= 0.75).mean() * 100),
            'pct_unreliable': float((pri_new < 0.30).mean() * 100),
        },
        'conformal': {
            'calibrated_on': 'validation partition',
            'normaliser': 'RF-GB disagreement + median(disagreement)',
            'levels': {
                a: {'quantile': q,
                    'empirical_coverage': float(per[f'conformal_covered_a{a}'].mean()),
                    'nominal_coverage': 1 - float(a),
                    'median_halfwidth_K': float(
                        per[f'conformal_halfwidth_a{a}'].median())}
                for a, q in q_levels.items()},
        },
        'weight_sensitivity_analysis': sens,
        'error_by_family': {
            f: {'n': int((per['family'] == f).sum()),
                'mae': float(per.loc[per['family'] == f, 'abs_error'].mean())}
            for f in sorted(per['family'].unique())},
    }
    dump(f's05_calibration_{proto}_{seed}.json', summary)

    print('\n--- error by recalibrated PRI quintile (low to high) ---')
    for b in summary['error_by_pri_quintile_recalibrated']:
        print(f"  {b['bin']}: n={b['n']:4d} MAE={b['mae']:.1f} K")
    print('--- conformal coverage ---')
    for a, v in summary['conformal']['levels'].items():
        print(f"  nominal {v['nominal_coverage']:.2f} -> empirical "
              f"{v['empirical_coverage']:.3f} (median half-width "
              f"{v['median_halfwidth_K']:.1f} K)")
    print(f"--- Spearman(disagreement, |error|) = {rho_dis:.3f} (p={p_dis:.2g})")


if __name__ == '__main__':
    main()
