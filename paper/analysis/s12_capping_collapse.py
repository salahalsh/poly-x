"""
Script 12 - how often does the capping convention merge distinct polymers?

Found while validating the similarity helper: under the hydrogen capping used
by Tier 2, poly(ethylene oxide) ``[*]CCO[*]`` and poly(vinyl alcohol)
``[*]CC(O)[*]`` both become ethanol, so they receive *identical* ECFP4
fingerprints despite differing by 152 K in measured Tg. Replacing a wildcard by
a hydrogen deletes the attachment topology, and any two repeat units that differ
only in where the backbone continues become indistinguishable.

This quantifies the effect over the whole collection and measures the label
noise it injects, which is the concrete version of Reviewer 2's Major 7 point
about capping and deduplication.

Writes: outputs/s12_capping_collapse.json
        outputs/s12_collapsed_groups.csv
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem

from polyx_rev import OUT, canonical_psmiles, dump, load_dataset, to_mol


def capped_key(psmiles, cap):
    mol = to_mol(psmiles, cap=cap)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def collapse_stats(df, cap):
    groups = defaultdict(list)
    for i, smi in zip(df.index, df['smiles']):
        k = capped_key(smi, cap)
        if k is not None:
            groups[k].append(int(i))
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    n_polymers_merged = sum(len(v) for v in merged.values())

    spreads = []
    for v in merged.values():
        t = df.loc[v, 'target'].values
        spreads.append(float(t.max() - t.min()))
    return groups, merged, {
        'cap': cap,
        'n_distinct_molecules': len(groups),
        'n_polymers': int(len(df)),
        'n_merged_groups': len(merged),
        'n_polymers_in_merged_groups': n_polymers_merged,
        'pct_polymers_merged': round(100 * n_polymers_merged / len(df), 2),
        'max_tg_spread_within_a_merged_group_K': (
            round(max(spreads), 1) if spreads else 0.0),
        'median_tg_spread_within_merged_groups_K': (
            round(float(np.median(spreads)), 1) if spreads else 0.0),
        'mean_tg_spread_within_merged_groups_K': (
            round(float(np.mean(spreads)), 1) if spreads else 0.0),
    }


def main():
    df, _ = load_dataset()

    # The motivating pair, checked explicitly.
    pair = {}
    for name, psmi in (('PEO', '[*]CCO[*]'), ('PVOH', '[*]CC(O)[*]')):
        pair[name] = {
            'psmiles': psmi,
            'canonical_psmiles': canonical_psmiles(psmi),
            'hydrogen_capped': capped_key(psmi, 'hydrogen'),
            'methyl_capped': capped_key(psmi, 'methyl'),
            'wildcard_retained': capped_key(psmi, 'wildcard'),
        }
    pair['identical_under_hydrogen_capping'] = (
        pair['PEO']['hydrogen_capped'] == pair['PVOH']['hydrogen_capped'])
    pair['identical_under_methyl_capping'] = (
        pair['PEO']['methyl_capped'] == pair['PVOH']['methyl_capped'])
    pair['literature_tg_difference_K'] = 358 - 206

    out = {'motivating_pair': pair}
    per_cap = {}
    merged_h = None
    for cap in ('hydrogen', 'methyl', 'wildcard'):
        groups, merged, stats = collapse_stats(df, cap)
        per_cap[cap] = stats
        print(f'{cap:9s}: {stats["n_distinct_molecules"]:,} distinct molecules '
              f'from {stats["n_polymers"]:,} polymers; '
              f'{stats["n_polymers_in_merged_groups"]:,} polymers merged into '
              f'{stats["n_merged_groups"]:,} groups; max Tg spread '
              f'{stats["max_tg_spread_within_a_merged_group_K"]} K')
        if cap == 'hydrogen':
            merged_h = merged
    out['by_capping'] = per_cap

    # Polymers that hydrogen capping merges but methyl capping separates: the
    # cases where the shipped convention destroys real information.
    rows = []
    for key, idx in (merged_h or {}).items():
        methyl_keys = {capped_key(df.loc[i, 'smiles'], 'methyl') for i in idx}
        if len(methyl_keys) > 1:
            t = df.loc[idx, 'target']
            rows.append({
                'hydrogen_capped_molecule': key,
                'n_polymers': len(idx),
                'n_distinct_under_methyl_capping': len(methyl_keys),
                'tg_min_K': float(t.min()), 'tg_max_K': float(t.max()),
                'tg_spread_K': float(t.max() - t.min()),
                'psmiles': ' | '.join(df.loc[idx, 'smiles'].tolist()[:6]),
            })
    frame = pd.DataFrame(rows).sort_values('tg_spread_K', ascending=False)
    frame.to_csv(OUT / 's12_collapsed_groups.csv', index=False)

    out['merged_by_hydrogen_but_separated_by_methyl'] = {
        'n_groups': int(len(frame)),
        'n_polymers': int(frame['n_polymers'].sum()) if len(frame) else 0,
        'max_tg_spread_K': (float(frame['tg_spread_K'].max())
                            if len(frame) else 0.0),
        'median_tg_spread_K': (float(frame['tg_spread_K'].median())
                               if len(frame) else 0.0),
        'worst_10': frame.head(10).to_dict('records'),
    }
    # Irreducible error floor: a model on hydrogen-capped fingerprints cannot
    # distinguish members of a merged group, so its best possible prediction is
    # the group mean.
    if merged_h:
        resid = []
        for idx in merged_h.values():
            t = df.loc[idx, 'target'].values
            resid.extend(np.abs(t - t.mean()))
        out['irreducible_mae_floor_from_merging_K'] = round(
            float(np.sum(resid) / len(df)), 3)
        out['irreducible_mae_floor_note'] = (
            'Mean absolute error that no model on hydrogen-capped fingerprints '
            'can go below on this collection, because merged polymers share a '
            'feature vector and the best it can do is predict their mean.')
    dump('s12_capping_collapse.json', out)


if __name__ == '__main__':
    main()
