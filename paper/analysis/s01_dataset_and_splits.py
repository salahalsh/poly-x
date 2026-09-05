"""
Script 01 - dataset accounting and split characterisation.

Answers:
  * Reviewer 2, Minor 1  - which records are removed at each cleaning stage.
  * Reviewer 2, Minor 2  - the number and size distribution of scaffold groups
                           (the submitted text wrongly implied one scaffold per
                           polymer).
  * Reviewer 2, Major 6  - which partition receives the null scaffold, the
                           chemical-family and target distribution per
                           partition, and where the eight canonical polymers
                           land.
  * Reviewer 2, Minor 6  - are the eight canonical validation polymers present
                           in the training data?

Writes: outputs/s01_dataset_accounting.json
        outputs/s01_scaffold_stats.json
        outputs/s01_split_indices_seed<N>.npz
        outputs/s01_cluster_split.npz
"""
from collections import Counter

import numpy as np
from rdkit import Chem

from polyx_rev import (OUT, SEEDS, butina_clusters, canonical_psmiles,
                       cluster_split, dump, load_dataset, murcko_scaffold,
                       scaffold_groups, scaffold_split, to_mol)

# The eight canonical homopolymers of Table 1 / Table 4.
CANONICAL = {
    'PE':   '[*]CC[*]',
    'PP':   '[*]CC(C)[*]',
    'PS':   '[*]CC(c1ccccc1)[*]',
    'PVC':  '[*]CC(Cl)[*]',
    'PET':  '[*]CCOC(=O)c1ccc(C(=O)O[*])cc1',
    'PEO':  '[*]CCO[*]',
    'POM':  '[*]CO[*]',
    'PVOH': '[*]CC(O)[*]',
}

# Coarse chemical families, matched in priority order on the H-capped repeat
# unit. First match wins; anything unmatched is "other".
FAMILY_SMARTS = [
    ('fluoropolymer',  '[CX4][F]'),
    ('siloxane',       '[Si][OX2][Si]'),
    ('polyimide',      'O=C1[NX3]C(=O)c2ccccc21'),
    ('polyamide',      'C(=O)[NX3H]'),
    ('polyurethane',   '[NX3][CX3](=O)[OX2]'),
    ('polycarbonate',  '[OX2][CX3](=O)[OX2]'),
    ('polyester',      '[CX3](=O)[OX2][#6]'),
    ('polysulfone',    'S(=O)(=O)'),
    ('nitrile',        '[NX1]#[CX2]'),
    ('polyether',      '[#6][OX2][#6]'),
    ('halogenated',    '[Cl,Br,I]'),
    ('hydroxyl',       '[OX2H]'),
    ('aromatic_hc',    'c1ccccc1'),
    ('heterocycle',    '[a;!c]'),
]
_FAM = [(n, Chem.MolFromSmarts(s)) for n, s in FAMILY_SMARTS]


def family(psmiles: str) -> str:
    m = to_mol(psmiles, cap='hydrogen')
    if m is None:
        return 'unparseable'
    for name, patt in _FAM:
        if patt is not None and m.HasSubstructMatch(patt):
            return name
    return 'aliphatic_hc'


def main():
    df, log = load_dataset()
    print(f"raw={log.raw} after_nan={log.after_nan} after_parse={log.after_parse} "
          f"after_canonical_dedup={log.after_canonical_dedup}")

    # ---- Reviewer 2, Minor 1: reconcile against the submitted pipeline -----
    # The submitted pipeline deduplicated on the RAW string and applied a
    # global 3-sigma target filter before splitting. Reproduce both counts so
    # the two accountings can be compared line by line.
    import pandas as pd
    raw = pd.read_csv('D:/myTools/Tool - InsilicoX Project/insilicox_web_app/'
                      'poly_x/data/tg_training_data.csv')
    raw = raw.rename(columns={'PSMILES': 'smiles', 'Tg': 'target'})
    raw['target'] = pd.to_numeric(raw['target'], errors='coerce')
    r = raw.dropna(subset=['smiles', 'target']).copy()
    n_nan = len(r)
    r['ok'] = r['smiles'].apply(
        lambda s: Chem.MolFromSmiles(str(s).replace('[*]', '[H]')) is not None)
    r = r[r['ok']]
    n_valid = len(r)
    mu, sd = r['target'].mean(), r['target'].std()
    r_out = r[(r['target'] >= mu - 3 * sd) & (r['target'] <= mu + 3 * sd)]
    n_sigma = len(r_out)
    dropped_sigma = r[~r.index.isin(r_out.index)][['smiles', 'target']]
    r_dedup = r_out.drop_duplicates(subset=['smiles'], keep='first')
    n_dedup = len(r_dedup)
    dup_raw = r_out[r_out.duplicated(subset=['smiles'], keep='first')][
        ['smiles', 'target']]

    accounting = {
        'as_submitted': {
            'raw_rows': int(len(raw)),
            'after_nan_drop': int(n_nan),
            'after_psmiles_validation': int(n_valid),
            'after_global_3sigma': int(n_sigma),
            'after_raw_string_dedup': int(n_dedup),
            'rows_removed_by_3sigma': dup_or_list(dropped_sigma),
            'rows_removed_by_raw_dedup': dup_or_list(dup_raw),
            'target_mean_used_for_sigma': float(mu),
            'target_sd_used_for_sigma': float(sd),
        },
        'revised_pipeline': {
            'raw_rows': log.raw,
            'after_nan_drop': log.after_nan,
            'after_canonical_parse': log.after_parse,
            'after_canonical_dedup': log.after_canonical_dedup,
            'unparseable': list(log.removed_unparseable),
            'canonical_duplicates_removed': [
                {'smiles': s, 'target': float(t)} for s, t in log.removed_duplicates],
            'note': ('The 3-sigma target filter is target-dependent and is '
                     'therefore no longer applied globally; it is fitted on '
                     'the training partition only (see script 02).'),
        },
        'target_summary_revised': {
            'n': int(len(df)),
            'min': float(df['target'].min()),
            'max': float(df['target'].max()),
            'mean': float(df['target'].mean()),
            'sd': float(df['target'].std(ddof=1)),
        },
    }
    dump('s01_dataset_accounting.json', accounting)

    # ---- scaffold statistics (Reviewer 2, Minor 2) -------------------------
    groups = scaffold_groups(df, cap='hydrogen')
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    null_size = len(groups.get('', []))
    stats = {
        'n_polymers': int(len(df)),
        'n_scaffold_groups': int(len(groups)),
        'n_singleton_groups': int(sum(1 for s in sizes if s == 1)),
        'largest_group_size': int(sizes[0]),
        'null_scaffold_group_size': int(null_size),
        'null_scaffold_fraction': float(null_size / len(df)),
        'size_distribution_top20': sizes[:20],
        'size_histogram': dict(Counter(sizes)),
        'median_group_size': float(np.median(sizes)),
        'note': ('The submitted manuscript stated that the procedure '
                 '"identified distinct Murcko scaffolds for each polymer". '
                 'That is incorrect: polymers are deliberately grouped, and '
                 'all acyclic repeat units share the empty (null) scaffold.'),
    }

    # ---- per-seed split characterisation -----------------------------------
    df = df.copy()
    df['family'] = df['smiles'].apply(family)
    df['scaffold'] = df['smiles'].apply(lambda s: murcko_scaffold(s, cap='hydrogen'))

    canon_canonical = {k: canonical_psmiles(v) for k, v in CANONICAL.items()}
    canon_scaffold = {k: murcko_scaffold(v, cap='hydrogen') for k, v in CANONICAL.items()}
    canon_in_data = {}
    for name, cs in canon_canonical.items():
        hit = df.index[df['canonical'] == cs].tolist()
        canon_in_data[name] = {
            'canonical_psmiles': cs,
            'murcko_scaffold': canon_scaffold[name] or '(null / acyclic)',
            'present_in_dataset': bool(hit),
            'dataset_row': int(hit[0]) if hit else None,
            'dataset_target_K': float(df.loc[hit[0], 'target']) if hit else None,
        }

    per_seed = {}
    for seed in SEEDS:
        tr, va, te = scaffold_split(df, seed=seed, groups=groups)
        np.savez(OUT / f's01_split_indices_seed{seed}.npz',
                 train=tr, val=va, test=te)
        null_idx = set(groups.get('', []))
        where_null = ('train' if null_idx & set(tr.tolist())
                      else 'val' if null_idx & set(va.tolist())
                      else 'test' if null_idx & set(te.tolist()) else 'none')
        part = {}
        for label, idx in (('train', tr), ('val', va), ('test', te)):
            sub = df.loc[idx]
            part[label] = {
                'n': int(len(idx)),
                'target_mean': float(sub['target'].mean()),
                'target_sd': float(sub['target'].std(ddof=1)),
                'target_min': float(sub['target'].min()),
                'target_max': float(sub['target'].max()),
                'n_scaffold_groups': int(sub['scaffold'].nunique()),
                'family_counts': {k: int(v) for k, v in
                                  sub['family'].value_counts().items()},
            }
        # canonical polymer membership
        canon_part = {}
        for name, info in canon_in_data.items():
            row = info['dataset_row']
            canon_part[name] = (
                'absent' if row is None else
                'train' if row in set(tr.tolist()) else
                'val' if row in set(va.tolist()) else
                'test' if row in set(te.tolist()) else 'dropped')
        per_seed[str(seed)] = {
            'partitions': part,
            'null_scaffold_partition': where_null,
            'canonical_polymer_partition': canon_part,
            'scaffold_disjoint': bool(
                not (set(df.loc[tr, 'scaffold']) & set(df.loc[te, 'scaffold']))),
        }

    stats['per_seed'] = per_seed
    stats['canonical_polymers'] = canon_in_data
    dump('s01_scaffold_stats.json', stats)

    # ---- polymer-aware clustering (Reviewer 2, Major 6) --------------------
    print("clustering (Butina, ECFP4 Tanimoto, cutoff 0.65) ...")
    clusters = butina_clusters(df, cutoff=0.65)
    csizes = sorted((len(c) for c in clusters), reverse=True)
    ctr, cva, cte = cluster_split(df, clusters, seed=42)
    np.savez(OUT / 's01_cluster_split.npz', train=ctr, val=cva, test=cte)
    dump('s01_cluster_stats.json', {
        'n_clusters': len(clusters),
        'cutoff_distance': 0.65,
        'largest_cluster': csizes[0],
        'n_singletons': int(sum(1 for s in csizes if s == 1)),
        'median_cluster_size': float(np.median(csizes)),
        'size_distribution_top20': csizes[:20],
        'split_sizes': {'train': int(len(ctr)), 'val': int(len(cva)),
                        'test': int(len(cte))},
    })
    print("done.")


def dup_or_list(frame):
    return [{'smiles': s, 'target': float(t)}
            for s, t in frame.itertuples(index=False, name=None)]


if __name__ == '__main__':
    main()
