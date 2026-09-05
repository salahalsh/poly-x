"""
Script 06 - the PolyMetriX native hierarchical descriptors as a baseline.

Reviewer 2, Major 8: "Because the work uses the PolyMetriX dataset, the native
hierarchical full-polymer, backbone, and side-chain descriptors provided with
that ecosystem are particularly relevant baselines."

This builds exactly those descriptors - every chemical featurizer in
``polymetrix.featurizers.chemical_featurizer`` applied at the full-polymer,
backbone and side-chain levels, plus the polymer-specific topological
featurizers - and stores the matrix so that script 02 can evaluate it under the
identical partitions and identical RF/GB heads as ECFP4 and polyBERT.

polymetrix 0.2.0 declares ``requires-python <3.13`` and this environment is
3.13.7, so the wheel is vendored (extracted, not pip-installed) rather than
resolved by pip; only ``networkx``, ``numpy``, ``pandas`` and ``rdkit`` are
actually needed by the featurizers used here, and all are present.

Writes: outputs/s06_polymetrix_descriptors.npy
        outputs/s06_polymetrix_feature_labels.json
"""
import sys
from pathlib import Path

import numpy as np

VENDOR = Path(r"D:\tmp\pmx")
sys.path.insert(0, str(VENDOR))

from polymetrix.featurizers import chemical_featurizer as cf   # noqa: E402
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer  # noqa: E402
from polymetrix.featurizers.polymer import Polymer             # noqa: E402
from polymetrix.featurizers.sidechain_backbone_featurizer import (  # noqa: E402
    BackBoneFeaturizer, FullPolymerFeaturizer, NumBackBoneFeaturizer,
    NumSideChainFeaturizer, SideChainFeaturizer, SidechainDiversityFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer)

from polyx_rev import OUT, dump, load_dataset  # noqa: E402

# Every scalar chemical featurizer shipped with the package.
CHEM_CLASSES = [
    cf.NumHBondDonors, cf.NumHBondAcceptors, cf.NumRotatableBonds,
    cf.NumRings, cf.NumNonAromaticRings, cf.NumAromaticRings, cf.NumAtoms,
    cf.TopologicalSurfaceArea, cf.FractionBicyclicRings, cf.NumAliphaticHeterocycles,
    cf.SlogPVSA1, cf.BalabanJIndex, cf.MolecularWeight, cf.Sp3CarbonCountFeaturizer,
    cf.Sp2CarbonCountFeaturizer, cf.MaxEStateIndex, cf.SmrVSA5, cf.FpDensityMorgan1,
    cf.HalogenCounts, cf.BondCounts, cf.BridgingRingsCount, cf.MaxRingSize,
    cf.HeteroatomDensity, cf.HeteroatomCount,
]


def build_featurizer():
    feats = []
    for klass in CHEM_CLASSES:
        try:
            calc_full = klass()
            feats.append(FullPolymerFeaturizer(calc_full))
            feats.append(BackBoneFeaturizer(klass()))
            feats.append(SideChainFeaturizer(klass(agg=['sum', 'mean', 'max'])))
        except Exception as e:  # noqa: BLE001
            print(f'  [skip] {klass.__name__}: {type(e).__name__}: {e}')
    feats += [
        NumSideChainFeaturizer(),
        NumBackBoneFeaturizer(),
        SidechainDiversityFeaturizer(),
        SidechainLengthToStarAttachmentDistanceRatioFeaturizer(agg=['mean', 'max']),
        StarToSidechainMinDistanceFeaturizer(agg=['mean', 'min']),
    ]
    return MultipleFeaturizer(feats)


def main():
    df, _ = load_dataset()
    mf = build_featurizer()

    # Probe one polymer to learn the vector width and the labels.
    width, labels = None, None
    for smi in df['smiles']:
        try:
            v = mf.featurize(Polymer.from_psmiles(smi))
            width, labels = len(v), mf.feature_labels()
            break
        except Exception:
            continue
    if width is None:
        raise SystemExit('could not featurise any polymer')
    print(f'[featurizer] {width} descriptors '
          f'({len(CHEM_CLASSES)} chemical x 3 levels + topological)')

    X = np.zeros((len(df), width), dtype=np.float64)
    failed = []
    for i, smi in enumerate(df['smiles']):
        try:
            v = mf.featurize(Polymer.from_psmiles(smi))
            if len(v) == width:
                X[i] = v
            else:
                failed.append((i, f'width {len(v)}'))
        except Exception as e:  # noqa: BLE001
            failed.append((i, type(e).__name__))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(df)} ({len(failed)} failed)', flush=True)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(OUT / 's06_polymetrix_descriptors.npy', X.astype(np.float32))
    dump('s06_polymetrix_feature_labels.json', {
        'n_features': int(width),
        'n_polymers': int(len(df)),
        'n_failed': len(failed),
        'failure_reasons': dict(_count([r for _, r in failed])),
        'labels': labels,
        'source': 'polymetrix 0.2.0 (vendored wheel)',
    })
    print(f'[write] descriptors {X.shape}, {len(failed)} failures')


def _count(xs):
    from collections import Counter
    return Counter(xs).most_common()


if __name__ == '__main__':
    main()
