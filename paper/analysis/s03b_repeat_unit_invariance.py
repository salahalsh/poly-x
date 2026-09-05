"""
Script 03b - invariance to repeat-unit multiplication.

Reviewer 2, Major 7 asks explicitly for a test of "invariance to equivalent
repeat-unit choices and translated or multiplied repeat units". Writing
[*]A[*] as [*]AA[*] describes the *same* polymer: any representation used for
polymer property prediction should place the two at essentially the same point.

The doubling is performed with RDKit rather than by string surgery: two copies
of the repeat unit are combined, the tail dummy of the first copy is bonded to
the head dummy of the second, and the two spent dummies are deleted. The result
retains exactly two dummies and is a valid PSMILES.

Both representations under discussion are tested: ECFP4 (Tier 2, where the
effect is expected to be small because the fingerprint is local and binary) and
polyBERT CLS / mean embeddings (Tier 3).

Writes: outputs/s03b_repeat_unit_invariance.json
"""
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from polyx_rev import OUT, dump, load_dataset, to_mol

RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def multiply_repeat_unit(psmiles: str, n: int = 2) -> str | None:
    """Return the PSMILES of ``n`` concatenated copies of the repeat unit."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None

    combo = Chem.RWMol(mol)
    for _ in range(n - 1):
        offset = combo.GetNumAtoms()
        combo = Chem.RWMol(Chem.CombineMols(combo, mol))
        d = [a.GetIdx() for a in combo.GetAtoms() if a.GetAtomicNum() == 0]
        # d is ordered: [head_a, tail_a, head_b, tail_b] where b is offset.
        left = [i for i in d if i < offset]
        right = [i for i in d if i >= offset]
        if len(left) != 2 or len(right) != 2:
            return None
        tail_a, head_b = left[1], right[0]
        na = [x.GetIdx() for x in combo.GetAtomWithIdx(tail_a).GetNeighbors()]
        nb = [x.GetIdx() for x in combo.GetAtomWithIdx(head_b).GetNeighbors()]
        if not na or not nb:
            return None
        combo.AddBond(na[0], nb[0], Chem.BondType.SINGLE)
        for idx in sorted([tail_a, head_b], reverse=True):
            combo.RemoveAtom(idx)
    try:
        out = combo.GetMol()
        Chem.SanitizeMol(out)
        smi = Chem.MolToSmiles(out)
        return smi if Chem.MolFromSmiles(smi) is not None else None
    except Exception:
        return None


def ecfp(psmiles, cap='hydrogen'):
    m = to_mol(psmiles, cap=cap)
    return GEN.GetFingerprint(m) if m is not None else None


@torch.no_grad()
def embed(tok, mdl, smiles):
    enc = tok(list(smiles), padding=True, truncation=True, max_length=512,
              return_tensors='pt')
    hid = mdl(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).to(hid.dtype)
    cls = hid[:, 0, :].cpu().numpy()
    mean = ((hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
    return cls, mean


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def main():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained('xushijie/polyBERT')
    mdl = AutoModel.from_pretrained('xushijie/polyBERT')
    mdl.eval()
    torch.manual_seed(42)

    df, _ = load_dataset()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=300, replace=False)

    rows = {2: [], 3: []}
    for i in idx:
        base = df['smiles'].iloc[int(i)]
        fp0 = ecfp(base)
        if fp0 is None:
            continue
        c0, m0 = embed(tok, mdl, [base])
        for n in (2, 3):
            var = multiply_repeat_unit(base, n=n)
            if var is None:
                continue
            fpv = ecfp(var)
            if fpv is None:
                continue
            cv, mv = embed(tok, mdl, [var])
            rows[n].append({
                'tanimoto_ecfp4': float(DataStructs.TanimotoSimilarity(fp0, fpv)),
                'cls_cosine': cos(c0[0], cv[0]),
                'mean_cosine': cos(m0[0], mv[0]),
            })

    summary = {}
    for n, recs in rows.items():
        if not recs:
            continue
        summary[f'x{n}_repeat_units'] = {
            'n_pairs': len(recs),
            'ecfp4_tanimoto_mean': float(np.mean([r['tanimoto_ecfp4'] for r in recs])),
            'ecfp4_tanimoto_min': float(np.min([r['tanimoto_ecfp4'] for r in recs])),
            'polybert_cls_cosine_mean': float(np.mean([r['cls_cosine'] for r in recs])),
            'polybert_cls_cosine_min': float(np.min([r['cls_cosine'] for r in recs])),
            'polybert_mean_cosine_mean': float(np.mean([r['mean_cosine'] for r in recs])),
            'polybert_mean_cosine_min': float(np.min([r['mean_cosine'] for r in recs])),
        }
    dump('s03b_repeat_unit_invariance.json', {
        'n_probe': int(len(idx)),
        'summary': summary,
        'interpretation': (
            'A polymer written as one repeat unit and as n concatenated repeat '
            'units is the same polymer. Tanimoto/cosine below 1 measures how '
            'much each representation depends on that arbitrary choice.'),
    })
    import json
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
