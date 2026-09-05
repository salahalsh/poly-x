"""
Script 03 - polyBERT representation: CLS versus mean pooling, and invariance.

Answers Reviewer 2, Major 2:
  * the submitted Tier 3 used the final-layer CLS vector, whereas the published
    polyBERT fingerprint is the *mean* over the token dimension. Both are
    extracted here for every polymer so the two can be compared under identical
    splits and heads (script 02).
  * canonicalisation of equivalent PSMILES is tested explicitly: identical
    polymers written differently must give the same fingerprint if the
    representation is to be trusted.
  * the description of the pre-training corpus is corrected from the model card.

Provenance note: the original ``kuelumbus/polyBERT`` repository is no longer
resolvable on the Hugging Face Hub. A public mirror is used instead and is
**verified** by checking that its CLS vectors reproduce the embedding matrix
stored with the submitted models (``all_embeddings_scaffold.npy``) to numerical
tolerance. If the check fails the script aborts rather than proceeding.

Writes: outputs/s03_embeddings_cls.npy      (N x 600, dataset row order)
        outputs/s03_embeddings_mean.npy     (N x 600, dataset row order)
        outputs/s03_mirror_verification.json
        outputs/s03_invariance.json
"""
import json
import sys

import numpy as np
import torch
from rdkit import Chem

from polyx_rev import OUT, PB_MODELS, canonical_psmiles, dump, load_dataset

MIRRORS = ["xushijie/polyBERT", "HAYDERphd/polyBERT", "kuelumbus/polyBERT"]
MAXLEN = 512
BATCH = 64


def load_model():
    from transformers import AutoTokenizer, AutoModel
    last = None
    for name in MIRRORS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            mdl.eval()
            torch.manual_seed(42)
            np.random.seed(42)
            print(f"[model] loaded {name}: hidden={mdl.config.hidden_size} "
                  f"layers={mdl.config.num_hidden_layers} "
                  f"params={sum(p.numel() for p in mdl.parameters())/1e6:.1f}M")
            return name, tok, mdl
        except Exception as e:  # noqa: BLE001
            print(f"[model] {name} unavailable: {type(e).__name__}: {e}")
            last = e
    raise SystemExit(f"no polyBERT source reachable: {last}")


@torch.no_grad()
def embed(tok, mdl, smiles, batch=BATCH):
    """Return (cls, mean) embedding matrices.

    ``mean`` is the attention-mask-weighted average over the token dimension of
    the final hidden state, i.e. the pooling used by the published polyBERT
    fingerprint. ``cls`` is the first-position vector, i.e. what the submitted
    Tier 3 used.
    """
    cls_out, mean_out = [], []
    for i in range(0, len(smiles), batch):
        chunk = list(smiles[i:i + batch])
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors='pt')
        hid = mdl(**enc).last_hidden_state             # (B, T, H)
        mask = enc['attention_mask'].unsqueeze(-1).to(hid.dtype)
        cls_out.append(hid[:, 0, :].cpu().numpy())
        mean_out.append(((hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
                        .cpu().numpy())
        if (i // batch) % 20 == 0:
            print(f"  embedded {min(i + batch, len(smiles))}/{len(smiles)}",
                  flush=True)
    return np.vstack(cls_out), np.vstack(mean_out)


# --------------------------------------------------------------------------
# Equivalent-PSMILES generators for the invariance test
# --------------------------------------------------------------------------
def rotated_smiles(psmiles, n=4):
    """Alternative valid SMILES writings of the same repeat unit."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return []
    out, seen = [], {psmiles}
    for atom_idx in range(min(n * 3, mol.GetNumAtoms())):
        try:
            s = Chem.MolToSmiles(mol, rootedAtAtom=atom_idx, canonical=False)
        except Exception:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out


def doubled_repeat_unit(psmiles):
    """Two copies of the repeat unit written as one longer unit.

    ``[*]A[*]`` -> ``[*]AA[*]``. Physically the same polymer, so a
    polymer-aware representation should place them close together.
    """
    s = psmiles.strip()
    if s.count('[*]') != 2:
        return None
    body = s.replace('[*]', '', 1)
    head, _, tail = body.rpartition('[*]')
    if not head:
        return None
    return f'[*]{head}{head}[*]{tail}' if tail else f'[*]{head}{head}[*]'


def cosine(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return float(np.sum(a * b, axis=-1) / np.clip(na * nb, 1e-12, None))


def main():
    df, _ = load_dataset()
    smiles = df['smiles'].tolist()
    name, tok, mdl = load_model()

    # ---- verification against the stored embeddings ------------------------
    stored_path = PB_MODELS / 'all_embeddings_scaffold.npy'
    order_path = PB_MODELS / 'all_smiles_scaffold_order.txt'
    verification = {'mirror': name, 'verified': False}
    if stored_path.exists() and order_path.exists():
        stored = np.load(stored_path)
        with open(order_path, encoding='utf-8') as f:
            stored_smiles = [l.strip() for l in f if l.strip()]
        k = min(256, len(stored_smiles))
        probe = stored_smiles[:k]
        cls_probe, _ = embed(tok, mdl, probe)
        diff = np.abs(cls_probe - stored[:k])
        cos = np.mean([cosine(cls_probe[i], stored[i]) for i in range(k)])
        verification.update({
            'n_probe': k,
            'max_abs_diff': float(diff.max()),
            'mean_abs_diff': float(diff.mean()),
            'mean_cosine': float(cos),
            'stored_shape': list(stored.shape),
            'verified': bool(cos > 0.999 and diff.max() < 1e-2),
        })
        print(f"[verify] mean cosine to stored CLS = {cos:.6f}, "
              f"max|diff| = {diff.max():.3e}")
        if not verification['verified']:
            print("[verify] WARNING: mirror does not reproduce the stored "
                  "embeddings. Downstream Tier 3 numbers would not be "
                  "comparable with the submitted ones.")
    dump('s03_mirror_verification.json', verification)
    if not verification.get('verified'):
        print("Aborting: mirror unverified. Re-run once a verified source is "
              "available.")
        sys.exit(2)

    # ---- full extraction ---------------------------------------------------
    print(f"[embed] {len(smiles)} polymers, CLS and mean pooling ...")
    cls, mean = embed(tok, mdl, smiles)
    np.save(OUT / 's03_embeddings_cls.npy', cls.astype(np.float32))
    np.save(OUT / 's03_embeddings_mean.npy', mean.astype(np.float32))
    print(f"[embed] wrote {cls.shape} CLS and {mean.shape} mean embeddings")

    # ---- invariance to equivalent PSMILES writings -------------------------
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(len(smiles), size=200, replace=False)
    rows = []
    for i in probe_idx:
        base = smiles[int(i)]
        variants = {'canonical': canonical_psmiles(base)}
        for j, s in enumerate(rotated_smiles(base, n=2)):
            variants[f'rotated_{j}'] = s
        dbl = doubled_repeat_unit(base)
        if dbl and Chem.MolFromSmiles(dbl) is not None:
            variants['doubled_repeat_unit'] = dbl
        variants = {k: v for k, v in variants.items() if v and v != base}
        if not variants:
            continue
        keys = list(variants)
        c0, m0 = embed(tok, mdl, [base], batch=1)
        cv, mv = embed(tok, mdl, [variants[k] for k in keys], batch=8)
        for k, cvi, mvi in zip(keys, cv, mv):
            rows.append({'kind': k,
                         'cls_cosine': cosine(c0[0], cvi),
                         'mean_cosine': cosine(m0[0], mvi),
                         'cls_l2': float(np.linalg.norm(c0[0] - cvi)),
                         'mean_l2': float(np.linalg.norm(m0[0] - mvi))})

    summary = {}
    for kind in sorted({r['kind'] for r in rows}):
        sub = [r for r in rows if r['kind'] == kind]
        summary[kind] = {
            'n': len(sub),
            'cls_cosine_mean': float(np.mean([r['cls_cosine'] for r in sub])),
            'cls_cosine_min': float(np.min([r['cls_cosine'] for r in sub])),
            'mean_cosine_mean': float(np.mean([r['mean_cosine'] for r in sub])),
            'mean_cosine_min': float(np.min([r['mean_cosine'] for r in sub])),
        }
    dump('s03_invariance.json', {
        'model': name,
        'n_probe_polymers': int(len(probe_idx)),
        'summary': summary,
        'note': ('cosine of 1.0 means the representation is invariant to the '
                 'rewriting; values below 1 quantify how much the embedding '
                 'depends on the arbitrary choice of SMILES writing rather '
                 'than on the chemistry.'),
    })
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
