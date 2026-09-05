"""PSMILES parsing, canonicalisation and capping.

Reviewer 1 asked for "unit tests for PSMILES parsing and group matching".
Reviewer 2, Major 7 asked that the capping conventions be made explicit and
that invariance to equivalent repeat-unit writings be tested. These tests pin
the behaviour that the revision relies on, including the behaviours that are
known to be imperfect: where a representation is *not* invariant, the test
records the fact rather than asserting a property the code does not have.
"""
import numpy as np
import pytest
from rdkit import Chem

from conftest import CANONICAL


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize('name,psmiles', [(k, v[0]) for k, v in CANONICAL.items()])
def test_canonical_psmiles_parses_with_wildcards_intact(name, psmiles):
    """RDKit must parse PSMILES directly, keeping [*] as dummy atoms.

    This is the single canonicalisation the revised pipeline depends on: it
    must not require a capping choice to be made first.
    """
    mol = Chem.MolFromSmiles(psmiles)
    assert mol is not None, f'{name} failed to parse'
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(dummies) == 2, f'{name}: expected 2 wildcards, got {len(dummies)}'


def test_canonicalisation_is_idempotent():
    for _, (psmiles, _) in CANONICAL.items():
        once = Chem.MolToSmiles(Chem.MolFromSmiles(psmiles))
        twice = Chem.MolToSmiles(Chem.MolFromSmiles(once))
        assert once == twice


def test_canonicalisation_collapses_equivalent_writings():
    """Two different writings of polypropylene must give one canonical form."""
    a = Chem.MolToSmiles(Chem.MolFromSmiles('[*]CC(C)[*]'))
    b = Chem.MolToSmiles(Chem.MolFromSmiles('[*]C(C)C[*]'))
    assert a == b


def test_unparseable_input_returns_none_not_an_exception():
    assert Chem.MolFromSmiles('[*]C(((C[*]') is None


# --------------------------------------------------------------------------
# Capping
# --------------------------------------------------------------------------
def test_methyl_and_hydrogen_capping_give_different_molecules():
    """The three tiers use different capping rules; that is a real difference
    and the code must not silently treat them as interchangeable."""
    psmiles = '[*]CC[*]'
    methyl = Chem.MolFromSmiles(psmiles.replace('[*]', '[CH3]'))
    hydrogen = Chem.MolFromSmiles(psmiles.replace('[*]', '[H]'))
    assert methyl is not None and hydrogen is not None
    assert methyl.GetNumHeavyAtoms() == 4      # C-C-C-C
    assert hydrogen.GetNumHeavyAtoms() == 2    # C-C
    assert Chem.MolToSmiles(methyl) != Chem.MolToSmiles(hydrogen)


def test_tier1_subtracts_the_methyl_caps_from_the_repeat_unit_mass(gc_calc):
    """Polyethylene's repeat unit is C2H4, 28.05 Da, not the capped butane."""
    r = gc_calc.calculate('[*]CC[*]')
    assert r.success
    assert r.repeat_unit_mw == pytest.approx(28.05, abs=0.1)


def test_tier2_fingerprinter_uses_hydrogen_capping(fingerprinter):
    """Pins the shipped Tier 2 convention. The manuscript previously claimed
    methyl capping everywhere; this test makes the real behaviour explicit so
    that a future change cannot be silent."""
    fp_service = fingerprinter.featurize('[*]CC[*]')
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_hydrogen = gen.GetFingerprintAsNumPy(
        Chem.MolFromSmiles('[*]CC[*]'.replace('[*]', '[H]'))).astype(np.float32)
    assert np.array_equal(fp_service, fp_hydrogen)


# --------------------------------------------------------------------------
# Invariance (documented, including where it fails)
# --------------------------------------------------------------------------
def test_fingerprint_is_invariant_to_smiles_rewriting(fingerprinter):
    """Two writings of the same molecule must give the same fingerprint."""
    a = fingerprinter.featurize('[*]CC(C)[*]')
    b = fingerprinter.featurize('[*]C(C)C[*]')
    assert np.array_equal(a, b)


def test_fingerprint_is_not_invariant_to_repeat_unit_multiplication(fingerprinter):
    """[*]CC[*] and [*]CCCC[*] are the same polymer but not the same molecule.

    This asserts the limitation rather than a capability: the manuscript
    reports the size of this effect (Figure 7), and the test exists so that the
    reported behaviour cannot drift unnoticed.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    one = gen.GetFingerprint(Chem.MolFromSmiles('[*]CC[*]'.replace('[*]', '[H]')))
    two = gen.GetFingerprint(Chem.MolFromSmiles('[*]CCCC[*]'.replace('[*]', '[H]')))
    tanimoto = DataStructs.TanimotoSimilarity(one, two)
    assert tanimoto < 1.0, ('if this ever becomes 1.0 the fingerprint has '
                            'become repeat-unit invariant and Figure 7 must '
                            'be regenerated')
