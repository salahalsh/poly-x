"""Atom-exclusive group matching, coverage reporting and the fallback path.

Reviewer 1 asked for unit tests of group matching. Reviewer 2, Minor 4 and
Minor 5 asked that atom-level coverage be returned for every query so that a
partial decomposition cannot silently produce an apparently precise number, and
that the behaviour on unsupported structures be stated explicitly. These tests
pin both.
"""
import pytest

from conftest import CANONICAL


def coverage(calc, psmiles):
    """Fraction of heavy atoms consumed by a matched group."""
    mol, _ = calc._parse_psmiles(psmiles)
    assert mol is not None
    consumed = set()
    for pattern, *_ in calc._group_patterns:
        for match in sorted(mol.GetSubstructMatches(pattern),
                            key=lambda m: m[0] if m else 0):
            s = set(match)
            if not s & consumed:
                consumed |= s
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return 1.0 - len([i for i in heavy if i not in consumed]) / len(heavy)


# --------------------------------------------------------------------------
# Atom-exclusive matching
# --------------------------------------------------------------------------
def test_ester_is_matched_as_one_group_not_carbonyl_plus_ether(gc_calc):
    """The priority scheme exists to stop an ester being double-counted."""
    r = gc_calc.calculate('[*]CCOC(=O)c1ccc(C(=O)O[*])cc1')       # PET
    assert r.success
    names = ' '.join(r.groups_found)
    assert 'Ester' in names
    # If the composite match failed, the constituent atoms would be picked up
    # separately by the lower-priority carbonyl and ether patterns.
    carbonyl = sum(v for k, v in r.groups_found.items()
                   if 'Carbonyl' in k and 'Ester' not in k)
    assert carbonyl == 0, (f'ester atoms leaked into a carbonyl match: '
                           f'{r.groups_found}')


def test_methyl_count_is_decremented_by_the_caps(gc_calc):
    """Polyethylene has no methyl group; its two caps must not be counted."""
    r = gc_calc.calculate('[*]CC[*]')
    assert r.success
    # Exact key: 'Methylene (-CH2-)' also starts with 'Methyl'.
    methyls = r.groups_found.get('Methyl (-CH3)', 0)
    assert methyls == 0, f'cap methyls were counted: {r.groups_found}'
    assert r.groups_found.get('Methylene (-CH2-)') == 2, r.groups_found


def test_polypropylene_keeps_its_real_methyl(gc_calc):
    r = gc_calc.calculate('[*]CC(C)[*]')
    assert r.success
    methyls = r.groups_found.get('Methyl (-CH3)', 0)
    assert methyls == 1, f'expected exactly one methyl: {r.groups_found}'


def test_phenylene_matched_as_a_ring_not_as_loose_carbons(gc_calc):
    r = gc_calc.calculate('[*]CC(c1ccccc1)[*]')                   # PS
    assert r.success
    assert any('henyl' in k for k in r.groups_found), r.groups_found


def test_matching_is_deterministic(gc_calc):
    """Substructure match order is arbitrary in RDKit; the sort by lowest atom
    index is what makes the decomposition reproducible across platforms."""
    first = gc_calc.calculate('[*]CCOC(=O)c1ccc(C(=O)O[*])cc1').groups_found
    for _ in range(5):
        assert gc_calc.calculate(
            '[*]CCOC(=O)c1ccc(C(=O)O[*])cc1').groups_found == first


# --------------------------------------------------------------------------
# Coverage and the fallback
# --------------------------------------------------------------------------
@pytest.mark.parametrize('name,psmiles', [(k, v[0]) for k, v in CANONICAL.items()])
def test_canonical_polymers_are_fully_covered(gc_calc, name, psmiles):
    """The eight benchmark structures are exactly the regime where Tier 1 is
    calibrated, so all of their heavy atoms should be matched."""
    assert coverage(gc_calc, psmiles) == pytest.approx(1.0), name


def test_an_unsupported_element_leaves_atoms_unmatched(gc_calc):
    """A structure outside the group library must report reduced coverage
    rather than silently returning an apparently precise number."""
    cov = coverage(gc_calc, '[*]c1ccc(-c2nc3ccc([*])cc3s2)cc1')   # benzothiazole
    assert cov < 1.0


# --------------------------------------------------------------------------
# Coverage is returned by the service itself, not only by the analysis scripts
# --------------------------------------------------------------------------
@pytest.mark.parametrize('name,psmiles', [(k, v[0]) for k, v in CANONICAL.items()])
def test_service_reports_full_coverage_for_canonical_polymers(gc_calc, name, psmiles):
    r = gc_calc.calculate(psmiles)
    assert r.atom_coverage == pytest.approx(1.0), name
    assert r.n_unmatched_atoms == 0
    assert r.unmatched_elements == []
    assert r.used_fallback is False


def test_service_reports_partial_coverage_and_warns(gc_calc):
    """Reviewer 2, Minor 4: a partial decomposition must not silently produce
    an apparently precise prediction."""
    r = gc_calc.calculate('[*]c1ccc(-c2nc3ccc([*])cc3s2)cc1')
    assert r.success
    assert 0.0 < r.atom_coverage < 1.0
    assert r.n_unmatched_atoms > 0
    assert r.unmatched_elements
    assert any('not covered by the group library' in w for w in r.warnings), \
        r.warnings


def test_coverage_is_exposed_in_the_serialised_result(gc_calc):
    """The web layer renders from to_dict(), so the fields must survive it."""
    d = gc_calc.calculate('[*]c1ccc(-c2nc3ccc([*])cc3s2)cc1').to_dict()
    for key in ('gc_atom_coverage', 'gc_n_unmatched_atoms',
                'gc_unmatched_elements', 'gc_used_fallback'):
        assert key in d, f'{key} missing from to_dict()'
    assert 0.0 < d['gc_atom_coverage'] < 1.0


def test_service_coverage_matches_the_independent_recomputation(gc_calc):
    """The value the platform shows must equal the one the paper reports."""
    for _, (psmiles, _) in CANONICAL.items():
        r = gc_calc.calculate(psmiles)
        assert r.atom_coverage == pytest.approx(coverage(gc_calc, psmiles))


def test_adding_coverage_did_not_change_any_prediction(gc_calc):
    """Guards against the coverage walk perturbing the decomposition.

    The reference values are the archived analysis output, not the submitted
    manuscript's Table 1. Those two disagree: for PVC, PET, PEO and PVOH the
    submitted table lists a predicted value whose error has the right magnitude
    but the opposite sign to what the code produces, while the mean absolute
    error happens to agree. The revised manuscript's tier table is generated
    from the code, so it is self-consistent by construction; this test pins the
    code against its own archived output.
    """
    import json
    from pathlib import Path
    from test_regression_reported_numbers import _find_outputs
    out = _find_outputs() / 's04_gc_summary.json'
    if not out.exists():
        pytest.skip('analysis output not present')
    with open(out, encoding='utf-8') as f:
        canon = json.load(f)['canonical_eight']
    for name, rec in canon.items():
        got = gc_calc.calculate(rec['psmiles']).tg
        assert got == pytest.approx(rec['gc_predicted_tg_K'], abs=0.05), name


def test_fallback_is_flagged_for_a_structure_with_no_matchable_group(gc_calc):
    """When nothing matches, Tier 1 falls back to Tg = 200 + 0.5M and the
    output must say so (Reviewer 2, Minor 5)."""
    r = gc_calc.calculate('[*][Se][*]')
    if r.success and r.groups_found:
        pytest.skip('this structure now matches a group; pick another probe')
    assert any('No functional groups matched' in w for w in (r.warnings or [])), \
        f'fallback was used without a warning: {r.warnings}'


def test_fallback_matches_its_documented_formula(gc_calc):
    r = gc_calc.calculate('[*][Se][*]')
    if not any('No functional groups matched' in w for w in (r.warnings or [])):
        pytest.skip('probe structure no longer triggers the fallback')
    assert r.tg == pytest.approx(200.0 + 0.5 * r.repeat_unit_mw, rel=1e-6)


# --------------------------------------------------------------------------
# Structural corrections
# --------------------------------------------------------------------------
def test_perfluoro_correction_lowers_ptfe(gc_calc):
    r = gc_calc.calculate('[*]C(F)(F)C(F)(F)[*]')
    assert r.success
    assert any('fluor' in c.lower() for c in (r.corrections_applied or [])), \
        r.corrections_applied
    assert r.tg < 220, f'perfluoro correction did not apply: Tg = {r.tg}'


def test_alpha_methyl_correction_raises_pmma(gc_calc):
    r = gc_calc.calculate('[*]CC(C)(C(=O)OC)[*]')
    assert r.success
    assert r.corrections_applied, 'expected a steric correction for PMMA'
    assert r.tg > 340, f'steric correction did not apply: Tg = {r.tg}'
