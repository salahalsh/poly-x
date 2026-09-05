# POLY-X release checklist for the JCAMD revision

Everything in this folder is ready to be copied into
`https://github.com/salahalsh/poly-x`. Nothing here has been pushed: the
push and the tag are yours to make.

Both referees asked for the same thing in different words. Reviewer 1: *"a
tagged release corresponding exactly to the manuscript, installation tests,
unit tests for PSMILES parsing and group matching, regression tests for the
reported numerical examples, and a reproducibility script for every figure and
table."* Reviewer 2, Major 13: *"A versioned archival release should contain all
source code, model artifacts, data hashes, canonicalized input data, split
assignments, raw outputs underlying every table and figure, automated tests, and
a one-command reproduction workflow."*

## What is in this folder

| Path | Purpose | Referee point |
|---|---|---|
| `tests/conftest.py` | Loads the service modules by file path so the suite runs with no Django app registry, and therefore in CI | R1 §1.9 |
| `tests/test_psmiles.py` | PSMILES parsing, canonicalisation, the three capping conventions, and the invariance properties (including the ones that fail) | R1 §1.9; R2 Major 7 |
| `tests/test_group_matching.py` | Atom-exclusive matching, cap-methyl decrement, determinism, coverage reporting, the fallback path, structural corrections | R1 §1.9; R2 Minor 4, 5 |
| `tests/test_regression_reported_numbers.py` | Reproduces the originally submitted metrics to 9 decimal places, and checks that every manuscript token still matches its source file | R1 §1.9 |
| `reproduce.py` | One command runs all twelve analysis steps in dependency order, resumable | R1 §1.9; R2 Major 13 |
| `.github/workflows/ci.yml` | Runs an import check and the test suite on push, across Python 3.11-3.13 | R1 §1.9 |
| `requirements-ci.txt` | Minimal dependency set for the tests | R1 §1.9 |
| `environment.lock.txt` | Exact versions used for every reported number, plus the polyBERT weight provenance note | R2 Major 13 |

## Steps to publish

1. **Copy** `tests/`, `reproduce.py`, `.github/`, `requirements-ci.txt` and
   `environment.lock.txt` into the repository root.
2. **Copy** the analysis directory (`../analysis/`) to `paper/analysis/` and
   the archived outputs (`../outputs/`) to `paper/outputs/`. The outputs are
   what make every table and figure auditable: split index files for every
   protocol and seed, the full validation and test prediction arrays for all
   72 runs, the per-polymer reliability output, and the group-contribution
   predictions for the whole collection.
3. **Verify locally** before pushing:
   ```bash
   pytest -q -m "not slow" tests/     # fast suite
   pytest -q -m slow tests/           # reproduces the submitted metrics
   python reproduce.py --list         # confirm every step reports "done"
   ```
4. **Commit** in more than one commit. Reviewer 1 noted that the repository had
   a single commit; a linear history of the revision work is itself part of the
   answer.
5. **Tag** the release to this manuscript, e.g.
   ```bash
   git tag -a v1.0.0-jcamd-r1 -m "JCAMD revision 1: analysis, tests, reproduction"
   git push origin main --tags
   ```
   Then create a GitHub Release from the tag so it has a citable archive, and
   (optionally) mint a DOI by enabling the Zenodo-GitHub integration before
   creating the release.
6. **Update** the manuscript's Code Availability statement with the tag name
   and, if minted, the Zenodo DOI. The statement currently says "with a release
   tagged to this manuscript" and needs the concrete identifier.

## Two things that still need a decision

- **Large artefacts.** `outputs/` contains the polyBERT embedding matrices
  (2 x 17.7 MB) and 72 prediction archives. These are within GitHub's limits
  but are binary; consider Git LFS, or attach them to the GitHub Release
  instead of committing them, and say which in the Data Availability
  statement.
- **Trained checkpoints.** Reviewer 2 asked for model artifacts. The Tier 2
  checkpoints live in `poly_x/trained_models/tg/checkpoints` and the Tier 3
  head in `poly_x/trained_models/polybert/heads`; the training fingerprint
  matrix is 48 MB. Attaching these to the Release rather than committing them
  keeps the clone small.

## Live platform

The PI1M/PolyMetriX contradiction Reviewer 1 found has been corrected in the
source, in eight places across six files:

- `poly_x/templates/poly_x/home.html` (tier card)
- `poly_x/templates/poly_x/includes/result_display.html` (results card)
- `poly_x/services/__init__.py`
- `poly_x/services/poly_aggregator.py`
- `poly_x/training/train_tg.py` and `poly_x/training/dataset_loaders.py`
  (docstrings; the `PI1MLoader` class name is retained for compatibility but
  its docstring now states that the shipped Tg models use PolyMetriX)

These are source edits only. **Nothing has been deployed** - the deploy is
yours to run.
