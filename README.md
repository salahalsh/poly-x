# POLY-X: Multi-Tier Polymer Property Prediction Platform

POLY-X returns three independent estimates of a polymer's glass transition
temperature (Tg) for the same query, each with a reliability assessment
calibrated against observed error.

| Tier | Method | Representation |
|------|--------|----------------|
| 1 | Van Krevelen group contribution | 25 SMARTS-defined groups, atom-exclusive matching |
| 2 | Random Forest + Gradient Boosting ensemble | Morgan fingerprints (ECFP4, 2048-bit) |
| 3 | Frozen polyBERT embeddings + RF/GB heads | 600-dimensional mean-pooled embeddings |

This repository accompanies the revised manuscript submitted to the *Journal of
Computer-Aided Molecular Design*. The tag matching that manuscript is
`v1.0.0-jcamd-r1`.

## What the revision changed

The originally submitted version reported that ECFP4 fingerprints generalise
better than frozen polyBERT embeddings. **That claim is withdrawn.** Under the
randomised scaffold protocol ECFP4 reaches R2 = 0.747 +/- 0.020
against 0.716 +/- 0.015 for mean-pooled polyBERT, a difference of
+0.031 with a 95% confidence interval of [-0.003, 0.065]. Under the Butina
leave-clusters-out protocol the ordering reverses: polyBERT reaches
0.563 against 0.438 for ECFP4. The ranking is a property of the
evaluation protocol, not of the representations.

Two further corrections to the submitted description are carried here: the
training collection is the PolyMetriX curated Tg dataset (7,367 polymers),
not PI1M; and the polyBERT fingerprint is the mean over the token dimension of
the final encoder layer, not the CLS vector.

## Headline results

All values below are read from `paper/outputs/s10_tokens.json`, the same file
the manuscript is built from.

| Quantity | Value |
|---|---|
| Collection size | 7,367 polymers |
| Tier 1 MAE, ten canonical homopolymers | 3.9 K |
| Tier 1 MAE, full collection | 96 K (R2 = -0.27) |
| Tier 1 MAE at full structural coverage | 57 K |
| Tier 1 MAE below three quarters coverage | 226 K |
| Tier 2 R2, randomised scaffold split | 0.747 +/- 0.020 |
| Tier 2 R2, leave-clusters-out split | 0.438 +/- 0.015 |
| Tier 2 R2, random split | 0.863 +/- 0.009 |
| Tier 3 R2 (mean-pooled), scaffold split | 0.716 +/- 0.015 |
| Tier 3 R2 (mean-pooled), cluster split | 0.563 +/- 0.002 |
| Count-based fingerprint gain in R2 | +0.036 |
| Reliability index against error, Spearman | -0.165 |
| Conformal coverage at a nominal 80% | 73.7% |

A single number quoted without its protocol is not interpretable. The random
split gives R2 = 0.863 and the cluster split gives 0.438 for the same model
and the same data.

Machine-learning validation covers Tg only. The platform's other
thermophysical outputs are exploratory group contribution estimates.

## Reproducing every table and figure

```bash
pip install -r requirements.txt
python reproduce.py            # runs all twelve analysis steps in order
python reproduce.py --list     # show step status without running anything
```

`reproduce.py` is resumable: a step whose outputs already exist is skipped.
The archived outputs of a complete run are committed under `paper/outputs/`,
so every table and figure can be audited without re-running the pipeline.

## Tests

```bash
pip install -r requirements-ci.txt
pytest -q -m "not slow" tests/   # 61 unit tests, under a second
pytest -q -m slow tests/         # reproduces the submitted metrics to 9 dp
```

| Suite | Covers |
|---|---|
| `tests/test_psmiles.py` | PSMILES parsing, canonicalisation, the three capping conventions, and the invariance properties including the ones that fail |
| `tests/test_group_matching.py` | Atom-exclusive matching, cap-methyl decrement, determinism, coverage reporting, the fallback path, structural corrections |
| `tests/test_regression_reported_numbers.py` | Reproduction of the originally submitted metrics, and agreement between every manuscript token and its source file |

Continuous integration runs the import check and the test suite on every push
across Python 3.11, 3.12 and 3.13, and the slow regression test on 3.13. The
suite runs with `POLYX_STRICT=1` there, so a module it cannot find is a
failure rather than a skip. See `.github/workflows/ci.yml`.

Two tests skip in a checkout by design: they compare the built manuscript and
its figures against the archived outputs, and neither the manuscript source nor
the built figures are distributed in this repository.

## Repository layout

```
poly-x/
  poly_x/services/            the prediction modules the deployed platform
                              runs: group contribution, fingerprints, polyBERT,
                              applicability domain, reliability, aggregator
  poly_x/training/            the code that produced the Tier 2 and Tier 3
                              checkpoints
  poly_x/data/                the CC BY 4.0 training extract, with attribution
  poly_x/scaffold_split.py    Bemis-Murcko scaffold partitioning
  paper/analysis/             the twelve numbered analysis scripts, s01 to s12
  paper/outputs/              archived outputs of a complete run: split index
                              files for every protocol and seed, per-run model
                              results, per-polymer predictions, embeddings
  paper/RELEASE_NOTES.md      what this release contains and why
  tests/                      unit and regression suites
  reproduce.py                one-command reproduction entry point
  environment.lock.txt        exact versions behind every reported number
  requirements-ci.txt         minimal dependency set for the tests
```

## Data and model artifacts

The training collection is the published PolyMetriX curated Tg dataset,
[DOI: 10.5281/zenodo.14980914](https://doi.org/10.5281/zenodo.14980914),
released under CC BY 4.0. No new experimental data were generated for this
work. A two-column extract of 7,367 polymers is vendored at
`poly_x/data/tg_training_data.csv` with attribution, so that the regression
test reproducing the reported metrics runs in a bare checkout and in CI rather
than skipping. See `poly_x/data/README.md`.

Trained Tier 2 and Tier 3 prediction heads are attached to the GitHub Release
rather than committed, to keep the clone small. The fingerprint and embedding
matrices are not distributed because `reproduce.py` regenerates them
deterministically from the collection above.

## Quick start

```bash
python examples/predict_example.py
```

## Provenance note on polyBERT weights

The original `kuelumbus/polyBERT` repository was no longer resolvable on the
Hugging Face Hub at the time of this revision. A public mirror was used and
verified before use: its CLS vectors reproduce the embedding matrix stored with
the submitted models to a mean cosine of 1.000000 and a maximum absolute
elementwise difference of 0. See `environment.lock.txt`.

## Licence

See `LICENSE`.
