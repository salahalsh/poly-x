# POLY-X: Multi-Tier Polymer Property Prediction Platform

**POLY-X** is a computational platform for polymer thermophysical property prediction that integrates three complementary prediction paradigms:

| Tier | Method | Description |
|------|--------|-------------|
| **Tier 1** | Van Krevelen Group Contribution | 25 SMARTS-defined functional groups, structural corrections, multi-level Boyer-Beaman Tm estimation |
| **Tier 2** | ML Ensemble (RF + GB) | ECFP4 fingerprints (2048-bit), trained on 7,365 polymers from PolyMetriX, scaffold-split evaluation |
| **Tier 3** | polyBERT Embeddings | Pre-trained DeBERTa transformer (600-D), lightweight RF+GB prediction heads |

## Key Results

- **Tier 1 (GC):** MAE = 3.9 K for Tg across 8 canonical homopolymers
- **Tier 2 (ML):** Test R² = 0.553, MAE = 43.6 K (scaffold split)
- **Tier 3 (polyBERT):** Test R² = 0.517, MAE = 47.0 K (scaffold split)

ECFP4 fingerprints outperform frozen polyBERT embeddings under scaffold splitting, demonstrating that scaffold-agnostic local representations generalize more effectively to structurally novel polymers.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, RDKit, scikit-learn, numpy, scipy, pandas, matplotlib

For Tier 3 (polyBERT), additionally install:
```bash
pip install transformers torch
```

## Quick Start

### Tier 1: Group Contribution Prediction

```bash
python -m poly_x.group_contribution "[*]CC(c1ccccc1)[*]"
```

### Example Script

```bash
python examples/predict_example.py
```

### Training (Tier 2)

```bash
python -m poly_x.train_tg --data-path <path_to_tg_data.csv> --output-dir results/
```

The training data (PolyMetriX, 7,365 polymers) is available from Zenodo: [DOI: 10.5281/zenodo.14980914](https://doi.org/10.5281/zenodo.14980914)

## Repository Structure

```
poly-x/
├── poly_x/                     # Core computational modules
│   ├── group_contribution.py   # Tier 1: Van Krevelen GC calculator
│   ├── fingerprints.py         # ECFP4 fingerprint generation
│   ├── scaffold_split.py       # Murcko scaffold splitting
│   └── train_tg.py             # Tier 2: ML training pipeline
├── figures/                    # Figure generation scripts
│   ├── fig1_architecture.py    # System architecture diagram
│   ├── fig2_gc_validation.py   # GC validation bar chart
│   ├── fig3_ml_parity.py       # ML parity plot (real predictions)
│   ├── fig4_tier_comparison.py # Multi-tier comparison
│   ├── fig5_enhanced_features.py # Enhanced features panel
│   └── data/                   # Pre-computed test predictions (.npy)
├── results/                    # Model performance metadata
│   ├── tier2_metrics.json      # ECFP4 ML metrics
│   └── tier3_metrics.json      # polyBERT metrics
├── examples/
│   └── predict_example.py      # Quick demo script
├── requirements.txt
└── LICENSE
```

## Data Sources

- **Training data:** PolyMetriX dataset (7,365 polymers with experimental Tg values) — [Zenodo DOI: 10.5281/zenodo.14980914](https://doi.org/10.5281/zenodo.14980914)
- **Benchmark dataset:** PI1M — Ma, R.; Luo, T. *J. Chem. Inf. Model.* 2020, 60, 4684–4690
- **polyBERT model:** [kuelumbus/polyBERT on HuggingFace](https://huggingface.co/kuelumbus/polyBERT)
- **Group contribution parameters:** Van Krevelen & Te Nijenhuis, *Properties of Polymers*, 4th Ed., 2009

## Web Application

POLY-X is freely accessible as a web application at [https://insilicosigma.com/poly-x/](https://insilicosigma.com/poly-x/), where users can submit polymer structures in PSMILES notation and obtain multi-tier predictions with all enhanced analytical features.

## Citation

If you use POLY-X in your research, please cite:

```bibtex
@article{Jebril2026POLYX,
  title   = {POLY-X: A Multi-Tier Computational Platform for Polymer
             Thermophysical Property Prediction Integrating Group
             Contribution Theory, Machine Learning Ensembles, and
             Transformer Embeddings},
  author  = {Jebril, Iqbal H. and Alshehade, Salah Addin A.},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This software is free for academic, research, educational, and personal use. Commercial use requires written permission from the corresponding author. See [LICENSE](LICENSE) for details.
