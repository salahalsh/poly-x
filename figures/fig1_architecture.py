"""
Figure 1: POLY-X System Architecture Diagram
Generates a schematic of the 3-tier prediction pipeline with enhanced features.

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

c_input = '#E8F4FD'
c_tier1 = '#D4EDDA'
c_tier2 = '#FFF3CD'
c_tier3 = '#F8D7DA'
c_enhanced = '#E2D9F3'
c_output = '#D1ECF1'
c_border = '#333333'

def add_box(x, y, w, h, text, color, fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True)

def add_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

# === Input Layer ===
add_box(3.5, 7.0, 3.0, 0.7, 'PSMILES Input\n[*]CC(c1ccccc1)[*]', c_input, fontsize=9, bold=True)

add_arrow(5.0, 7.0, 5.0, 6.6)
add_box(3.5, 6.0, 3.0, 0.55, 'PSMILES Parser\n([*] \u2192 [CH3] capping)', c_input, fontsize=8)

add_arrow(5.0, 6.0, 5.0, 5.6)
add_box(3.0, 5.0, 4.0, 0.55, 'PolyPropertyAggregator\n(Orchestrator)', '#CFE2FF', fontsize=9, bold=True)

# === Three Tiers ===
add_arrow(3.8, 5.0, 1.5, 4.5)
add_box(0.3, 3.5, 2.5, 0.95,
        'Tier 1: Group Contribution\n25 SMARTS groups\nVan Krevelen equations\nStructural corrections',
        c_tier1, fontsize=7.5)

add_arrow(5.0, 5.0, 5.0, 4.5)
add_box(3.7, 3.5, 2.6, 0.95,
        'Tier 2: ML Ensemble\nECFP4 (2048-bit)\nRF(500) + GB(200)\nScaffold split on 7,365',
        c_tier2, fontsize=7.5)

add_arrow(6.2, 5.0, 8.5, 4.5)
add_box(7.2, 3.5, 2.5, 0.95,
        'Tier 3: polyBERT\nDeBERTa (600-D CLS)\nTransformer embeddings\nRF+GB head',
        c_tier3, fontsize=7.5)

# Arrows converge to enhanced features
add_arrow(1.5, 3.5, 3.5, 2.9)
add_arrow(5.0, 3.5, 5.0, 2.9)
add_arrow(8.5, 3.5, 6.5, 2.9)

# === Enhanced Features ===
add_box(2.5, 2.1, 5.0, 0.75,
        'Enhanced Features\nAD Assessment  |  PRI  |  Comparative Profiling\nProcessability  |  Group Analysis',
        c_enhanced, fontsize=8, bold=True)

add_arrow(5.0, 2.1, 5.0, 1.7)

# === Output Layer ===
add_box(2.0, 0.8, 6.0, 0.75,
        'Output: Properties (Tg, Tm, density, \u03b4, CED) + Reliability Scores\nCSV/Excel Export  |  Interactive Web Dashboard  |  Celery Batch Queue',
        c_output, fontsize=8, bold=True)

# === Side labels ===
ax.text(0.1, 7.3, 'Input', fontsize=10, fontweight='bold', color='#0066CC')
ax.text(0.1, 5.2, 'Orchestration', fontsize=9, fontweight='bold', color='#0066CC')
ax.text(0.1, 2.4, 'Enhancement', fontsize=9, fontweight='bold', color='#6600CC')
ax.text(0.1, 1.0, 'Output', fontsize=9, fontweight='bold', color='#006666')

ax.text(5.0, 7.9, 'Figure 1. POLY-X System Architecture',
        ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig1_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(out_dir, 'fig1_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 saved: fig1_architecture.png / .pdf")
plt.close()
