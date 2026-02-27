"""
Figure 4: Multi-Tier Comparison
Tg predictions from each tier alongside experimental values for 8 canonical polymers.
Uses REAL model predictions from trained Tier 2 and Tier 3 models.

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
})

polymers = ['PE', 'PP', 'PS', 'PVC', 'PET', 'PEO', 'POM', 'PVOH']
exp_tg =      [195, 253, 373, 354, 342, 206, 198, 358]
gc_tg =       [192.5, 251.9, 376.4, 349.6, 348.9, 210.0, 206.5, 357.3]
ml_tg =       [257.3, 276.2, 375.4, 268.8, 341.0, 250.6, 246.8, 248.6]
polybert_tg = [282.1, 327.2, 351.0, 319.0, 371.3, 272.6, 243.0, 277.8]

x = np.arange(len(polymers))
width = 0.2

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.bar(x - 1.5*width, exp_tg, width, label='Experimental',
       color='#2C3E50', edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*width, gc_tg, width, label='Tier 1 (GC)',
       color='#27AE60', edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*width, ml_tg, width, label='Tier 2 (ML)',
       color='#F39C12', edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*width, polybert_tg, width, label='Tier 3 (polyBERT)',
       color='#E74C3C', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Glass Transition Temperature, $T_g$ (K)')
ax.set_xlabel('Polymer')
ax.set_xticks(x)
ax.set_xticklabels(polymers, fontsize=11, fontweight='bold')
ax.set_ylim(100, 430)
ax.legend(loc='upper left', frameon=True, fancybox=True, ncol=2, fontsize=10)
ax.set_title('Figure 4. Multi-Tier Comparison of $T_g$ Predictions for Canonical Polymers',
             pad=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, (xp, etg) in enumerate(zip(x, exp_tg)):
    ax.fill_between([xp - 2*width, xp + 2*width],
                     etg - 10, etg + 10,
                     alpha=0.08, color='blue', zorder=0)

ax.text(0.98, 0.97,
        'Shaded bands: \u00b110 K around experimental value',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, fontstyle='italic', color='#555555')

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig4_tier_comparison.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(out_dir, 'fig4_tier_comparison.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 4 saved: fig4_tier_comparison.png / .pdf")
plt.close()
