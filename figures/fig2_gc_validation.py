"""
Figure 2: Group Contribution Validation
Predicted vs experimental Tg for 8 canonical homopolymers.

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
exp_tg = [195, 253, 373, 354, 342, 206, 198, 358]
pred_tg = [192.5, 251.9, 376.4, 349.6, 348.9, 210.0, 206.5, 357.3]
errors = [abs(e - p) for e, p in zip(exp_tg, pred_tg)]

x = np.arange(len(polymers))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1],
                                 gridspec_kw={'hspace': 0.08})

bars_exp = ax1.bar(x - width/2, exp_tg, width, label='Experimental',
                    color='#2E86AB', edgecolor='black', linewidth=0.5)
bars_pred = ax1.bar(x + width/2, pred_tg, width, label='Predicted (GC)',
                     color='#E8702A', edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Glass Transition Temperature, $T_g$ (K)')
ax1.set_ylim(100, 420)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
ax1.set_title('Figure 2. Group Contribution Validation: Predicted vs. Experimental $T_g$',
              pad=12)

for bar in bars_exp:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 3,
             f'{h:.0f}', ha='center', va='bottom', fontsize=7.5)
for bar in bars_pred:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 3,
             f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)

ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

colors = ['#27AE60' if e <= 5 else '#F39C12' if e <= 8 else '#E74C3C' for e in errors]
ax2.bar(x, errors, 0.5, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Tolerance (10 K)')
ax2.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='5 K threshold')
ax2.set_ylabel('|Error| (K)')
ax2.set_ylim(0, 12)
ax2.set_xticks(x)
ax2.set_xticklabels(polymers, fontsize=10, fontweight='bold')
ax2.set_xlabel('Polymer')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, e in enumerate(errors):
    ax2.text(i, e + 0.3, f'{e:.1f}', ha='center', va='bottom', fontsize=8)

mean_err = np.mean(errors)
ax2.text(0.50, 0.85, f'Mean |Error| = {mean_err:.1f} K',
         transform=ax2.transAxes, ha='center', va='top',
         fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig2_gc_validation.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(out_dir, 'fig2_gc_validation.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 2 saved: fig2_gc_validation.png / .pdf")
plt.close()
