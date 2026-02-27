"""
Figure 3: ML Ensemble Parity Plot
Predicted vs actual Tg for the scaffold-split test set.
Uses REAL model predictions extracted from the trained RF+GB ensemble.

Citation:
    Jebril, I.H.; Alshehade, S.A.A. POLY-X: A Multi-Tier Computational
    Platform for Polymer Thermophysical Property Prediction. J. Chem. Inf.
    Model. 2026.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
})

# Load REAL test set predictions (relative paths)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
y_actual = np.load(os.path.join(data_dir, 'fig3_test_actual.npy'))
y_pred = np.load(os.path.join(data_dir, 'fig3_test_pred.npy'))

n_test = len(y_actual)
r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(np.mean((y_actual - y_pred)**2))

print(f"Loaded {n_test} test polymers. R2={r2:.4f}, MAE={mae:.2f} K, RMSE={rmse:.2f} K")

fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))

from scipy.stats import gaussian_kde

xy = np.vstack([y_actual, y_pred])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
y_actual_sorted = y_actual[idx]
y_pred_sorted = y_pred[idx]
z_sorted = z[idx]

scatter = ax.scatter(y_actual_sorted, y_pred_sorted, c=z_sorted, s=12, alpha=0.7,
                      cmap='viridis', edgecolors='none', rasterized=True)

lims = [min(y_actual.min(), y_pred.min()) - 20,
        max(y_actual.max(), y_pred.max()) + 20]
ax.plot(lims, lims, 'k-', linewidth=1.5, label='Identity ($y = x$)')

ax.plot(lims, [l + 50 for l in lims], 'k--', linewidth=0.7, alpha=0.4)
ax.plot(lims, [l - 50 for l in lims], 'k--', linewidth=0.7, alpha=0.4)
ax.fill_between(lims, [l - 50 for l in lims], [l + 50 for l in lims],
                alpha=0.05, color='gray')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Point Density', fontsize=10)

textstr = (f'$N$ = {n_test}\n'
           f'$R^2$ = {r2:.3f}\n'
           f'MAE = {mae:.1f} K\n'
           f'RMSE = {rmse:.1f} K')
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

ax.set_xlabel('Experimental $T_g$ (K)')
ax.set_ylabel('Predicted $T_g$ (K)')
ax.set_title('Figure 3. ML Ensemble Parity Plot (Scaffold-Split Test Set)', pad=12)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.2, linestyle='--')

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'fig3_ml_parity.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(out_dir, 'fig3_ml_parity.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 3 saved: fig3_ml_parity.png / .pdf")
plt.close()
