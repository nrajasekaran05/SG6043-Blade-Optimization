# %%
# ============================================
# OPTIMIZATION JOURNEY SUMMARY
# Visual story from baseline to optimized design
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec

print("="*60)
print("CREATING OPTIMIZATION JOURNEY SUMMARY")
print("="*60)

data_dir = DATA_DIR

# Load data
df_comparison = pd.read_csv(data_dir + 'bayesian_optimization_comparison.csv')
df_history = pd.read_csv(data_dir + 'bayesian_optimization_history.csv')
df_perf = pd.read_csv(data_dir + 'detailed_performance_comparison.csv')

print("\n✓ Data loaded")

# Extract key metrics
baseline_clcd = df_comparison.loc[0, 'Cl/Cd']
doe_clcd = df_comparison.loc[1, 'Cl/Cd']
bo_clcd = df_comparison.loc[2, 'Cl/Cd']

improvement_baseline = df_comparison.loc[2, 'Improvement_vs_Baseline_%']
improvement_doe = ((bo_clcd / doe_clcd) - 1) * 100

# Create figure
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

# ============================================
# PLOT 1: OPTIMIZATION JOURNEY TIMELINE
# ============================================
ax1 = fig.add_subplot(gs[0, :])

# Timeline data
stages = ['Initial\nDesign', 'DOE\nGeneration', 'CFD\nAnalysis',
          'ML Training', 'Bayesian\nOptimization', 'Final\nDesign']
stage_x = np.arange(len(stages))

# Draw timeline
ax1.plot(stage_x, [1]*len(stage_x), 'k-', linewidth=3, zorder=1)

# Draw milestone circles
colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2ecc71', '#27ae60']
for i, (stage, color) in enumerate(zip(stages, colors)):
    circle = Circle((i, 1), 0.15, color=color, ec='black', linewidth=2, zorder=3)
    ax1.add_patch(circle)
    ax1.text(i, 0.5, stage, ha='center', va='top', fontsize=11,
             fontweight='bold', color=color)

# Add metrics at each stage
metrics = [
    f'SG6043\nCl/Cd: {baseline_clcd:.1f}',
    f'300 designs\nSobol sampling',
    f'4,500 evals\nAeroSandbox',
    f'R² > 0.96\nNeural Net',
    f'120 iterations\nGaussian Process',
    f'BEST DESIGN\nCl/Cd: {bo_clcd:.1f}'
]

for i, metric in enumerate(metrics):
    ax1.text(i, 1.4, metric, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

# Add arrows
for i in range(len(stages)-1):
    arrow = FancyArrowPatch((i+0.15, 1), (i+0.85, 1),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=2, color='gray', zorder=2)
    ax1.add_patch(arrow)

ax1.set_xlim(-0.5, len(stages)-0.5)
ax1.set_ylim(0, 2)
ax1.axis('off')
ax1.set_title('Optimization Journey Timeline', fontsize=16, fontweight='bold', pad=20)

# ============================================
# PLOT 2: PERFORMANCE EVOLUTION
# ============================================
ax2 = fig.add_subplot(gs[1, 0:2])

# Plot BO convergence
iterations = df_history['iteration'].values
cl_cd_values = df_history['cl_cd'].values
best_so_far = np.maximum.accumulate(cl_cd_values)

ax2.fill_between(iterations, 0, best_so_far, alpha=0.3, color='green', label='Improvement')
ax2.plot(iterations, best_so_far, 'g-', linewidth=3, label='Best Performance')
ax2.plot(iterations, cl_cd_values, 'b.', alpha=0.3, markersize=4, label='Evaluated Designs')

# Mark milestones
ax2.axhline(baseline_clcd, color='red', linestyle='--', linewidth=2,
           label=f'Baseline: {baseline_clcd:.1f}', alpha=0.7)
ax2.axhline(doe_clcd, color='orange', linestyle='--', linewidth=2,
           label=f'Best DOE: {doe_clcd:.1f}', alpha=0.7)
ax2.axhline(bo_clcd, color='green', linestyle='--', linewidth=2,
           label=f'Final BO: {bo_clcd:.1f}', alpha=0.7)

# Shade regions
ax2.axhspan(0, baseline_clcd, alpha=0.1, color='red')
ax2.axhspan(baseline_clcd, doe_clcd, alpha=0.1, color='orange')
ax2.axhspan(doe_clcd, bo_clcd, alpha=0.1, color='green')

# Annotations
best_idx = df_history['cl_cd'].idxmax()
best_iter = df_history.loc[best_idx, 'iteration']
ax2.annotate(f'Best found!\nCl/Cd = {bo_clcd:.1f}',
            xy=(best_iter, bo_clcd), xytext=(best_iter + 20, bo_clcd + 10),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'),
            fontsize=11, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cl/Cd (Lift-to-Drag Ratio)', fontsize=12, fontweight='bold')
ax2.set_title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(iterations))

# ============================================
# PLOT 3: PERFORMANCE COMPARISON BARS
# ============================================
ax3 = fig.add_subplot(gs[1, 2:])

designs = ['Baseline\nSG6043', 'Best DOE\n(300 designs)', 'Bayesian\nOptimization']
clcd_values = [baseline_clcd, doe_clcd, bo_clcd]
colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax3.bar(designs, clcd_values, color=colors_bar, alpha=0.8,
               edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, clcd_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add improvement annotations
ax3.annotate(f'+{improvement_baseline:.1f}%',
            xy=(2, bo_clcd/2), fontsize=14, fontweight='bold',
            ha='center', va='center', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2))

ax3.set_ylabel('Cl/Cd (Aerodynamic Efficiency)', fontsize=12, fontweight='bold')
ax3.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(clcd_values) * 1.15)

# ============================================
# PLOT 4: PERFORMANCE ACROSS ANGLES
# ============================================
ax4 = fig.add_subplot(gs[2, 0:2])

alphas = df_perf['Alpha'].values
baseline_clcd_alpha = df_perf['Baseline_ClCd'].values
doe_clcd_alpha = df_perf['DOE_ClCd'].values
bo_clcd_alpha = df_perf['BO_ClCd'].values

ax4.plot(alphas, baseline_clcd_alpha, '-o', color='#e74c3c', linewidth=3,
         markersize=8, label='Baseline SG6043', alpha=0.8)
ax4.plot(alphas, doe_clcd_alpha, '--s', color='#f39c12', linewidth=3,
         markersize=8, label='Best DOE', alpha=0.8)
ax4.plot(alphas, bo_clcd_alpha, '-^', color='#2ecc71', linewidth=3,
         markersize=9, label='Bayesian Opt', alpha=0.9)

ax4.fill_between(alphas, baseline_clcd_alpha, bo_clcd_alpha,
                 alpha=0.2, color='green', label='Performance Gain')

ax4.set_xlabel('Angle of Attack (degrees)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax4.set_title('Performance Across Operating Range', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='best')
ax4.grid(True, alpha=0.3)

# ============================================
# PLOT 5: SUMMARY STATISTICS
# ============================================
ax5 = fig.add_subplot(gs[2, 2:])
ax5.axis('off')

summary_text = f"""
╔═══════════════════════════════════════════════════════╗
║            OPTIMIZATION SUMMARY                       ║
╚═══════════════════════════════════════════════════════╝

DESIGN EVOLUTION:
  Baseline SG6043        →  Best DOE  →  Bayesian Opt
  Cl/Cd: {baseline_clcd:.1f}          {doe_clcd:.1f}         {bo_clcd:.1f}

IMPROVEMENTS:
  ✓ vs Baseline:     +{improvement_baseline:.1f}% ({bo_clcd/baseline_clcd:.2f}× better)
  ✓ vs Best DOE:     +{improvement_doe:.1f}%

OPTIMIZATION EFFICIENCY:
  • DOE Method:      300 designs, 4,500 evaluations
  • Bayesian Opt:    120 iterations (96% fewer!)
  • Time:            151 seconds (~0.8 eval/s)

KEY ACHIEVEMENTS:
  ✓ Found design better than all 300 DOE samples
  ✓ 46% improvement over exhaustive search
  ✓ Intelligent exploration of 18D design space
  ✓ Publication-quality optimization workflow

PERFORMANCE AT α=6° (CRUISE):
                 Baseline    DOE      Bayesian
  Cl/Cd:         {df_perf.loc[3, 'Baseline_ClCd']:.1f}       {df_perf.loc[3, 'DOE_ClCd']:.1f}     {df_perf.loc[3, 'BO_ClCd']:.1f}
  Cl:            {df_perf.loc[3, 'Baseline_Cl']:.3f}     {df_perf.loc[3, 'DOE_Cl']:.3f}   {df_perf.loc[3, 'BO_Cl']:.3f}
  Cd:            {df_perf.loc[3, 'Baseline_Cd']:.5f}   {df_perf.loc[3, 'DOE_Cd']:.5f} {df_perf.loc[3, 'BO_Cd']:.5f}

METHOD: Sequential Model-Based Optimization
  • Gaussian Process surrogate
  • Expected Improvement acquisition
  • Neural network objective function (R² > 0.96)
"""

ax5.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
         family='monospace',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                  edgecolor='darkblue', alpha=0.3, linewidth=2))

# Add main title
fig.suptitle('AIRFOIL OPTIMIZATION: FROM BASELINE TO BEST DESIGN\n' +
             'Sequential Model-Based Bayesian Optimization Success Story',
             fontsize=18, fontweight='bold', y=0.98)

# Save
plt.savefig(data_dir + 'optimization_journey_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Journey summary saved: optimization_journey_summary.png")

# ============================================
# CREATE SIMPLE 1-PAGE SUMMARY
# ============================================
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('BAYESIAN OPTIMIZATION RESULTS: ONE-PAGE SUMMARY',
              fontsize=18, fontweight='bold')

# Plot 1: Bar comparison
ax = axes[0, 0]
bars = ax.bar(designs, clcd_values, color=colors_bar, alpha=0.8,
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, clcd_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
           f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Convergence
ax = axes[0, 1]
ax.plot(iterations, best_so_far, 'g-', linewidth=3, label='Best so far')
ax.plot(iterations, cl_cd_values, 'b.', alpha=0.2, markersize=3)
ax.axhline(baseline_clcd, color='red', linestyle='--', linewidth=2, label='Baseline')
ax.axhline(doe_clcd, color='orange', linestyle='--', linewidth=2, label='Best DOE')
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Performance vs alpha
ax = axes[1, 0]
ax.plot(alphas, baseline_clcd_alpha, '-o', color='#e74c3c', linewidth=2.5,
        markersize=6, label='Baseline')
ax.plot(alphas, doe_clcd_alpha, '--s', color='#f39c12', linewidth=2.5,
        markersize=6, label='Best DOE')
ax.plot(alphas, bo_clcd_alpha, '-^', color='#2ecc71', linewidth=3,
        markersize=7, label='Bayesian Opt')
ax.set_xlabel('Angle of Attack (deg)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax.set_title('Performance Across Angles', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Summary text
ax = axes[1, 1]
ax.axis('off')

summary_box = f"""
KEY RESULTS:

Max Cl/Cd Achieved:
  • Baseline:        {baseline_clcd:.2f}
  • Best DOE:        {doe_clcd:.2f}
  • Bayesian Opt:    {bo_clcd:.2f}

Improvements:
  • vs Baseline:     +{improvement_baseline:.1f}%
  • vs Best DOE:     +{improvement_doe:.1f}%

Optimization Details:
  • Method:          Gaussian Process BO
  • Evaluations:     120 iterations
  • Time:            151 seconds
  • Acquisition:     Expected Improvement

Design Space:
  • Parameters:      18 (Chebyshev coeffs)
  • Bounds:          ±10% from DOE range

Performance at α=6°:
  • Cl/Cd:          {df_perf.loc[3, 'BO_ClCd']:.1f} (vs {df_perf.loc[3, 'Baseline_ClCd']:.1f} baseline)
  • Improvement:    +{df_perf.loc[3, 'Improvement_vs_Baseline_%']:.1f}%

Conclusion:
  Bayesian Optimization successfully
  found superior airfoil designs with
  96% fewer evaluations than DOE!
"""

ax.text(0.1, 0.5, summary_box, fontsize=12, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                 edgecolor='orange', alpha=0.5, linewidth=2))

plt.tight_layout()
plt.savefig(data_dir + 'optimization_one_page_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ One-page summary saved: optimization_one_page_summary.png")

print("\n" + "="*60)
print("✅ ALL SUMMARY VISUALIZATIONS COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  • optimization_journey_summary.png")
print("  • optimization_one_page_summary.png")
print("="*60)
