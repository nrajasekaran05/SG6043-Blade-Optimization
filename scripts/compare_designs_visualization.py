# %%
# ============================================
# COMPREHENSIVE DESIGN COMPARISON VISUALIZATION
# Baseline vs Best DOE vs Bayesian Optimization
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow import keras
import joblib

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

print("="*70)
print("COMPREHENSIVE DESIGN COMPARISON VISUALIZATION")
print("Baseline SG6043 vs Best DOE vs Bayesian Optimization")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

data_dir = DATA_DIR

# Alpha range for performance evaluation
ALPHA_RANGE = np.linspace(-2, 12, 30)

# ============================================
# STEP 1: LOAD MODELS AND DATA
# ============================================
print("\nStep 1: Loading models and data...")

model = keras.models.load_model(data_dir + 'airfoil_surrogate_model.keras')
scaler_X = joblib.load(data_dir + 'scaler_X.pkl')
scaler_y = joblib.load(data_dir + 'scaler_y.pkl')

df_params = pd.read_csv(data_dir + 'DOE_parameters.csv')
df_results = pd.read_csv(data_dir + 'aerosandbox_results.csv')
df_bo_best = pd.read_csv(data_dir + 'bayesian_best_design.csv')

print("✓ Models and data loaded")

# ============================================
# STEP 2: IDENTIFY DESIGNS
# ============================================
print("\nStep 2: Identifying designs to compare...")

# Design 1: Baseline SG6043
baseline_id = 0
baseline_coeffs = df_params[df_params['design_id'] == baseline_id].iloc[0]

# Design 2: Best DOE
best_doe_id = 221  # From previous results
best_doe_coeffs = df_params[df_params['design_id'] == best_doe_id].iloc[0]

# Design 3: Bayesian Optimization
bo_coeffs = df_bo_best.iloc[0]

print(f"✓ Comparing three designs:")
print(f"  1. Baseline SG6043 (Design ID {baseline_id})")
print(f"  2. Best DOE (Design ID {best_doe_id})")
print(f"  3. Bayesian Optimization (BO_best)")

# ============================================
# STEP 3: RECONSTRUCT AIRFOIL SHAPES
# ============================================
print("\nStep 3: Reconstructing airfoil shapes...")

def chebyshev_polynomials(x, n_order=8):
    """Generate Chebyshev polynomials up to order n"""
    T = np.zeros((len(x), n_order + 1))
    T[:, 0] = 1
    T[:, 1] = x
    for i in range(2, n_order + 1):
        T[:, i] = 2 * x * T[:, i-1] - T[:, i-2]
    return T

def reconstruct_airfoil(coeffs, n_points=200):
    """Reconstruct airfoil coordinates from Chebyshev coefficients"""
    # Cosine spacing for better resolution near leading/trailing edges
    theta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(theta))

    # Transform x from [0,1] to [-1,1] for Chebyshev
    x_cheb = 2 * x - 1
    T = chebyshev_polynomials(x_cheb, n_order=8)

    # Extract coefficients
    upper_coeffs = [coeffs[f'upper_c{i}'] for i in range(9)]
    lower_coeffs = [coeffs[f'lower_c{i}'] for i in range(9)]

    # Calculate y coordinates
    y_upper = T @ np.array(upper_coeffs)
    y_lower = T @ np.array(lower_coeffs)

    return x, y_upper, y_lower

# Reconstruct all three designs
x_baseline, y_upper_baseline, y_lower_baseline = reconstruct_airfoil(baseline_coeffs)
x_doe, y_upper_doe, y_lower_doe = reconstruct_airfoil(best_doe_coeffs)
x_bo, y_upper_bo, y_lower_bo = reconstruct_airfoil(bo_coeffs)

print("✓ Airfoil shapes reconstructed")

# ============================================
# STEP 4: EVALUATE PERFORMANCE
# ============================================
print("\nStep 4: Evaluating performance across angles...")

def predict_performance(coeffs, alpha_range):
    """Predict performance across angle of attack range"""
    coef_names = [f'upper_c{i}' for i in range(9)] + [f'lower_c{i}' for i in range(9)]
    coef_values = [coeffs[c] for c in coef_names]

    results = {'alpha': [], 'cl': [], 'cd': [], 'cm': [], 'cl_cd': []}

    for alpha in alpha_range:
        X_input = np.array(coef_values + [alpha]).reshape(1, -1)
        X_scaled = scaler_X.transform(X_input)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        cl = y_pred[0, 0]
        cd = y_pred[0, 1]
        cm = y_pred[0, 2]
        cl_cd = cl / (cd + 1e-10)

        results['alpha'].append(alpha)
        results['cl'].append(cl)
        results['cd'].append(cd)
        results['cm'].append(cm)
        results['cl_cd'].append(cl_cd)

    return pd.DataFrame(results)

# Evaluate all designs
perf_baseline = predict_performance(baseline_coeffs, ALPHA_RANGE)
perf_doe = predict_performance(best_doe_coeffs, ALPHA_RANGE)
perf_bo = predict_performance(bo_coeffs, ALPHA_RANGE)

print("✓ Performance evaluated")
print(f"\n  Max Cl/Cd:")
print(f"    Baseline: {perf_baseline['cl_cd'].max():.2f}")
print(f"    Best DOE: {perf_doe['cl_cd'].max():.2f}")
print(f"    Bayesian Opt: {perf_bo['cl_cd'].max():.2f}")

# ============================================
# STEP 5: CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================
print("\nStep 5: Creating comprehensive visualizations...")

# Create a large figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

# Colors for each design
color_baseline = '#1f77b4'  # Blue
color_doe = '#ff7f0e'       # Orange
color_bo = '#2ca02c'        # Green

# ============================================
# PLOT 1: AIRFOIL SHAPE COMPARISON (Large)
# ============================================
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(x_baseline, y_upper_baseline, '-', color=color_baseline, linewidth=2.5,
         label='Baseline SG6043', alpha=0.8)
ax1.plot(x_baseline, y_lower_baseline, '-', color=color_baseline, linewidth=2.5, alpha=0.8)
ax1.fill_between(x_baseline, y_lower_baseline, y_upper_baseline,
                  color=color_baseline, alpha=0.15)

ax1.plot(x_doe, y_upper_doe, '--', color=color_doe, linewidth=2.5,
         label=f'Best DOE (ID {best_doe_id})', alpha=0.8)
ax1.plot(x_doe, y_lower_doe, '--', color=color_doe, linewidth=2.5, alpha=0.8)

ax1.plot(x_bo, y_upper_bo, '-', color=color_bo, linewidth=3,
         label='Bayesian Optimization', alpha=0.9)
ax1.plot(x_bo, y_lower_bo, '-', color=color_bo, linewidth=3, alpha=0.9)

ax1.set_xlabel('x/c (Chordwise Position)', fontsize=14, fontweight='bold')
ax1.set_ylabel('y/c (Thickness)', fontsize=14, fontweight='bold')
ax1.set_title('Airfoil Shape Comparison', fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal')
ax1.set_xlim(-0.05, 1.05)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

# ============================================
# PLOT 2: SHAPE DIFFERENCE (ZOOMED LEADING EDGE)
# ============================================
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(x_baseline, y_upper_baseline, '-', color=color_baseline, linewidth=2,
         label='Baseline', alpha=0.8)
ax2.plot(x_doe, y_upper_doe, '--', color=color_doe, linewidth=2,
         label='Best DOE', alpha=0.8)
ax2.plot(x_bo, y_upper_bo, '-', color=color_bo, linewidth=2.5,
         label='Bayesian Opt', alpha=0.9)

ax2.set_xlabel('x/c', fontsize=11, fontweight='bold')
ax2.set_ylabel('y/c (Upper Surface)', fontsize=11, fontweight='bold')
ax2.set_title('Leading Edge Detail', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0, 0.08)

# ============================================
# PLOT 3: THICKNESS DISTRIBUTION
# ============================================
ax3 = fig.add_subplot(gs[1, 1])

thickness_baseline = y_upper_baseline - y_lower_baseline
thickness_doe = y_upper_doe - y_lower_doe
thickness_bo = y_upper_bo - y_lower_bo

ax3.plot(x_baseline, thickness_baseline * 100, '-', color=color_baseline,
         linewidth=2, label='Baseline', alpha=0.8)
ax3.plot(x_doe, thickness_doe * 100, '--', color=color_doe,
         linewidth=2, label='Best DOE', alpha=0.8)
ax3.plot(x_bo, thickness_bo * 100, '-', color=color_bo,
         linewidth=2.5, label='Bayesian Opt', alpha=0.9)

ax3.set_xlabel('x/c', fontsize=11, fontweight='bold')
ax3.set_ylabel('Thickness (% chord)', fontsize=11, fontweight='bold')
ax3.set_title('Thickness Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

# ============================================
# PLOT 4: CAMBER LINE
# ============================================
ax4 = fig.add_subplot(gs[1, 2])

camber_baseline = (y_upper_baseline + y_lower_baseline) / 2
camber_doe = (y_upper_doe + y_lower_doe) / 2
camber_bo = (y_upper_bo + y_lower_bo) / 2

ax4.plot(x_baseline, camber_baseline * 100, '-', color=color_baseline,
         linewidth=2, label='Baseline', alpha=0.8)
ax4.plot(x_doe, camber_doe * 100, '--', color=color_doe,
         linewidth=2, label='Best DOE', alpha=0.8)
ax4.plot(x_bo, camber_bo * 100, '-', color=color_bo,
         linewidth=2.5, label='Bayesian Opt', alpha=0.9)

ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.set_xlabel('x/c', fontsize=11, fontweight='bold')
ax4.set_ylabel('Camber (% chord)', fontsize=11, fontweight='bold')
ax4.set_title('Mean Camber Line', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)

# ============================================
# PLOT 5: DRAG POLAR (Cl vs Cd)
# ============================================
ax5 = fig.add_subplot(gs[2, 0])

ax5.plot(perf_baseline['cd'], perf_baseline['cl'], '-o', color=color_baseline,
         linewidth=2.5, markersize=5, label='Baseline', alpha=0.8)
ax5.plot(perf_doe['cd'], perf_doe['cl'], '--s', color=color_doe,
         linewidth=2.5, markersize=5, label='Best DOE', alpha=0.8)
ax5.plot(perf_bo['cd'], perf_bo['cl'], '-^', color=color_bo,
         linewidth=3, markersize=6, label='Bayesian Opt', alpha=0.9)

# Mark best points
ax5.scatter([perf_baseline.loc[perf_baseline['cl_cd'].idxmax(), 'cd']],
           [perf_baseline.loc[perf_baseline['cl_cd'].idxmax(), 'cl']],
           s=200, marker='*', color=color_baseline, edgecolors='black',
           linewidth=2, zorder=5)
ax5.scatter([perf_doe.loc[perf_doe['cl_cd'].idxmax(), 'cd']],
           [perf_doe.loc[perf_doe['cl_cd'].idxmax(), 'cl']],
           s=200, marker='*', color=color_doe, edgecolors='black',
           linewidth=2, zorder=5)
ax5.scatter([perf_bo.loc[perf_bo['cl_cd'].idxmax(), 'cd']],
           [perf_bo.loc[perf_bo['cl_cd'].idxmax(), 'cl']],
           s=300, marker='*', color=color_bo, edgecolors='black',
           linewidth=2, zorder=5)

ax5.set_xlabel('Cd (Drag Coefficient)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cl (Lift Coefficient)', fontsize=11, fontweight='bold')
ax5.set_title('Drag Polar', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9, loc='best')
ax5.grid(True, alpha=0.3)

# ============================================
# PLOT 6: LIFT CURVE (Cl vs Alpha)
# ============================================
ax6 = fig.add_subplot(gs[2, 1])

ax6.plot(perf_baseline['alpha'], perf_baseline['cl'], '-o', color=color_baseline,
         linewidth=2.5, markersize=4, label='Baseline', alpha=0.8)
ax6.plot(perf_doe['alpha'], perf_doe['cl'], '--s', color=color_doe,
         linewidth=2.5, markersize=4, label='Best DOE', alpha=0.8)
ax6.plot(perf_bo['alpha'], perf_bo['cl'], '-^', color=color_bo,
         linewidth=3, markersize=5, label='Bayesian Opt', alpha=0.9)

ax6.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax6.set_xlabel('Angle of Attack (deg)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cl (Lift Coefficient)', fontsize=11, fontweight='bold')
ax6.set_title('Lift Curve', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, loc='best')
ax6.grid(True, alpha=0.3)

# ============================================
# PLOT 7: DRAG CURVE (Cd vs Alpha)
# ============================================
ax7 = fig.add_subplot(gs[2, 2])

ax7.plot(perf_baseline['alpha'], perf_baseline['cd'] * 1000, '-o',
         color=color_baseline, linewidth=2.5, markersize=4, label='Baseline', alpha=0.8)
ax7.plot(perf_doe['alpha'], perf_doe['cd'] * 1000, '--s',
         color=color_doe, linewidth=2.5, markersize=4, label='Best DOE', alpha=0.8)
ax7.plot(perf_bo['alpha'], perf_bo['cd'] * 1000, '-^',
         color=color_bo, linewidth=3, markersize=5, label='Bayesian Opt', alpha=0.9)

ax7.set_xlabel('Angle of Attack (deg)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Cd × 1000 (Drag Coefficient)', fontsize=11, fontweight='bold')
ax7.set_title('Drag Curve', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9, loc='best')
ax7.grid(True, alpha=0.3)

# ============================================
# PLOT 8: EFFICIENCY CURVE (Cl/Cd vs Alpha)
# ============================================
ax8 = fig.add_subplot(gs[3, 0])

ax8.plot(perf_baseline['alpha'], perf_baseline['cl_cd'], '-o',
         color=color_baseline, linewidth=2.5, markersize=4, label='Baseline', alpha=0.8)
ax8.plot(perf_doe['alpha'], perf_doe['cl_cd'], '--s',
         color=color_doe, linewidth=2.5, markersize=4, label='Best DOE', alpha=0.8)
ax8.plot(perf_bo['alpha'], perf_bo['cl_cd'], '-^',
         color=color_bo, linewidth=3, markersize=5, label='Bayesian Opt', alpha=0.9)

# Mark maximum points
max_baseline = perf_baseline['cl_cd'].max()
max_doe = perf_doe['cl_cd'].max()
max_bo = perf_bo['cl_cd'].max()

ax8.axhline(y=max_baseline, color=color_baseline, linestyle=':',
            linewidth=1.5, alpha=0.5)
ax8.axhline(y=max_doe, color=color_doe, linestyle=':',
            linewidth=1.5, alpha=0.5)
ax8.axhline(y=max_bo, color=color_bo, linestyle=':',
            linewidth=1.5, alpha=0.5)

ax8.set_xlabel('Angle of Attack (deg)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Cl/Cd (Lift-to-Drag Ratio)', fontsize=11, fontweight='bold')
ax8.set_title('Aerodynamic Efficiency', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9, loc='best')
ax8.grid(True, alpha=0.3)

# ============================================
# PLOT 9: MOMENT COEFFICIENT (Cm vs Alpha)
# ============================================
ax9 = fig.add_subplot(gs[3, 1])

ax9.plot(perf_baseline['alpha'], perf_baseline['cm'], '-o',
         color=color_baseline, linewidth=2.5, markersize=4, label='Baseline', alpha=0.8)
ax9.plot(perf_doe['alpha'], perf_doe['cm'], '--s',
         color=color_doe, linewidth=2.5, markersize=4, label='Best DOE', alpha=0.8)
ax9.plot(perf_bo['alpha'], perf_bo['cm'], '-^',
         color=color_bo, linewidth=3, markersize=5, label='Bayesian Opt', alpha=0.9)

ax9.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax9.set_xlabel('Angle of Attack (deg)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Cm (Moment Coefficient)', fontsize=11, fontweight='bold')
ax9.set_title('Pitching Moment', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9, loc='best')
ax9.grid(True, alpha=0.3)

# ============================================
# PLOT 10: PERFORMANCE COMPARISON BAR CHART
# ============================================
ax10 = fig.add_subplot(gs[3, 2])

categories = ['Max Cl/Cd', 'Max Cl', 'Min Cd×1000']
baseline_vals = [perf_baseline['cl_cd'].max(),
                 perf_baseline['cl'].max(),
                 perf_baseline['cd'].min() * 1000]
doe_vals = [perf_doe['cl_cd'].max(),
            perf_doe['cl'].max(),
            perf_doe['cd'].min() * 1000]
bo_vals = [perf_bo['cl_cd'].max(),
           perf_bo['cl'].max(),
           perf_bo['cd'].min() * 1000]

x_pos = np.arange(len(categories))
width = 0.25

bars1 = ax10.bar(x_pos - width, baseline_vals, width, label='Baseline',
                 color=color_baseline, alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax10.bar(x_pos, doe_vals, width, label='Best DOE',
                 color=color_doe, alpha=0.8, edgecolor='black', linewidth=1)
bars3 = ax10.bar(x_pos + width, bo_vals, width, label='Bayesian Opt',
                 color=color_bo, alpha=0.9, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

ax10.set_ylabel('Value', fontsize=11, fontweight='bold')
ax10.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax10.set_xticks(x_pos)
ax10.set_xticklabels(categories, fontsize=9)
ax10.legend(fontsize=9, loc='best')
ax10.grid(True, alpha=0.3, axis='y')

# Add main title
fig.suptitle('COMPREHENSIVE AIRFOIL DESIGN COMPARISON\n' +
             'Baseline SG6043 vs Best DOE vs Bayesian Optimization',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig(data_dir + 'comprehensive_design_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Main comparison saved: comprehensive_design_comparison.png")

# ============================================
# STEP 6: CREATE DETAILED PERFORMANCE TABLE
# ============================================
print("\nStep 6: Creating detailed performance table...")

# Create performance comparison at key angles
key_alphas = [0, 2, 4, 6, 8, 10]
comparison_data = []

for alpha in key_alphas:
    # Get closest alpha values
    idx_base = (perf_baseline['alpha'] - alpha).abs().idxmin()
    idx_doe = (perf_doe['alpha'] - alpha).abs().idxmin()
    idx_bo = (perf_bo['alpha'] - alpha).abs().idxmin()

    comparison_data.append({
        'Alpha': alpha,
        'Baseline_Cl': perf_baseline.loc[idx_base, 'cl'],
        'Baseline_Cd': perf_baseline.loc[idx_base, 'cd'],
        'Baseline_ClCd': perf_baseline.loc[idx_base, 'cl_cd'],
        'DOE_Cl': perf_doe.loc[idx_doe, 'cl'],
        'DOE_Cd': perf_doe.loc[idx_doe, 'cd'],
        'DOE_ClCd': perf_doe.loc[idx_doe, 'cl_cd'],
        'BO_Cl': perf_bo.loc[idx_bo, 'cl'],
        'BO_Cd': perf_bo.loc[idx_bo, 'cd'],
        'BO_ClCd': perf_bo.loc[idx_bo, 'cl_cd'],
        'Improvement_vs_Baseline_%':
            (perf_bo.loc[idx_bo, 'cl_cd'] / perf_baseline.loc[idx_base, 'cl_cd'] - 1) * 100,
        'Improvement_vs_DOE_%':
            (perf_bo.loc[idx_bo, 'cl_cd'] / perf_doe.loc[idx_doe, 'cl_cd'] - 1) * 100
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(data_dir + 'detailed_performance_comparison.csv', index=False)
print("✓ Detailed comparison saved: detailed_performance_comparison.csv")

# ============================================
# STEP 7: CREATE GEOMETRY COMPARISON TABLE
# ============================================
print("\nStep 7: Creating geometry comparison table...")

# Calculate key geometric parameters
def calc_geometry_params(x, y_upper, y_lower):
    thickness = y_upper - y_lower
    camber = (y_upper + y_lower) / 2

    max_thickness = thickness.max()
    max_thickness_loc = x[thickness.argmax()]
    max_camber = camber.max()
    max_camber_loc = x[camber.argmax()]
    leading_edge_radius = y_upper[0] * 2  # Approximation
    trailing_edge_thickness = thickness[-1]

    return {
        'Max_Thickness_%': max_thickness * 100,
        'Max_Thickness_Location_x/c': max_thickness_loc,
        'Max_Camber_%': max_camber * 100,
        'Max_Camber_Location_x/c': max_camber_loc,
        'LE_Radius_%': leading_edge_radius * 100,
        'TE_Thickness_%': trailing_edge_thickness * 100
    }

geom_baseline = calc_geometry_params(x_baseline, y_upper_baseline, y_lower_baseline)
geom_doe = calc_geometry_params(x_doe, y_upper_doe, y_lower_doe)
geom_bo = calc_geometry_params(x_bo, y_upper_bo, y_lower_bo)

geometry_df = pd.DataFrame({
    'Parameter': list(geom_baseline.keys()),
    'Baseline': list(geom_baseline.values()),
    'Best_DOE': list(geom_doe.values()),
    'Bayesian_Opt': list(geom_bo.values())
})

geometry_df.to_csv(data_dir + 'geometry_comparison.csv', index=False)
print("✓ Geometry comparison saved: geometry_comparison.csv")

# ============================================
# STEP 8: PRINT SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

print("\n📊 PERFORMANCE AT OPTIMAL ANGLE (α = 6°):")
print(f"{'Metric':<20} {'Baseline':<15} {'Best DOE':<15} {'Bayesian Opt':<15}")
print("-" * 70)

idx_base = (perf_baseline['alpha'] - 6).abs().idxmin()
idx_doe = (perf_doe['alpha'] - 6).abs().idxmin()
idx_bo = (perf_bo['alpha'] - 6).abs().idxmin()

print(f"{'Cl/Cd':<20} {perf_baseline.loc[idx_base, 'cl_cd']:<15.2f} "
      f"{perf_doe.loc[idx_doe, 'cl_cd']:<15.2f} "
      f"{perf_bo.loc[idx_bo, 'cl_cd']:<15.2f}")
print(f"{'Cl':<20} {perf_baseline.loc[idx_base, 'cl']:<15.4f} "
      f"{perf_doe.loc[idx_doe, 'cl']:<15.4f} "
      f"{perf_bo.loc[idx_bo, 'cl']:<15.4f}")
print(f"{'Cd':<20} {perf_baseline.loc[idx_base, 'cd']:<15.6f} "
      f"{perf_doe.loc[idx_doe, 'cd']:<15.6f} "
      f"{perf_bo.loc[idx_bo, 'cd']:<15.6f}")

print("\n🏆 MAXIMUM PERFORMANCE:")
print(f"{'Metric':<20} {'Baseline':<15} {'Best DOE':<15} {'Bayesian Opt':<15}")
print("-" * 70)
print(f"{'Max Cl/Cd':<20} {perf_baseline['cl_cd'].max():<15.2f} "
      f"{perf_doe['cl_cd'].max():<15.2f} "
      f"{perf_bo['cl_cd'].max():<15.2f}")
print(f"{'Max Cl':<20} {perf_baseline['cl'].max():<15.4f} "
      f"{perf_doe['cl'].max():<15.4f} "
      f"{perf_bo['cl'].max():<15.4f}")
print(f"{'Min Cd':<20} {perf_baseline['cd'].min():<15.6f} "
      f"{perf_doe['cd'].min():<15.6f} "
      f"{perf_bo['cd'].min():<15.6f}")

improvement_vs_baseline = (perf_bo['cl_cd'].max() / perf_baseline['cl_cd'].max() - 1) * 100
improvement_vs_doe = (perf_bo['cl_cd'].max() / perf_doe['cl_cd'].max() - 1) * 100

print(f"\n📈 IMPROVEMENTS:")
print(f"  Bayesian Opt vs Baseline: +{improvement_vs_baseline:.2f}%")
print(f"  Bayesian Opt vs Best DOE: +{improvement_vs_doe:.2f}%")

print("\n📁 FILES CREATED:")
print("  • comprehensive_design_comparison.png (10-panel visualization)")
print("  • detailed_performance_comparison.csv")
print("  • geometry_comparison.csv")

print("\n" + "="*70)
print("✅ COMPARISON VISUALIZATION COMPLETE!")
print("="*70)
