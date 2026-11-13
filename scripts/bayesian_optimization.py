# %%
# ============================================
# BAYESIAN OPTIMIZATION FOR AIRFOIL DESIGN
# Sequential Model-Based Optimization using Gaussian Processes
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import joblib
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt.utils import use_named_args
import time

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

print("="*60)
print("BAYESIAN OPTIMIZATION FOR AIRFOIL DESIGN")
print("Sequential Model-Based Optimization using Gaussian Processes")
print("="*60)

# ============================================
# CONFIGURATION
# ============================================

# Directory
data_dir = DATA_DIR

# Optimization settings
TARGET_ALPHA = 6.0  # Target angle of attack for optimization (degrees)
N_INITIAL_POINTS = 20  # Random exploration points
N_ITERATIONS = 100  # Total optimization iterations
ACQUISITION_FUNC = 'EI'  # Expected Improvement (or 'gp_hedge', 'LCB', 'PI')
N_RANDOM_STARTS = 3  # Number of random restarts for acquisition optimization

print(f"\nOptimization Configuration:")
print(f"  Target angle of attack: {TARGET_ALPHA}°")
print(f"  Initial random points: {N_INITIAL_POINTS}")
print(f"  Optimization iterations: {N_ITERATIONS}")
print(f"  Acquisition function: {ACQUISITION_FUNC}")

# ============================================
# STEP 1: LOAD TRAINED MODEL AND SCALERS
# ============================================
print("\nStep 1: Loading trained surrogate model...")

model = keras.models.load_model(data_dir + 'airfoil_surrogate_model.keras')
scaler_X = joblib.load(data_dir + 'scaler_X.pkl')
scaler_y = joblib.load(data_dir + 'scaler_y.pkl')

print("✓ Model and scalers loaded")

# ============================================
# STEP 2: DEFINE SEARCH SPACE
# ============================================
print("\nStep 2: Defining search space...")

# Load DOE parameters to determine bounds
df_params = pd.read_csv(data_dir + 'DOE_parameters.csv')

# Calculate bounds for each coefficient (with some margin for exploration)
coef_names = [f'upper_c{i}' for i in range(9)] + [f'lower_c{i}' for i in range(9)]
bounds = []

for coef in coef_names:
    min_val = df_params[coef].min()
    max_val = df_params[coef].max()
    # Add 10% margin for exploration beyond DOE space
    margin = 0.1 * (max_val - min_val)
    bounds.append(Real(min_val - margin, max_val + margin, name=coef))

print(f"✓ Search space defined with {len(bounds)} dimensions")
print(f"  Coefficient ranges:")
for i, (coef, bound) in enumerate(zip(coef_names, bounds)):
    if i < 3 or i >= len(coef_names) - 3:  # Show first 3 and last 3
        print(f"    {coef}: [{bound.low:.6f}, {bound.high:.6f}]")
    elif i == 3:
        print(f"    ...")

# ============================================
# STEP 3: DEFINE OBJECTIVE FUNCTION
# ============================================
print("\nStep 3: Defining objective function...")

# Track optimization history
history = {
    'iteration': [],
    'cl_cd': [],
    'cl': [],
    'cd': [],
    'cm': [],
    'coefficients': []
}

iteration_counter = [0]  # Use list to make it mutable in nested function

@use_named_args(bounds)
def objective(**params):
    """
    Objective function to MINIMIZE (negative Cl/Cd for maximization)

    Args:
        **params: Dictionary of upper_c0..upper_c8, lower_c0..lower_c8

    Returns:
        Negative Cl/Cd ratio (we minimize, so negate for maximization)
    """
    # Extract coefficient values in correct order
    coefficients = [params[coef] for coef in coef_names]

    # Create input: 18 coefficients + alpha
    X_input = np.array(coefficients + [TARGET_ALPHA]).reshape(1, -1)

    # Scale input
    X_scaled = scaler_X.transform(X_input)

    # Predict using surrogate model
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Extract predictions
    cl = y_pred[0, 0]
    cd = y_pred[0, 1]
    cm = y_pred[0, 2]

    # Calculate Cl/Cd (add small epsilon to avoid division by zero)
    cl_cd = cl / (cd + 1e-10)

    # Store in history
    iteration_counter[0] += 1
    history['iteration'].append(iteration_counter[0])
    history['cl_cd'].append(cl_cd)
    history['cl'].append(cl)
    history['cd'].append(cd)
    history['cm'].append(cm)
    history['coefficients'].append(coefficients.copy())

    # Print progress every 10 iterations
    if iteration_counter[0] % 10 == 0 or iteration_counter[0] <= N_INITIAL_POINTS:
        print(f"  Iteration {iteration_counter[0]:3d}: Cl/Cd = {cl_cd:6.2f}  "
              f"(Cl={cl:.4f}, Cd={cd:.6f})")

    # Return negative (because we minimize, but want to maximize Cl/Cd)
    return -cl_cd

print("✓ Objective function defined")

# ============================================
# STEP 4: RUN BAYESIAN OPTIMIZATION
# ============================================
print("\nStep 4: Running Bayesian Optimization...")
print(f"This will take approximately {(N_INITIAL_POINTS + N_ITERATIONS) * 0.1:.1f} seconds\n")

start_time = time.time()

# Run Gaussian Process-based Bayesian Optimization
result = gp_minimize(
    func=objective,                    # Objective function to minimize
    dimensions=bounds,                 # Search space
    n_calls=N_INITIAL_POINTS + N_ITERATIONS,  # Total evaluations
    n_initial_points=N_INITIAL_POINTS, # Random exploration
    acq_func=ACQUISITION_FUNC,         # Acquisition function
    n_restarts_optimizer=N_RANDOM_STARTS,  # Random restarts for acquisition
    random_state=42,                   # Reproducibility
    verbose=False                      # Suppress skopt output (we have our own)
)

elapsed_time = time.time() - start_time

print(f"\n✓ Optimization complete! ({elapsed_time:.1f} seconds)")

# ============================================
# STEP 5: EXTRACT BEST RESULT
# ============================================
print("\nStep 5: Extracting best design...")

best_idx = np.argmax(history['cl_cd'])
best_cl_cd = history['cl_cd'][best_idx]
best_cl = history['cl'][best_idx]
best_cd = history['cd'][best_idx]
best_cm = history['cm'][best_idx]
best_coefficients = history['coefficients'][best_idx]
best_iteration = history['iteration'][best_idx]

print(f"✓ Best design found at iteration {best_iteration}")
print(f"\nBest Bayesian Optimization Result:")
print(f"  Cl/Cd: {best_cl_cd:.2f}")
print(f"  Cl: {best_cl:.4f}")
print(f"  Cd: {best_cd:.6f}")
print(f"  Cm: {best_cm:.4f}")
print(f"  At α = {TARGET_ALPHA}°")

# ============================================
# STEP 6: COMPARE WITH BASELINE AND DOE
# ============================================
print("\nStep 6: Comparing with baseline and DOE...")

# Load baseline performance
df_results = pd.read_csv(data_dir + 'aerosandbox_results.csv')

# Baseline SG6043 at target alpha
baseline_data = df_results[(df_results['design_id'] == 0) &
                           (df_results['alpha'] == TARGET_ALPHA)]
if len(baseline_data) > 0:
    baseline_cl_cd = baseline_data.iloc[0]['cl_cd']
    baseline_cl = baseline_data.iloc[0]['cl']
    baseline_cd = baseline_data.iloc[0]['cd']
else:
    # Find closest alpha
    baseline_data = df_results[df_results['design_id'] == 0]
    closest_idx = (baseline_data['alpha'] - TARGET_ALPHA).abs().idxmin()
    baseline_cl_cd = baseline_data.loc[closest_idx, 'cl_cd']
    baseline_cl = baseline_data.loc[closest_idx, 'cl']
    baseline_cd = baseline_data.loc[closest_idx, 'cd']

# Best from DOE at target alpha
doe_at_target = df_results[df_results['alpha'] == TARGET_ALPHA]
if len(doe_at_target) == 0:
    # Find closest alpha
    alpha_diff = df_results.groupby('alpha').size().reset_index()
    closest_alpha = alpha_diff.iloc[(alpha_diff['alpha'] - TARGET_ALPHA).abs().idxmin()]['alpha']
    doe_at_target = df_results[df_results['alpha'] == closest_alpha]

best_doe_idx = doe_at_target['cl_cd'].idxmax()
best_doe = doe_at_target.loc[best_doe_idx]
best_doe_cl_cd = best_doe['cl_cd']
best_doe_id = best_doe['design_id']

# Calculate improvements
improvement_vs_baseline = (best_cl_cd / baseline_cl_cd - 1) * 100
improvement_vs_doe = (best_cl_cd / best_doe_cl_cd - 1) * 100

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

print(f"\nBaseline SG6043:")
print(f"  Cl/Cd: {baseline_cl_cd:.2f}")
print(f"  Cl: {baseline_cl:.4f}, Cd: {baseline_cd:.6f}")

print(f"\nBest DOE Design (ID {int(best_doe_id)}):")
print(f"  Cl/Cd: {best_doe_cl_cd:.2f}")

print(f"\nBayesian Optimization Result:")
print(f"  Cl/Cd: {best_cl_cd:.2f}")
print(f"  🚀 Improvement vs Baseline: {improvement_vs_baseline:+.2f}%")
print(f"  🎯 Improvement vs Best DOE: {improvement_vs_doe:+.2f}%")

# ============================================
# STEP 7: VISUALIZATIONS
# ============================================
print("\nStep 7: Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Convergence - Cl/Cd over iterations
ax1 = plt.subplot(3, 3, 1)
iterations = history['iteration']
cl_cd_values = history['cl_cd']
best_so_far = np.maximum.accumulate(cl_cd_values)

ax1.plot(iterations, cl_cd_values, 'b.', alpha=0.3, label='Samples', markersize=8)
ax1.plot(iterations, best_so_far, 'r-', linewidth=3, label='Best so far')
ax1.axvline(N_INITIAL_POINTS, color='green', linestyle='--',
            linewidth=2, label='End of random initialization')
ax1.axhline(baseline_cl_cd, color='orange', linestyle='--',
            linewidth=2, label=f'Baseline ({baseline_cl_cd:.1f})')
ax1.axhline(best_doe_cl_cd, color='purple', linestyle='--',
            linewidth=2, label=f'Best DOE ({best_doe_cl_cd:.1f})')
ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax1.set_title('Bayesian Optimization Convergence', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Cl vs Cd scatter
ax2 = plt.subplot(3, 3, 2)
cl_values = history['cl']
cd_values = history['cd']
colors = plt.cm.viridis(np.linspace(0, 1, len(cl_values)))

ax2.scatter(cd_values, cl_values, c=iterations, cmap='viridis',
            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.scatter(best_cd, best_cl, c='red', s=300, marker='*',
            edgecolors='black', linewidth=2, label='Best BO', zorder=5)
ax2.scatter(baseline_cd, baseline_cl, c='orange', s=200, marker='s',
            edgecolors='black', linewidth=2, label='Baseline', zorder=5)

cbar = plt.colorbar(ax2.scatter(cd_values, cl_values, c=iterations,
                                cmap='viridis', s=50, alpha=0.6), ax=ax2)
cbar.set_label('Iteration', fontsize=10)

ax2.set_xlabel('Cd (Drag Coefficient)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cl (Lift Coefficient)', fontsize=12, fontweight='bold')
ax2.set_title('Design Space Exploration', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Improvement over iterations
ax3 = plt.subplot(3, 3, 3)
improvement_vs_baseline_history = [(x / baseline_cl_cd - 1) * 100 for x in cl_cd_values]
improvement_best_so_far = [(x / baseline_cl_cd - 1) * 100 for x in best_so_far]

ax3.plot(iterations, improvement_vs_baseline_history, 'b.', alpha=0.3, markersize=8)
ax3.plot(iterations, improvement_best_so_far, 'r-', linewidth=3)
ax3.axhline(0, color='orange', linestyle='--', linewidth=2)
ax3.axvline(N_INITIAL_POINTS, color='green', linestyle='--', linewidth=2)
ax3.fill_between(iterations, 0, improvement_best_so_far, alpha=0.2, color='green')
ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax3.set_ylabel('Improvement vs Baseline (%)', fontsize=12, fontweight='bold')
ax3.set_title('Optimization Progress', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Cl history
ax4 = plt.subplot(3, 3, 4)
ax4.plot(iterations, cl_values, 'b-', alpha=0.5, linewidth=1)
ax4.scatter(iterations, cl_values, c=iterations, cmap='viridis', s=30, alpha=0.6)
ax4.axhline(best_cl, color='red', linestyle='--', linewidth=2, label=f'Best: {best_cl:.4f}')
ax4.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cl (Lift Coefficient)', fontsize=12, fontweight='bold')
ax4.set_title('Lift Coefficient Evolution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Cd history
ax5 = plt.subplot(3, 3, 5)
ax5.plot(iterations, cd_values, 'r-', alpha=0.5, linewidth=1)
ax5.scatter(iterations, cd_values, c=iterations, cmap='viridis', s=30, alpha=0.6)
ax5.axhline(best_cd, color='red', linestyle='--', linewidth=2, label=f'Best: {best_cd:.6f}')
ax5.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax5.set_ylabel('Cd (Drag Coefficient)', fontsize=12, fontweight='bold')
ax5.set_title('Drag Coefficient Evolution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: Coefficient distributions (sample)
ax6 = plt.subplot(3, 3, 6)
# Show distribution of first upper coefficient
upper_c0_values = [coefs[0] for coefs in history['coefficients']]
ax6.hist(upper_c0_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax6.axvline(best_coefficients[0], color='red', linestyle='--',
            linewidth=3, label=f'Best: {best_coefficients[0]:.4f}')
baseline_upper_c0 = df_params[df_params['design_id'] == 0]['upper_c0'].iloc[0]
ax6.axvline(baseline_upper_c0, color='orange', linestyle='--',
            linewidth=2, label=f'Baseline: {baseline_upper_c0:.4f}')
ax6.set_xlabel('upper_c0 value', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('Parameter Exploration (upper_c0)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

# Plot 7: Acquisition function visualization (Expected Improvement)
ax7 = plt.subplot(3, 3, 7)
# Plot the expected improvement based on distance from best
distances_from_best = []
for coefs in history['coefficients']:
    dist = np.linalg.norm(np.array(coefs) - np.array(best_coefficients))
    distances_from_best.append(dist)

ax7.scatter(distances_from_best, cl_cd_values, c=iterations,
            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax7.set_xlabel('Distance from Best Design', fontsize=12, fontweight='bold')
ax7.set_ylabel('Cl/Cd', fontsize=12, fontweight='bold')
ax7.set_title('Exploration vs Exploitation', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Performance distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(cl_cd_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax8.axvline(best_cl_cd, color='red', linestyle='--',
            linewidth=3, label=f'Best BO: {best_cl_cd:.1f}')
ax8.axvline(baseline_cl_cd, color='orange', linestyle='--',
            linewidth=2, label=f'Baseline: {baseline_cl_cd:.1f}')
ax8.axvline(best_doe_cl_cd, color='purple', linestyle='--',
            linewidth=2, label=f'Best DOE: {best_doe_cl_cd:.1f}')
ax8.set_xlabel('Cl/Cd', fontsize=12, fontweight='bold')
ax8.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax8.set_title('Performance Distribution', fontsize=13, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Plot 9: Summary text
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
╔═══════════════════════════════════════╗
║  BAYESIAN OPTIMIZATION RESULTS        ║
╚═══════════════════════════════════════╝

Target: α = {TARGET_ALPHA}°

Baseline SG6043:
  Cl/Cd: {baseline_cl_cd:.2f}
  Cl: {baseline_cl:.4f}
  Cd: {baseline_cd:.6f}

Best DOE Design (ID {int(best_doe_id)}):
  Cl/Cd: {best_doe_cl_cd:.2f}

Bayesian Optimization:
  Cl/Cd: {best_cl_cd:.2f}
  Cl: {best_cl:.4f}
  Cd: {best_cd:.6f}
  Cm: {best_cm:.4f}

🚀 Improvement:
  vs Baseline: {improvement_vs_baseline:+.2f}%
  vs Best DOE: {improvement_vs_doe:+.2f}%

Optimization Details:
  Total evaluations: {len(iterations)}
  Random init: {N_INITIAL_POINTS}
  BO iterations: {N_ITERATIONS}
  Best found at: Iteration {best_iteration}
  Acquisition: {ACQUISITION_FUNC}
  Time: {elapsed_time:.1f}s

Efficiency:
  {len(iterations) / elapsed_time:.1f} evals/sec
"""

ax9.text(0.05, 0.5, summary_text, fontsize=9,
         verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(data_dir + 'bayesian_optimization_results.png', dpi=200, bbox_inches='tight')
print(f"✓ Main visualization saved: bayesian_optimization_results.png")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================
print("\nStep 8: Saving results...")

# Save optimization history
history_df = pd.DataFrame({
    'iteration': history['iteration'],
    'cl_cd': history['cl_cd'],
    'cl': history['cl'],
    'cd': history['cd'],
    'cm': history['cm']
})

# Add coefficient columns
for i, coef_name in enumerate(coef_names):
    history_df[coef_name] = [coefs[i] for coefs in history['coefficients']]

history_df.to_csv(data_dir + 'bayesian_optimization_history.csv', index=False)
print(f"✓ History saved: bayesian_optimization_history.csv")

# Save best design coefficients
best_design_df = pd.DataFrame({
    'design_id': ['BO_best'],
    **{coef_name: [best_coefficients[i]] for i, coef_name in enumerate(coef_names)}
})
best_design_df.to_csv(data_dir + 'bayesian_best_design.csv', index=False)
print(f"✓ Best design saved: bayesian_best_design.csv")

# Save comparison summary
comparison_df = pd.DataFrame({
    'Method': ['Baseline SG6043', f'Best DOE (ID {int(best_doe_id)})', 'Bayesian Optimization'],
    'Cl/Cd': [baseline_cl_cd, best_doe_cl_cd, best_cl_cd],
    'Cl': [baseline_cl, best_doe['cl'], best_cl],
    'Cd': [baseline_cd, best_doe['cd'], best_cd],
    'Improvement_vs_Baseline_%': [0.0, (best_doe_cl_cd/baseline_cl_cd-1)*100, improvement_vs_baseline]
})
comparison_df.to_csv(data_dir + 'bayesian_optimization_comparison.csv', index=False)
print(f"✓ Comparison saved: bayesian_optimization_comparison.csv")

# ============================================
# STEP 9: GENERATE AIRFOIL COORDINATES
# ============================================
print("\nStep 9: Generating optimized airfoil coordinates...")

# Load the parametrization code to generate coordinates from coefficients
try:
    # Try to import the parametrization functions
    import sys
    sys.path.insert(0, data_dir)

    # We'll create a simple function to generate coordinates
    # using the same Chebyshev polynomial approach

    def chebyshev_polynomials(x, n_order=8):
        """Generate Chebyshev polynomials up to order n"""
        T = np.zeros((len(x), n_order + 1))
        T[:, 0] = 1
        T[:, 1] = x
        for i in range(2, n_order + 1):
            T[:, i] = 2 * x * T[:, i-1] - T[:, i-2]
        return T

    def reconstruct_airfoil(upper_coeffs, lower_coeffs, n_points=100):
        """Reconstruct airfoil coordinates from Chebyshev coefficients"""
        # Cosine spacing for better resolution near leading/trailing edges
        theta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(theta))

        # Generate Chebyshev polynomials
        # Transform x from [0,1] to [-1,1] for Chebyshev
        x_cheb = 2 * x - 1
        T = chebyshev_polynomials(x_cheb, n_order=8)

        # Calculate y coordinates
        y_upper = T @ np.array(upper_coeffs)
        y_lower = T @ np.array(lower_coeffs)

        return x, y_upper, y_lower

    # Generate coordinates for best design
    upper_coeffs = best_coefficients[:9]
    lower_coeffs = best_coefficients[9:]

    x, y_upper, y_lower = reconstruct_airfoil(upper_coeffs, lower_coeffs)

    # Save as .dat file (XFOIL format)
    with open(data_dir + 'sg6043_bayesian_optimized.dat', 'w') as f:
        f.write("SG6043_BAYESIAN_OPTIMIZED\n")
        # Upper surface (trailing edge to leading edge)
        for i in range(len(x)-1, -1, -1):
            f.write(f"  {x[i]:.6f}  {y_upper[i]:.6f}\n")
        # Lower surface (leading edge to trailing edge)
        for i in range(len(x)):
            f.write(f"  {x[i]:.6f}  {y_lower[i]:.6f}\n")

    print(f"✓ Airfoil coordinates saved: sg6043_bayesian_optimized.dat")

    # Create airfoil shape comparison plot
    fig_airfoil = plt.figure(figsize=(14, 6))

    # Plot optimized airfoil
    ax_airfoil1 = plt.subplot(1, 2, 1)
    ax_airfoil1.plot(x, y_upper, 'b-', linewidth=2, label='Upper surface')
    ax_airfoil1.plot(x, y_lower, 'r-', linewidth=2, label='Lower surface')
    ax_airfoil1.fill_between(x, y_lower, y_upper, alpha=0.2, color='gray')
    ax_airfoil1.set_xlabel('x/c', fontsize=12, fontweight='bold')
    ax_airfoil1.set_ylabel('y/c', fontsize=12, fontweight='bold')
    ax_airfoil1.set_title('Bayesian Optimized Airfoil Shape', fontsize=13, fontweight='bold')
    ax_airfoil1.legend(fontsize=10)
    ax_airfoil1.grid(True, alpha=0.3)
    ax_airfoil1.set_aspect('equal')
    ax_airfoil1.set_xlim(-0.05, 1.05)

    # Plot comparison with baseline
    ax_airfoil2 = plt.subplot(1, 2, 2)

    # Load baseline coefficients
    baseline_coeffs = df_params[df_params['design_id'] == 0].iloc[0]
    baseline_upper = [baseline_coeffs[f'upper_c{i}'] for i in range(9)]
    baseline_lower = [baseline_coeffs[f'lower_c{i}'] for i in range(9)]
    x_base, y_upper_base, y_lower_base = reconstruct_airfoil(baseline_upper, baseline_lower)

    ax_airfoil2.plot(x_base, y_upper_base, 'b--', linewidth=2, label='Baseline upper', alpha=0.7)
    ax_airfoil2.plot(x_base, y_lower_base, 'r--', linewidth=2, label='Baseline lower', alpha=0.7)
    ax_airfoil2.plot(x, y_upper, 'b-', linewidth=2, label='Optimized upper')
    ax_airfoil2.plot(x, y_lower, 'r-', linewidth=2, label='Optimized lower')
    ax_airfoil2.set_xlabel('x/c', fontsize=12, fontweight='bold')
    ax_airfoil2.set_ylabel('y/c', fontsize=12, fontweight='bold')
    ax_airfoil2.set_title('Baseline vs Optimized Comparison', fontsize=13, fontweight='bold')
    ax_airfoil2.legend(fontsize=10)
    ax_airfoil2.grid(True, alpha=0.3)
    ax_airfoil2.set_aspect('equal')
    ax_airfoil2.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(data_dir + 'bayesian_airfoil_shape.png', dpi=200, bbox_inches='tight')
    print(f"✓ Airfoil shape visualization saved: bayesian_airfoil_shape.png")

except Exception as e:
    print(f"⚠ Could not generate airfoil coordinates: {e}")
    print("  (This is optional - optimization results are still valid)")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 BAYESIAN OPTIMIZATION COMPLETE!")
print("="*60)

print(f"\n✨ Results Summary:")
print(f"  🏆 Best Cl/Cd: {best_cl_cd:.2f} (at α={TARGET_ALPHA}°)")
print(f"  📈 Improvement vs Baseline: {improvement_vs_baseline:+.2f}%")
print(f"  🎯 Improvement vs Best DOE: {improvement_vs_doe:+.2f}%")
print(f"  ⚡ Evaluations: {len(iterations)} total ({N_INITIAL_POINTS} random + {N_ITERATIONS} BO)")
print(f"  ⏱️  Time: {elapsed_time:.1f} seconds")

print(f"\n📁 Files Created:")
print(f"  • bayesian_optimization_results.png")
print(f"  • bayesian_optimization_history.csv")
print(f"  • bayesian_best_design.csv")
print(f"  • bayesian_optimization_comparison.csv")
print(f"  • bayesian_airfoil_shape.png")
print(f"  • sg6043_bayesian_optimized.dat")

print(f"\n🔬 Key Insights:")
if improvement_vs_doe > 1.0:
    print(f"  ✓ BO found a design BETTER than all 300 DOE designs!")
    print(f"  ✓ This demonstrates the power of intelligent search")
elif improvement_vs_doe > 0:
    print(f"  ✓ BO matched/slightly improved the best DOE design")
    print(f"  ✓ Try increasing N_ITERATIONS for more exploration")
else:
    print(f"  ✓ Best DOE design is still superior")
    print(f"  ✓ Consider expanding search bounds or different alpha values")

print(f"\n💡 Next Steps:")
print(f"  1. Validate best design with XFOIL: sg6043_bayesian_optimized.dat")
print(f"  2. Try multi-objective optimization (Cl AND Cd)")
print(f"  3. Optimize for multiple angles of attack simultaneously")
print(f"  4. Run more iterations ({N_ITERATIONS*2}+) for better exploration")

print("="*60)
