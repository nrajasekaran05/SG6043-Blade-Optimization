# %%
# ============================================
# ADVANCED BAYESIAN OPTIMIZATION FOR AIRFOIL DESIGN
# Multi-Objective and Multi-Alpha Optimization
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
from skopt.utils import use_named_args
import time

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

print("="*60)
print("ADVANCED BAYESIAN OPTIMIZATION FOR AIRFOIL DESIGN")
print("Multi-Objective & Multi-Alpha Optimization")
print("="*60)

# ============================================
# CONFIGURATION
# ============================================

# Directory
data_dir = DATA_DIR

# OPTIMIZATION MODE SELECTION
OPTIMIZATION_MODE = 'multi_alpha'  # Options: 'multi_objective', 'multi_alpha', 'robust'

# Multi-alpha settings
ALPHA_RANGE = np.linspace(2, 10, 5)  # Optimize across multiple angles
ALPHA_WEIGHTS = None  # None for equal weights, or array like [0.1, 0.2, 0.4, 0.2, 0.1]

# Multi-objective settings
OBJECTIVE_WEIGHTS = {
    'cl': 0.6,      # Weight for maximizing lift
    'cd': -0.4,     # Weight for minimizing drag (negative = minimize)
    'cl_cd': 0.0    # Can also directly weight Cl/Cd ratio
}

# Optimization settings
N_INITIAL_POINTS = 25
N_ITERATIONS = 150
ACQUISITION_FUNC = 'EI'  # Expected Improvement

print(f"\nOptimization Configuration:")
print(f"  Mode: {OPTIMIZATION_MODE}")
if OPTIMIZATION_MODE == 'multi_alpha':
    print(f"  Alpha range: {ALPHA_RANGE[0]:.1f}° to {ALPHA_RANGE[-1]:.1f}° ({len(ALPHA_RANGE)} points)")
elif OPTIMIZATION_MODE == 'multi_objective':
    print(f"  Objective weights: Cl={OBJECTIVE_WEIGHTS['cl']}, "
          f"Cd={OBJECTIVE_WEIGHTS['cd']}, Cl/Cd={OBJECTIVE_WEIGHTS['cl_cd']}")
print(f"  Initial points: {N_INITIAL_POINTS}")
print(f"  Optimization iterations: {N_ITERATIONS}")

# ============================================
# STEP 1: LOAD MODEL AND DATA
# ============================================
print("\nStep 1: Loading trained surrogate model...")

model = keras.models.load_model(data_dir + 'airfoil_surrogate_model.keras')
scaler_X = joblib.load(data_dir + 'scaler_X.pkl')
scaler_y = joblib.load(data_dir + 'scaler_y.pkl')

print("✓ Model and scalers loaded")

df_params = pd.read_csv(data_dir + 'DOE_parameters.csv')
df_results = pd.read_csv(data_dir + 'aerosandbox_results.csv')

# ============================================
# STEP 2: DEFINE SEARCH SPACE
# ============================================
print("\nStep 2: Defining search space...")

coef_names = [f'upper_c{i}' for i in range(9)] + [f'lower_c{i}' for i in range(9)]
bounds = []

for coef in coef_names:
    min_val = df_params[coef].min()
    max_val = df_params[coef].max()
    margin = 0.1 * (max_val - min_val)
    bounds.append(Real(min_val - margin, max_val + margin, name=coef))

print(f"✓ Search space defined: {len(bounds)} dimensions")

# ============================================
# STEP 3: DEFINE OBJECTIVE FUNCTIONS
# ============================================
print("\nStep 3: Defining objective function...")

history = {
    'iteration': [],
    'objective': [],
    'cl_cd_avg': [],
    'cl_avg': [],
    'cd_avg': [],
    'coefficients': [],
    'alpha_results': []  # Store results at each alpha
}

iteration_counter = [0]

def predict_performance(coefficients, alpha):
    """Predict aerodynamic performance for given coefficients and alpha"""
    X_input = np.array(coefficients + [alpha]).reshape(1, -1)
    X_scaled = scaler_X.transform(X_input)
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    cl = y_pred[0, 0]
    cd = y_pred[0, 1]
    cm = y_pred[0, 2]

    return cl, cd, cm

@use_named_args(bounds)
def objective_multi_alpha(**params):
    """
    Optimize for robust performance across multiple angles of attack

    Goal: Maximize MINIMUM Cl/Cd across the alpha range (robust design)
    or maximize AVERAGE Cl/Cd (best overall performance)
    """
    coefficients = [params[coef] for coef in coef_names]

    cl_cd_values = []
    cl_values = []
    cd_values = []
    alpha_results = {}

    # Evaluate at each alpha
    for alpha in ALPHA_RANGE:
        cl, cd, cm = predict_performance(coefficients, alpha)
        cl_cd = cl / (cd + 1e-10)

        cl_cd_values.append(cl_cd)
        cl_values.append(cl)
        cd_values.append(cd)
        alpha_results[f'alpha_{alpha:.1f}'] = {'cl': cl, 'cd': cd, 'cl_cd': cl_cd}

    # Compute weighted or robust objective
    if ALPHA_WEIGHTS is not None:
        # Weighted average
        objective_value = np.average(cl_cd_values, weights=ALPHA_WEIGHTS)
    else:
        # Robust: maximize the minimum Cl/Cd (max-min approach)
        # Or use average for best overall performance
        # objective_value = np.min(cl_cd_values)  # Robust (conservative)
        objective_value = np.mean(cl_cd_values)  # Average (aggressive)

    # Store history
    iteration_counter[0] += 1
    history['iteration'].append(iteration_counter[0])
    history['objective'].append(objective_value)
    history['cl_cd_avg'].append(np.mean(cl_cd_values))
    history['cl_avg'].append(np.mean(cl_values))
    history['cd_avg'].append(np.mean(cd_values))
    history['coefficients'].append(coefficients.copy())
    history['alpha_results'].append(alpha_results)

    if iteration_counter[0] % 10 == 0 or iteration_counter[0] <= N_INITIAL_POINTS:
        print(f"  Iter {iteration_counter[0]:3d}: Avg Cl/Cd = {np.mean(cl_cd_values):6.2f}  "
              f"Range: [{np.min(cl_cd_values):.1f}, {np.max(cl_cd_values):.1f}]")

    return -objective_value  # Minimize negative = maximize

@use_named_args(bounds)
def objective_multi_objective(**params):
    """
    Multi-objective optimization: balance Cl (lift) and Cd (drag)

    Allows explicit weighting of objectives beyond just Cl/Cd ratio
    """
    coefficients = [params[coef] for coef in coef_names]

    # Evaluate at a representative alpha (or average over several)
    alpha = 6.0  # Representative cruise angle
    cl, cd, cm = predict_performance(coefficients, alpha)
    cl_cd = cl / (cd + 1e-10)

    # Weighted multi-objective score
    objective_value = (
        OBJECTIVE_WEIGHTS['cl'] * cl +           # Maximize Cl
        OBJECTIVE_WEIGHTS['cd'] * cd * 100 +     # Minimize Cd (scaled)
        OBJECTIVE_WEIGHTS['cl_cd'] * cl_cd       # Maximize Cl/Cd ratio
    )

    # Store history
    iteration_counter[0] += 1
    history['iteration'].append(iteration_counter[0])
    history['objective'].append(objective_value)
    history['cl_cd_avg'].append(cl_cd)
    history['cl_avg'].append(cl)
    history['cd_avg'].append(cd)
    history['coefficients'].append(coefficients.copy())
    history['alpha_results'].append({f'alpha_{alpha}': {'cl': cl, 'cd': cd, 'cl_cd': cl_cd}})

    if iteration_counter[0] % 10 == 0 or iteration_counter[0] <= N_INITIAL_POINTS:
        print(f"  Iter {iteration_counter[0]:3d}: Objective = {objective_value:6.2f}  "
              f"Cl/Cd = {cl_cd:6.2f} (Cl={cl:.4f}, Cd={cd:.6f})")

    return -objective_value

@use_named_args(bounds)
def objective_robust(**params):
    """
    Robust optimization: maximize worst-case performance

    Find design that performs well even at unfavorable conditions
    """
    coefficients = [params[coef] for coef in coef_names]

    # Evaluate at critical alphas (low, cruise, high)
    critical_alphas = [2, 6, 10]
    cl_cd_values = []
    alpha_results = {}

    for alpha in critical_alphas:
        cl, cd, cm = predict_performance(coefficients, alpha)
        cl_cd = cl / (cd + 1e-10)
        cl_cd_values.append(cl_cd)
        alpha_results[f'alpha_{alpha}'] = {'cl': cl, 'cd': cd, 'cl_cd': cl_cd}

    # Worst-case performance (max-min)
    worst_case_cl_cd = np.min(cl_cd_values)
    avg_cl_cd = np.mean(cl_cd_values)

    # Hybrid objective: 70% worst-case + 30% average
    objective_value = 0.7 * worst_case_cl_cd + 0.3 * avg_cl_cd

    # Store history
    iteration_counter[0] += 1
    history['iteration'].append(iteration_counter[0])
    history['objective'].append(objective_value)
    history['cl_cd_avg'].append(avg_cl_cd)
    history['cl_avg'].append(np.mean([r['cl'] for r in alpha_results.values()]))
    history['cd_avg'].append(np.mean([r['cd'] for r in alpha_results.values()]))
    history['coefficients'].append(coefficients.copy())
    history['alpha_results'].append(alpha_results)

    if iteration_counter[0] % 10 == 0 or iteration_counter[0] <= N_INITIAL_POINTS:
        print(f"  Iter {iteration_counter[0]:3d}: Robust Obj = {objective_value:6.2f}  "
              f"Worst = {worst_case_cl_cd:.1f}, Avg = {avg_cl_cd:.1f}")

    return -objective_value

# Select objective based on mode
if OPTIMIZATION_MODE == 'multi_alpha':
    objective_func = objective_multi_alpha
    print("✓ Multi-alpha objective (robust performance across angles)")
elif OPTIMIZATION_MODE == 'multi_objective':
    objective_func = objective_multi_objective
    print("✓ Multi-objective (weighted Cl, Cd, Cl/Cd)")
elif OPTIMIZATION_MODE == 'robust':
    objective_func = objective_robust
    print("✓ Robust objective (max-min worst-case)")
else:
    raise ValueError(f"Unknown mode: {OPTIMIZATION_MODE}")

# ============================================
# STEP 4: RUN BAYESIAN OPTIMIZATION
# ============================================
print("\nStep 4: Running Advanced Bayesian Optimization...")
print(f"This will take approximately {(N_INITIAL_POINTS + N_ITERATIONS) * 0.15:.1f} seconds\n")

start_time = time.time()

result = gp_minimize(
    func=objective_func,
    dimensions=bounds,
    n_calls=N_INITIAL_POINTS + N_ITERATIONS,
    n_initial_points=N_INITIAL_POINTS,
    acq_func=ACQUISITION_FUNC,
    n_restarts_optimizer=3,
    random_state=42,
    verbose=False
)

elapsed_time = time.time() - start_time

print(f"\n✓ Optimization complete! ({elapsed_time:.1f} seconds)")

# ============================================
# STEP 5: EXTRACT AND ANALYZE RESULTS
# ============================================
print("\nStep 5: Analyzing results...")

best_idx = np.argmax(history['objective'])
best_objective = history['objective'][best_idx]
best_cl_cd_avg = history['cl_cd_avg'][best_idx]
best_coefficients = history['coefficients'][best_idx]
best_iteration = history['iteration'][best_idx]
best_alpha_results = history['alpha_results'][best_idx]

print(f"✓ Best design found at iteration {best_iteration}")
print(f"\nBest Design Performance:")
print(f"  Average Cl/Cd: {best_cl_cd_avg:.2f}")

if OPTIMIZATION_MODE == 'multi_alpha' or OPTIMIZATION_MODE == 'robust':
    print(f"\n  Performance across angles:")
    for alpha_key, results in sorted(best_alpha_results.items()):
        alpha_val = float(alpha_key.split('_')[1])
        print(f"    α={alpha_val:4.1f}°: Cl/Cd={results['cl_cd']:6.2f}  "
              f"(Cl={results['cl']:.4f}, Cd={results['cd']:.6f})")

    cl_cd_values_best = [r['cl_cd'] for r in best_alpha_results.values()]
    print(f"\n  Min Cl/Cd: {np.min(cl_cd_values_best):.2f}")
    print(f"  Max Cl/Cd: {np.max(cl_cd_values_best):.2f}")
    print(f"  Range: {np.max(cl_cd_values_best) - np.min(cl_cd_values_best):.2f}")

# ============================================
# STEP 6: COMPARE WITH BASELINE
# ============================================
print("\nStep 6: Comparing with baseline...")

baseline_coeffs = df_params[df_params['design_id'] == 0].iloc[0]
baseline_coefs_list = [baseline_coeffs[coef] for coef in coef_names]

if OPTIMIZATION_MODE == 'multi_alpha' or OPTIMIZATION_MODE == 'robust':
    # Evaluate baseline at same alphas
    baseline_results = {}
    for alpha_key in best_alpha_results.keys():
        alpha_val = float(alpha_key.split('_')[1])
        cl, cd, cm = predict_performance(baseline_coefs_list, alpha_val)
        baseline_results[alpha_key] = {'cl': cl, 'cd': cd, 'cl_cd': cl/(cd+1e-10)}

    baseline_cl_cd_avg = np.mean([r['cl_cd'] for r in baseline_results.values()])
    improvement = (best_cl_cd_avg / baseline_cl_cd_avg - 1) * 100

    print(f"\nBaseline SG6043:")
    print(f"  Average Cl/Cd: {baseline_cl_cd_avg:.2f}")
    for alpha_key, results in sorted(baseline_results.items()):
        alpha_val = float(alpha_key.split('_')[1])
        print(f"    α={alpha_val:4.1f}°: Cl/Cd={results['cl_cd']:6.2f}")

    print(f"\nOptimized Design:")
    print(f"  Average Cl/Cd: {best_cl_cd_avg:.2f}")
    print(f"  🚀 Improvement: {improvement:+.2f}%")
else:
    # Single alpha comparison
    alpha = 6.0
    cl_base, cd_base, _ = predict_performance(baseline_coefs_list, alpha)
    baseline_cl_cd = cl_base / (cd_base + 1e-10)
    improvement = (best_cl_cd_avg / baseline_cl_cd - 1) * 100

    print(f"  Baseline Cl/Cd: {baseline_cl_cd:.2f}")
    print(f"  Optimized Cl/Cd: {best_cl_cd_avg:.2f}")
    print(f"  🚀 Improvement: {improvement:+.2f}%")

# ============================================
# STEP 7: VISUALIZATIONS
# ============================================
print("\nStep 7: Creating visualizations...")

fig = plt.figure(figsize=(18, 14))

# Plot 1: Convergence
ax1 = plt.subplot(3, 3, 1)
iterations = history['iteration']
objectives = history['objective']
best_so_far = np.maximum.accumulate(objectives)

ax1.plot(iterations, objectives, 'b.', alpha=0.3, markersize=6)
ax1.plot(iterations, best_so_far, 'r-', linewidth=3, label='Best so far')
ax1.axvline(N_INITIAL_POINTS, color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('Iteration', fontweight='bold')
ax1.set_ylabel('Objective Value', fontweight='bold')
ax1.set_title(f'Convergence ({OPTIMIZATION_MODE})', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Average Cl/Cd over iterations
ax2 = plt.subplot(3, 3, 2)
cl_cd_avg_values = history['cl_cd_avg']
best_cl_cd_so_far = np.maximum.accumulate(cl_cd_avg_values)

ax2.plot(iterations, cl_cd_avg_values, 'b.', alpha=0.3, markersize=6)
ax2.plot(iterations, best_cl_cd_so_far, 'r-', linewidth=3)
ax2.axhline(baseline_cl_cd_avg if 'baseline_cl_cd_avg' in locals() else best_cl_cd_avg*0.95,
            color='orange', linestyle='--', linewidth=2, label='Baseline')
ax2.set_xlabel('Iteration', fontweight='bold')
ax2.set_ylabel('Average Cl/Cd', fontweight='bold')
ax2.set_title('Average Efficiency Evolution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Performance across alphas (if multi-alpha)
ax3 = plt.subplot(3, 3, 3)
if OPTIMIZATION_MODE in ['multi_alpha', 'robust']:
    # Plot best design vs baseline across alphas
    alphas_plot = sorted([float(k.split('_')[1]) for k in best_alpha_results.keys()])
    best_cl_cd_plot = [best_alpha_results[f'alpha_{a}']['cl_cd'] for a in alphas_plot]

    if 'baseline_results' in locals():
        baseline_cl_cd_plot = [baseline_results[f'alpha_{a}']['cl_cd'] for a in alphas_plot]
        ax3.plot(alphas_plot, baseline_cl_cd_plot, 'b--o', linewidth=2,
                markersize=8, label='Baseline')

    ax3.plot(alphas_plot, best_cl_cd_plot, 'r-s', linewidth=3,
            markersize=8, label='Optimized')
    ax3.set_xlabel('Angle of Attack (deg)', fontweight='bold')
    ax3.set_ylabel('Cl/Cd', fontweight='bold')
    ax3.set_title('Performance Across Angles', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    # Cl vs Cd scatter
    cl_vals = history['cl_avg']
    cd_vals = history['cd_avg']
    ax3.scatter(cd_vals, cl_vals, c=iterations, cmap='viridis',
                s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Cd', fontweight='bold')
    ax3.set_ylabel('Cl', fontweight='bold')
    ax3.set_title('Design Space Exploration', fontweight='bold')
    ax3.grid(True, alpha=0.3)

# Plot 4: Cl evolution
ax4 = plt.subplot(3, 3, 4)
ax4.plot(iterations, history['cl_avg'], 'b-', alpha=0.5)
ax4.scatter(iterations, history['cl_avg'], c=iterations,
            cmap='viridis', s=30, alpha=0.6)
ax4.set_xlabel('Iteration', fontweight='bold')
ax4.set_ylabel('Average Cl', fontweight='bold')
ax4.set_title('Lift Coefficient Evolution', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Cd evolution
ax5 = plt.subplot(3, 3, 5)
ax5.plot(iterations, history['cd_avg'], 'r-', alpha=0.5)
ax5.scatter(iterations, history['cd_avg'], c=iterations,
            cmap='viridis', s=30, alpha=0.6)
ax5.set_xlabel('Iteration', fontweight='bold')
ax5.set_ylabel('Average Cd', fontweight='bold')
ax5.set_title('Drag Coefficient Evolution', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Performance distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(cl_cd_avg_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
ax6.axvline(best_cl_cd_avg, color='red', linestyle='--', linewidth=3,
            label=f'Best: {best_cl_cd_avg:.1f}')
if 'baseline_cl_cd_avg' in locals():
    ax6.axvline(baseline_cl_cd_avg, color='orange', linestyle='--', linewidth=2,
                label=f'Baseline: {baseline_cl_cd_avg:.1f}')
ax6.set_xlabel('Average Cl/Cd', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('Performance Distribution', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Robustness metric (if multi-alpha)
ax7 = plt.subplot(3, 3, 7)
if OPTIMIZATION_MODE in ['multi_alpha', 'robust']:
    # Calculate range/std for each iteration
    robustness_metric = []
    for alpha_res in history['alpha_results']:
        cl_cd_vals = [r['cl_cd'] for r in alpha_res.values()]
        robustness_metric.append(np.std(cl_cd_vals))  # Lower std = more robust

    ax7.plot(iterations, robustness_metric, 'g-', alpha=0.5)
    ax7.scatter(iterations, robustness_metric, c=iterations,
                cmap='viridis', s=30, alpha=0.6)
    ax7.set_xlabel('Iteration', fontweight='bold')
    ax7.set_ylabel('Std Dev of Cl/Cd', fontweight='bold')
    ax7.set_title('Design Robustness (lower = better)', fontweight='bold')
    ax7.grid(True, alpha=0.3)
else:
    # Parameter exploration
    upper_c0_vals = [coefs[0] for coefs in history['coefficients']]
    ax7.hist(upper_c0_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax7.set_xlabel('upper_c0', fontweight='bold')
    ax7.set_ylabel('Frequency', fontweight='bold')
    ax7.set_title('Parameter Exploration', fontweight='bold')
    ax7.grid(True, alpha=0.3)

# Plot 8: Improvement over iterations
ax8 = plt.subplot(3, 3, 8)
if 'baseline_cl_cd_avg' in locals():
    improvements = [(x / baseline_cl_cd_avg - 1) * 100 for x in cl_cd_avg_values]
    improvements_best = [(x / baseline_cl_cd_avg - 1) * 100 for x in best_cl_cd_so_far]

    ax8.plot(iterations, improvements, 'b.', alpha=0.3, markersize=6)
    ax8.plot(iterations, improvements_best, 'r-', linewidth=3)
    ax8.axhline(0, color='orange', linestyle='--', linewidth=2)
    ax8.fill_between(iterations, 0, improvements_best, alpha=0.2, color='green')
    ax8.set_xlabel('Iteration', fontweight='bold')
    ax8.set_ylabel('Improvement vs Baseline (%)', fontweight='bold')
    ax8.set_title('Optimization Progress', fontweight='bold')
    ax8.grid(True, alpha=0.3)

# Plot 9: Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
╔═══════════════════════════════════════╗
║  ADVANCED BAYESIAN OPTIMIZATION       ║
╚═══════════════════════════════════════╝

Mode: {OPTIMIZATION_MODE}

"""

if OPTIMIZATION_MODE in ['multi_alpha', 'robust']:
    summary_text += f"""Baseline SG6043:
  Avg Cl/Cd: {baseline_cl_cd_avg:.2f}

Optimized Design:
  Avg Cl/Cd: {best_cl_cd_avg:.2f}
  Min Cl/Cd: {np.min(cl_cd_values_best):.2f}
  Max Cl/Cd: {np.max(cl_cd_values_best):.2f}

🚀 Improvement: {improvement:+.2f}%

Alpha Range:
  {ALPHA_RANGE[0]:.1f}° to {ALPHA_RANGE[-1]:.1f}°
  ({len(ALPHA_RANGE)} evaluation points)
"""
else:
    summary_text += f"""Baseline: {baseline_cl_cd if 'baseline_cl_cd' in locals() else 'N/A'}
Optimized: {best_cl_cd_avg:.2f}
🚀 Improvement: {improvement:+.2f}%
"""

summary_text += f"""
Optimization Stats:
  Total evals: {len(iterations)}
  Best at iter: {best_iteration}
  Time: {elapsed_time:.1f}s
  Throughput: {len(iterations)/elapsed_time:.1f} eval/s
"""

ax9.text(0.05, 0.5, summary_text, fontsize=9,
         verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(data_dir + f'bayesian_optimization_{OPTIMIZATION_MODE}.png',
            dpi=200, bbox_inches='tight')
print(f"✓ Visualization saved: bayesian_optimization_{OPTIMIZATION_MODE}.png")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================
print("\nStep 8: Saving results...")

# Save history
history_df = pd.DataFrame({
    'iteration': history['iteration'],
    'objective': history['objective'],
    'cl_cd_avg': history['cl_cd_avg'],
    'cl_avg': history['cl_avg'],
    'cd_avg': history['cd_avg']
})

for i, coef_name in enumerate(coef_names):
    history_df[coef_name] = [coefs[i] for coefs in history['coefficients']]

history_df.to_csv(data_dir + f'bayesian_optimization_{OPTIMIZATION_MODE}_history.csv',
                  index=False)
print(f"✓ History saved")

# Save best design
best_design_df = pd.DataFrame({
    'design_id': [f'BO_{OPTIMIZATION_MODE}'],
    **{coef_name: [best_coefficients[i]] for i, coef_name in enumerate(coef_names)}
})
best_design_df.to_csv(data_dir + f'bayesian_best_design_{OPTIMIZATION_MODE}.csv',
                      index=False)
print(f"✓ Best design saved")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 ADVANCED BAYESIAN OPTIMIZATION COMPLETE!")
print("="*60)

print(f"\n✨ Results Summary:")
print(f"  Mode: {OPTIMIZATION_MODE}")
print(f"  🏆 Best Avg Cl/Cd: {best_cl_cd_avg:.2f}")
print(f"  📈 Improvement: {improvement:+.2f}%")
print(f"  ⚡ Evaluations: {len(iterations)}")
print(f"  ⏱️  Time: {elapsed_time:.1f}s")

print(f"\n📁 Files Created:")
print(f"  • bayesian_optimization_{OPTIMIZATION_MODE}.png")
print(f"  • bayesian_optimization_{OPTIMIZATION_MODE}_history.csv")
print(f"  • bayesian_best_design_{OPTIMIZATION_MODE}.csv")

print("="*60)
