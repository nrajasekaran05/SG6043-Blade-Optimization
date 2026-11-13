# %%
# ============================================
# 2_GENERATE_DOE.PY
# Generate Design of Experiments for SG6043
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from numpy.polynomial.chebyshev import chebvander

print("="*60)
print("SG6043 DOE GENERATION")
print("="*60)

# ============================================
# STEP 1: LOAD BASELINE COEFFICIENTS
# ============================================
print("\nStep 1: Loading baseline coefficients...")

# Option A: Load from CSV (if you saved them)
try:
    df_coeffs = pd.read_csv((DATA_DIR + 'sg6043_coefficients.csv'))
    baseline_upper = df_coeffs['upper_value'].values
    baseline_lower = df_coeffs['lower_value'].values
    print("✓ Loaded from CSV")
except:
    print("❌ CSV not found - using manual input")
    # Option B: Paste them manually (from your previous output)
    baseline_upper = np.array([
        # PASTE YOUR VALUES HERE from popt_upper_final
        # Example: [0.123, 0.456, 0.789, ...]
    ])
    baseline_lower = np.array([
        # PASTE YOUR VALUES HERE from popt_lower_final
    ])

print(f"  Upper coefficients: {len(baseline_upper)} values")
print(f"  Lower coefficients: {len(baseline_lower)} values")

# ============================================
# STEP 2: DEFINE CST FUNCTIONS
# ============================================
print("\nStep 2: Defining parametrization functions...")

def class_func(x):
    """CST class function"""
    return np.sqrt(x) * (1 - x)

def CST_chebyshev_TE(x, *a):
    """CST + Chebyshev parametrization"""
    *cheb_coeffs, te_offset = a
    xi = 2 * x - 1
    T = chebvander(xi, len(cheb_coeffs) - 1)
    shape = T @ np.array(cheb_coeffs)
    y = class_func(x) * shape
    return y + te_offset * x

print("✓ Functions defined")

# ============================================
# STEP 3: GENERATE SOBOL DOE
# ============================================
print("\nStep 3: Generating Sobol samples...")

num_coeffs = len(baseline_upper)  # Should be 9
num_samples = 300
perturbation_range = 0.3  # ±30%

print(f"  Number of designs: {num_samples}")
print(f"  Parameters per design: {2 * num_coeffs}")
print(f"  Perturbation: ±{perturbation_range*100}%")

# Sobol sampling in unit hypercube
sobol_engine = qmc.Sobol(d=2 * num_coeffs, scramble=True)
sobol_raw = sobol_engine.random(n=num_samples)

# Scale to perturbation range
lower_mult = 1.0 - perturbation_range  # 0.7
upper_mult = 1.0 + perturbation_range  # 1.3
perturbations = lower_mult + sobol_raw * (2 * perturbation_range)

# Generate designs
DOE_designs = []
for i in range(num_samples):
    coeffs_u = baseline_upper * perturbations[i, :num_coeffs]
    coeffs_l = baseline_lower * perturbations[i, num_coeffs:]
    full_params = np.concatenate([coeffs_u, coeffs_l])
    DOE_designs.append(full_params)

DOE_designs = np.array(DOE_designs)

print(f"✓ Generated {len(DOE_designs)} designs")
print(f"  Shape: {DOE_designs.shape}")

# ============================================
# STEP 4: SAVE DOE PARAMETERS
# ============================================
print("\nStep 4: Saving DOE parameters...")

output_dir = DATA_DIR

# Save parameter matrix
df_doe = pd.DataFrame(DOE_designs)
df_doe.columns = [f'upper_c{i}' for i in range(num_coeffs)] + \
                 [f'lower_c{i}' for i in range(num_coeffs)]
df_doe.insert(0, 'design_id', range(len(DOE_designs)))

param_file = output_dir + 'DOE_parameters.csv'
df_doe.to_csv(param_file, index=False)
print(f"✓ Saved: {param_file}")

# ============================================
# STEP 5: VISUALIZE DESIGN SPACE
# ============================================
print("\nStep 5: Visualizing design space...")

x_dense = np.linspace(0, 1, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Airfoil variations
num_to_plot = min(100, num_samples)
for i in range(num_to_plot):
    coeffs_u = DOE_designs[i, :num_coeffs]
    coeffs_l = DOE_designs[i, num_coeffs:]
    
    y_u = CST_chebyshev_TE(x_dense, *coeffs_u)
    y_l = CST_chebyshev_TE(x_dense, *coeffs_l)
    
    ax1.plot(x_dense, y_u, 'b-', alpha=0.1, linewidth=0.5)
    ax1.plot(x_dense, y_l, 'r-', alpha=0.1, linewidth=0.5)

# Original baseline (bold)
y_u_orig = CST_chebyshev_TE(x_dense, *baseline_upper)
y_l_orig = CST_chebyshev_TE(x_dense, *baseline_lower)
ax1.plot(x_dense, y_u_orig, 'b-', linewidth=3, label='Baseline SG6043')
ax1.plot(x_dense, y_l_orig, 'r-', linewidth=3)

ax1.set_title(f'Design Space (showing {num_to_plot}/{num_samples})', 
              fontsize=13, fontweight='bold')
ax1.set_xlabel('x/c', fontsize=11)
ax1.set_ylabel('y/c', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Plot 2: Parameter distribution (first 2 params)
ax2.scatter(DOE_designs[:, 0], DOE_designs[:, num_coeffs], 
            alpha=0.6, s=20, c=range(num_samples), cmap='viridis')
ax2.scatter(baseline_upper[0], baseline_lower[0], 
            c='red', s=200, marker='*', edgecolors='black', 
            linewidths=2, label='Baseline', zorder=5)
ax2.set_xlabel('Upper Coeff 0', fontsize=11)
ax2.set_ylabel('Lower Coeff 0', fontsize=11)
ax2.set_title('Parameter Space Coverage (2D projection)', 
              fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'DOE_visualization.png', dpi=150)
plt.show()

print(f"✓ Visualization saved: {output_dir}DOE_visualization.png")

# ============================================
# STEP 6: EXPORT AIRFOIL .DAT FILES
# ============================================
print("\nStep 6: Exporting airfoil .dat files...")

dat_dir = output_dir + 'DOE_airfoils/'
import os
os.makedirs(dat_dir, exist_ok=True)

for i, params in enumerate(DOE_designs):
    coeffs_u = params[:num_coeffs]
    coeffs_l = params[num_coeffs:]
    
    # Reconstruct
    y_u = CST_chebyshev_TE(x_dense, *coeffs_u)
    y_l = CST_chebyshev_TE(x_dense, *coeffs_l)
    
    # Format: TE lower → LE → TE upper
    lower_idx = np.argsort(-x_dense)
    upper_idx = np.argsort(x_dense)
    
    x_all = np.concatenate([x_dense[lower_idx], x_dense[upper_idx][1:]])
    y_all = np.concatenate([y_l[lower_idx], y_u[upper_idx][1:]])
    
    # Save
    dat_file = dat_dir + f'sg6043_design_{i:04d}.dat'
    with open(dat_file, 'w') as f:
        f.write(f"SG6043_Design_{i:04d}\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f}  {yi:.6f}\n")
    
    if (i + 1) % 50 == 0:
        print(f"  Exported {i + 1}/{num_samples}...")

print(f"✓ All {num_samples} .dat files saved to: {dat_dir}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("DOE GENERATION COMPLETE!")
print("="*60)
print(f"Baseline: SG6043 (Order 8)")
print(f"Designs generated: {num_samples}")
print(f"Parameters per design: {2 * num_coeffs}")
print(f"Perturbation range: ±{perturbation_range*100}%")
print(f"\nFiles created:")
print(f"  • {param_file}")
print(f"  • {dat_dir} (contains {num_samples} .dat files)")
print(f"  • {output_dir}DOE_visualization.png")
print("="*60)
print("\nNext step: Run XFOIL on all designs (3_run_xfoil.py)")
# %%
