"""
Generate .dat file for optimized airfoil
Converts Chebyshev coefficients to x,y coordinates
"""
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("GENERATING OPTIMIZED AIRFOIL .DAT FILE")
print("="*70)

data_dir = DATA_DIR

# Load optimized airfoil coefficients
df_optimized = pd.read_csv(data_dir + 'bayesian_best_design.csv')
print("✓ Loaded optimized airfoil coefficients")

# Extract coefficients
upper_coeffs = [df_optimized.iloc[0][f'upper_c{i}'] for i in range(9)]
lower_coeffs = [df_optimized.iloc[0][f'lower_c{i}'] for i in range(9)]

print(f"\nUpper surface coefficients:")
for i, c in enumerate(upper_coeffs):
    print(f"  c{i}: {c:12.8f}")

print(f"\nLower surface coefficients:")
for i, c in enumerate(lower_coeffs):
    print(f"  c{i}: {c:12.8f}")

# Generate airfoil coordinates using Chebyshev polynomials
def chebyshev_polynomials(x, n_terms=9):
    """Generate Chebyshev polynomials T0 through T(n-1)"""
    T = np.zeros((len(x), n_terms))
    T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = x
    for i in range(2, n_terms):
        T[:, i] = 2 * x * T[:, i-1] - T[:, i-2]
    return T

# Generate x coordinates (chord-wise, from leading edge to trailing edge)
n_points = 200
x_upper = np.linspace(0, 1, n_points)  # Upper surface: LE to TE
x_lower = np.linspace(0, 1, n_points)  # Lower surface: LE to TE

# Transform x to Chebyshev domain [-1, 1]
x_cheb_upper = 2 * x_upper - 1
x_cheb_lower = 2 * x_lower - 1

# Compute Chebyshev basis functions
T_upper = chebyshev_polynomials(x_cheb_upper, 9)
T_lower = chebyshev_polynomials(x_cheb_lower, 9)

# Compute y coordinates as weighted sum of Chebyshev polynomials
y_upper = np.dot(T_upper, upper_coeffs)
y_lower = np.dot(T_lower, lower_coeffs)

print(f"\n✓ Generated {n_points} points per surface")
print(f"  Upper surface: y range [{y_upper.min():.4f}, {y_upper.max():.4f}]")
print(f"  Lower surface: y range [{y_lower.min():.4f}, {y_lower.max():.4f}]")

# Combine into single airfoil (start at TE upper, go to LE, then to TE lower)
# Standard .dat format: start at trailing edge, go around airfoil, back to TE
x_airfoil = np.concatenate([x_upper[::-1], x_lower[1:]])  # Reverse upper, skip duplicate LE
y_airfoil = np.concatenate([y_upper[::-1], y_lower[1:]])

print(f"\n✓ Combined airfoil: {len(x_airfoil)} total points")

# Save as .dat file (standard XFOIL format)
dat_filename = data_dir + 'optimized_airfoil_bayesian.dat'

with open(dat_filename, 'w') as f:
    f.write("BAYESIAN_OPTIMIZED_AIRFOIL\n")
    for x, y in zip(x_airfoil, y_airfoil):
        f.write(f"  {x:12.8f}  {y:12.8f}\n")

print(f"\n✓ Saved: {dat_filename}")

# Also save separate upper and lower for reference
np.savetxt(data_dir + 'optimized_airfoil_upper.csv',
           np.column_stack([x_upper, y_upper]),
           delimiter=',', header='x,y_upper', comments='')

np.savetxt(data_dir + 'optimized_airfoil_lower.csv',
           np.column_stack([x_lower, y_lower]),
           delimiter=',', header='x,y_lower', comments='')

print("✓ Saved: optimized_airfoil_upper.csv")
print("✓ Saved: optimized_airfoil_lower.csv")

# Load baseline for comparison
df_baseline = pd.read_csv(data_dir + 'DOE_parameters.csv')
baseline_coeffs = df_baseline[df_baseline['design_id'] == 0].iloc[0]
baseline_upper = [baseline_coeffs[f'upper_c{i}'] for i in range(9)]
baseline_lower = [baseline_coeffs[f'lower_c{i}'] for i in range(9)]

y_upper_baseline = np.dot(T_upper, baseline_upper)
y_lower_baseline = np.dot(T_lower, baseline_lower)

# Create visualization
print("\n" + "="*70)
print("CREATING AIRFOIL VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Airfoil comparison
ax1 = axes[0, 0]
ax1.plot(x_upper, y_upper_baseline, 'b--', linewidth=2, label='Baseline SG6043 Upper')
ax1.plot(x_lower, y_lower_baseline, 'b--', linewidth=2, label='Baseline SG6043 Lower')
ax1.plot(x_upper, y_upper, 'r-', linewidth=2.5, label='Optimized Upper')
ax1.plot(x_lower, y_lower, 'r-', linewidth=2.5, label='Optimized Lower')
ax1.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.set_xlabel('x/c', fontweight='bold', fontsize=12)
ax1.set_ylabel('y/c', fontweight='bold', fontsize=12)
ax1.set_title('Airfoil Shape Comparison', fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Thickness distribution
ax2 = axes[0, 1]
thickness_baseline = y_upper_baseline - y_lower_baseline
thickness_optimized = y_upper - y_lower
ax2.plot(x_upper, thickness_baseline * 100, 'b--', linewidth=2, label='Baseline')
ax2.plot(x_upper, thickness_optimized * 100, 'r-', linewidth=2.5, label='Optimized')
ax2.fill_between(x_upper, 0, thickness_optimized * 100, alpha=0.2, color='red')
ax2.set_xlabel('x/c', fontweight='bold', fontsize=12)
ax2.set_ylabel('Thickness (%)', fontweight='bold', fontsize=12)
ax2.set_title('Thickness Distribution', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Camber line
ax3 = axes[1, 0]
camber_baseline = (y_upper_baseline + y_lower_baseline) / 2
camber_optimized = (y_upper + y_lower) / 2
ax3.plot(x_upper, camber_baseline * 100, 'b--', linewidth=2, label='Baseline')
ax3.plot(x_upper, camber_optimized * 100, 'r-', linewidth=2.5, label='Optimized')
ax3.fill_between(x_upper, 0, camber_optimized * 100, alpha=0.2, color='red')
ax3.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax3.set_xlabel('x/c', fontweight='bold', fontsize=12)
ax3.set_ylabel('Camber (%)', fontweight='bold', fontsize=12)
ax3.set_title('Mean Camber Line', fontweight='bold', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Difference plot
ax4 = axes[1, 1]
diff_upper = (y_upper - y_upper_baseline) * 100
diff_lower = (y_lower - y_lower_baseline) * 100
ax4.plot(x_upper, diff_upper, 'r-', linewidth=2, label='Upper surface change')
ax4.plot(x_lower, diff_lower, 'b-', linewidth=2, label='Lower surface change')
ax4.fill_between(x_upper, 0, diff_upper, alpha=0.2, color='red')
ax4.fill_between(x_lower, 0, diff_lower, alpha=0.2, color='blue')
ax4.axhline(0, color='k', linestyle='--', linewidth=1)
ax4.set_xlabel('x/c', fontweight='bold', fontsize=12)
ax4.set_ylabel('Difference from Baseline (%c)', fontweight='bold', fontsize=12)
ax4.set_title('Optimization Changes', fontweight='bold', fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('Bayesian Optimized Airfoil vs. Baseline SG6043',
             fontsize=15, fontweight='bold')
plt.tight_layout()

plot_filename = data_dir + 'optimized_airfoil_comparison.png'
plt.savefig(plot_filename, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: optimized_airfoil_comparison.png")

# Print key geometry metrics
max_thickness_baseline = np.max(thickness_baseline) * 100
max_thickness_optimized = np.max(thickness_optimized) * 100
max_camber_baseline = np.max(camber_baseline) * 100
max_camber_optimized = np.max(camber_optimized) * 100

print("\n" + "="*70)
print("AIRFOIL GEOMETRY METRICS")
print("="*70)

print(f"\nBaseline SG6043:")
print(f"  Max thickness:  {max_thickness_baseline:.2f}% at x/c = {x_upper[np.argmax(thickness_baseline)]:.3f}")
print(f"  Max camber:     {max_camber_baseline:.2f}% at x/c = {x_upper[np.argmax(camber_baseline)]:.3f}")

print(f"\nOptimized Airfoil:")
print(f"  Max thickness:  {max_thickness_optimized:.2f}% at x/c = {x_upper[np.argmax(thickness_optimized)]:.3f}")
print(f"  Max camber:     {max_camber_optimized:.2f}% at x/c = {x_upper[np.argmax(camber_optimized)]:.3f}")

print(f"\nChanges:")
print(f"  Thickness: {max_thickness_optimized - max_thickness_baseline:+.2f}%")
print(f"  Camber:    {max_camber_optimized - max_camber_baseline:+.2f}%")

# Show first and last few lines of .dat file
print("\n" + "="*70)
print("PREVIEW OF .DAT FILE")
print("="*70)
print(f"\nFile: {dat_filename}")
print("\nFirst 10 lines:")
with open(dat_filename, 'r') as f:
    lines = f.readlines()
    for line in lines[:10]:
        print("  " + line.rstrip())

print(f"\n... ({len(lines)} total lines) ...\n")
print("Last 5 lines:")
for line in lines[-5:]:
    print("  " + line.rstrip())

print("\n" + "="*70)
print("✅ AIRFOIL .DAT FILE GENERATION COMPLETE!")
print("="*70)

print(f"\n📁 Files created:")
print(f"  • optimized_airfoil_bayesian.dat (XFOIL format)")
print(f"  • optimized_airfoil_upper.csv")
print(f"  • optimized_airfoil_lower.csv")
print(f"  • optimized_airfoil_comparison.png")

print(f"\n💡 You can use the .dat file in XFOIL, ANSYS, or any CFD tool!")
print("="*70)
