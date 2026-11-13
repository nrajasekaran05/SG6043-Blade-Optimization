# %%
# ============================================
# 3_RUN_AEROSANDBOX.PY - WORKING VERSION
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from functools import partial

print("="*60)
print("AERODYNAMIC ANALYSIS (AeroSandbox)")
print("="*60)

# ============================================
# STEP 1: CHECK AEROSANDBOX
# ============================================
print("\nStep 1: Loading AeroSandbox...")

try:
    import aerosandbox as asb
    print(f"✓ AeroSandbox v{asb.__version__} loaded")
except ImportError:
    print("❌ AeroSandbox not installed!")
    print("Install with: pip install aerosandbox")
    exit(1)

# ============================================
# STEP 2: SETUP
# ============================================
print("\nStep 2: Setting up analysis...")

data_dir = DATA_DIR
dat_dir = data_dir + 'DOE_airfoils/'
output_file = data_dir + 'aerosandbox_results.csv'

# Parameters
Re = 500000
alphas = np.arange(-2, 13, 1)

print(f"  Reynolds number: {Re:,}")
print(f"  Alpha range: {alphas[0]}° to {alphas[-1]}° ({len(alphas)} points)")

# Get files
dat_files = sorted([f for f in os.listdir(dat_dir) if f.endswith('.dat')])
print(f"✓ Found {len(dat_files)} airfoil files")

# ============================================
# STEP 3: WORKER FUNCTION
# ============================================

def load_airfoil_from_dat(dat_path):
    """Load airfoil coordinates from .dat file"""
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    
    coords = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
            except:
                continue
    
    return np.array(coords)

def analyze_airfoil_asb(dat_file, dat_dir, Re, alphas):
    """Analyze one airfoil using AeroSandbox"""
    
    design_id = int(dat_file.split('_')[-1].replace('.dat', ''))
    dat_path = os.path.join(dat_dir, dat_file)
    
    results = []
    
    try:
        import aerosandbox as asb
        
        # Load coordinates
        coords = load_airfoil_from_dat(dat_path)
        
        if len(coords) < 10:
            return []
        
        # Create airfoil object
        airfoil = asb.Airfoil(
            name=f"design_{design_id}",
            coordinates=coords
        )
        
        # Analyze at each alpha
        for alpha in alphas:
            try:
                # Run neuralfoil analysis
                aero = airfoil.get_aero_from_neuralfoil(
                    alpha=float(alpha),
                    Re=float(Re),
                    mach=0.0
                )
                
                # Extract values (handle potential dict or object)
                if isinstance(aero, dict):
                    cl = float(aero['CL'])
                    cd = float(aero['CD'])
                    cm = float(aero['CM'])
                else:
                    cl = float(aero.CL)
                    cd = float(aero.CD)
                    cm = float(aero.CM)
                
                # Check for valid values
                if np.isnan(cl) or np.isnan(cd) or np.isinf(cl) or np.isinf(cd):
                    continue
                
                results.append({
                    'design_id': design_id,
                    'alpha': float(alpha),
                    'cl': cl,
                    'cd': cd,
                    'cm': cm,
                    'cl_cd': cl / cd if abs(cd) > 1e-6 else 0.0
                })
                
            except Exception as e:
                # Skip this alpha if it fails
                continue
                
    except Exception as e:
        # Skip this airfoil if loading fails
        pass
    
    return results

# ============================================
# STEP 4: RUN PARALLEL ANALYSIS
# ============================================
print("\nStep 4: Running parallel analysis...")

num_cores = mp.cpu_count()
print(f"  Using {num_cores} CPU cores")
print(f"  Estimated time: ~{len(dat_files) / num_cores * 0.5 / 60:.1f} hours")
print("\nStarting analysis...\n")

if __name__ == '__main__':
    with mp.Pool(processes=num_cores) as pool:
        analyze_func = partial(analyze_airfoil_asb,
                              dat_dir=dat_dir,
                              Re=Re,
                              alphas=alphas)
        
        results_nested = []
        completed = 0
        
        for result in pool.imap(analyze_func, dat_files):
            results_nested.append(result)
            completed += 1
            
            if completed % 20 == 0 or completed == len(dat_files):
                print(f"  Progress: {completed}/{len(dat_files)} "
                      f"({completed/len(dat_files)*100:.1f}%)")
    
    # ============================================
    # STEP 5: PROCESS RESULTS
    # ============================================
    print("\nStep 5: Processing results...")
    
    # Flatten nested results
    results = [item for sublist in results_nested for item in sublist]
    
    print(f"  Collected {len(results)} data points")
    
    if len(results) == 0:
        print("\n❌ No results collected!")
        print("Something went wrong with the analysis.")
        exit(1)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Load parameters and merge
    try:
        df_params = pd.read_csv(data_dir + 'DOE_parameters.csv')
        df_full = df_results.merge(df_params, on='design_id', how='left')
        print("✓ Merged with design parameters")
    except Exception as e:
        print(f"⚠️  Could not merge parameters: {e}")
        df_full = df_results
    
    # Save results
    df_full.to_csv(output_file, index=False)
    print(f"✓ Results saved: {output_file}")
    
    # ============================================
    # STEP 6: SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    num_successful = df_results['design_id'].nunique()
    num_failed = len(dat_files) - num_successful
    
    print(f"\nResults:")
    print(f"  Total designs: {len(dat_files)}")
    print(f"  Successful: {num_successful}")
    print(f"  Failed: {num_failed}")
    print(f"  Total data points: {len(df_results)}")
    print(f"  Avg points per design: {len(df_results) / max(1, num_successful):.1f}")
    
    print(f"\nPerformance summary:")
    print(f"  Cl range: [{df_results['cl'].min():.3f}, {df_results['cl'].max():.3f}]")
    print(f"  Cd range: [{df_results['cd'].min():.5f}, {df_results['cd'].max():.5f}]")
    print(f"  Max Cl/Cd: {df_results['cl_cd'].max():.1f}")
    
    # Best design
    best_idx = df_results['cl_cd'].idxmax()
    best = df_results.loc[best_idx]
    print(f"\nBest design:")
    print(f"  Design ID: {int(best['design_id'])}")
    print(f"  Alpha: {best['alpha']:.1f}°")
    print(f"  Cl: {best['cl']:.3f}")
    print(f"  Cd: {best['cd']:.5f}")
    print(f"  Cl/Cd: {best['cl_cd']:.1f}")
    
    print(f"\nOutput: {output_file}")
    print("="*60)
    print("\n✅ Next step: Train neural network (4_train_model.py)")