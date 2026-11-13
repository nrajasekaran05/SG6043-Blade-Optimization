# %%
# ============================================
# 3_RUN_XFOIL_NATIVE.PY
# Using Homebrew XFOIL (most reliable for Mac!)
# ============================================

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
import numpy as np
import pandas as pd
import os
import subprocess
import multiprocessing as mp
from functools import partial
import shutil

print("="*60)
print("XFOIL BATCH ANALYSIS (Native XFOIL)")
print("="*60)

# ============================================
# STEP 1: CHECK XFOIL INSTALLATION
# ============================================
print("\nStep 1: Checking XFOIL installation...")

try:
    result = subprocess.run(['which', 'xfoil'], 
                          capture_output=True, 
                          text=True,
                          timeout=5)
    xfoil_path = result.stdout.strip()
    
    if xfoil_path:
        print(f"✓ XFOIL found at: {xfoil_path}")
    else:
        print("❌ XFOIL not found!")
        print("\nPlease install with:")
        print("  brew install xfoil")
        exit(1)
        
except Exception as e:
    print(f"❌ Error checking for XFOIL: {e}")
    print("\nPlease install with:")
    print("  brew install xfoil")
    exit(1)
import os
print("Python current working directory:", os.getcwd())

# ============================================
# STEP 2: SETUP
# ============================================
print("\nStep 2: Setting up directories...")

data_dir = DATA_DIR
dat_dir = data_dir + 'DOE_airfoils/'
output_file = data_dir + 'xfoil_results.csv'
temp_dir = data_dir + 'xfoil_temp/'

# Create temp directory
os.makedirs(temp_dir, exist_ok=True)
print(f"✓ Temp directory: {temp_dir}")

# Parameters
Re = 500000
alphas = np.arange(-2, 13, 1)

print(f"\nAnalysis parameters:")
print(f"  Reynolds number: {Re:,}")
print(f"  Alpha range: {alphas[0]}° to {alphas[-1]}°")
print(f"  Number of alphas: {len(alphas)}")

# ============================================
# STEP 3: GET AIRFOIL FILES
# ============================================
print("\nStep 3: Loading airfoil files...")

dat_files = sorted([f for f in os.listdir(dat_dir) if f.endswith('.dat')])
print(f"✓ Found {len(dat_files)} airfoil files")

# ============================================
# STEP 4: DEFINE WORKER FUNCTION
# ============================================

def analyze_airfoil_native(dat_file, dat_dir, temp_dir, Re, alphas):
    """
    Analyze one airfoil using native XFOIL executable.
    This function runs in parallel.
    """
    
    design_id = int(dat_file.split('_')[-1].replace('.dat', ''))
    #dat_path = os.path.join(dat_dir, dat_file)
    dat_path =  dat_file

    # Create unique temp files for this process
    pid = os.getpid()
    script_file = os.path.join(temp_dir, f'cmd_{design_id}_{pid}.txt')
    #polar_file = os.path.join(temp_dir, f'polar_{design_id}_{pid}.txt')
    polar_file = f'polar_{design_id}_{pid}.txt'

    # Build XFOIL command script
    commands = f"""LOAD {dat_path}
PANE
OPER
VISC {Re}
PACC
{polar_file}


"""
    if os.path.exists(polar_file):
        print(f"✓ XFOIL produced: {os.path.basename(polar_file)}")
    else:
        print(f"⚠️ No output for {dat_file}")

    
    # Add commands for each angle of attack
    for alpha in alphas:
        commands += f"ALFA {alpha}\n"
    
    # Close and quit
    commands += """PACC

QUIT
"""
    
    # Write command script
    with open(script_file, 'w') as f:
        f.write(commands)
    
    # Run XFOIL
    results = []
    try:
        with open(script_file, 'r') as cmd_input:
            subprocess.run(['xfoil'], 
                        stdin=cmd_input,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        cwd=dat_dir,  # <-- set XFOIL’s working dir
                        timeout=120)
        
        # Parse polar output file
        if os.path.exists(polar_file):
            try:
                df = pd.read_csv(polar_file,
                               delim_whitespace=True,
                               skiprows=12,
                               names=['alpha', 'CL', 'CD', 'CDp', 'CM', 
                                     'Top_Xtr', 'Bot_Xtr'],
                               on_bad_lines='skip')
                
                for _, row in df.iterrows():
                    if not pd.isna(row['CL']) and not pd.isna(row['CD']):
                        results.append({
                            'design_id': design_id,
                            'alpha': row['alpha'],
                            'cl': row['CL'],
                            'cd': row['CD'],
                            'cm': row['CM'],
                            'cl_cd': row['CL'] / row['CD'] if row['CD'] > 1e-6 else 0
                        })
            except Exception as e:
                pass
        
        # Cleanup temp files
        for f in [script_file, polar_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
                    
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    
    return results


# ============================================
# STEP 5: RUN PARALLEL ANALYSIS
# ============================================
print("\nStep 5: Running parallel XFOIL analysis...")

num_cores = mp.cpu_count()
print(f"  Detected {num_cores} CPU cores")
print(f"  Using {num_cores} parallel workers")
print(f"  Estimated time: ~{len(dat_files) / num_cores * 1.5 / 60:.1f} hours")
print("\nStarting analysis (this will take a while)...\n")

if __name__ == '__main__':
    # Create worker pool
    with mp.Pool(processes=num_cores) as pool:
        # Partial function with fixed parameters
        analyze_func = partial(analyze_airfoil_native,
                              dat_dir=dat_dir,
                              temp_dir=temp_dir,
                              Re=Re,
                              alphas=alphas)
        
        # Run in parallel with progress tracking
        results_nested = []
        for i, result in enumerate(pool.imap(analyze_func, dat_files)):
            results_nested.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(dat_files):
                print(f"  Progress: {i + 1}/{len(dat_files)} airfoils "
                      f"({(i+1)/len(dat_files)*100:.1f}%)")
    
    # ============================================
    # STEP 6: PROCESS AND SAVE RESULTS
    # ============================================
    print("\nStep 6: Processing results...")
    
    # Flatten nested results
    results = [item for sublist in results_nested for item in sublist]
    df_results = pd.DataFrame(results)
    
    print(f"✓ Collected {len(df_results)} data points")
    
    # Merge with design parameters
    df_params = pd.read_csv(data_dir + 'DOE_parameters.csv')
    if df_results.empty:
        print("⚠️  No data collected from XFOIL! Skipping merge.")
        print("This means XFOIL didn’t produce any polar files. Check PANE and XFOIL logs.")
        exit(1)
    df_full = df_results.merge(df_params, on='design_id', how='left')
    
    # Save results
    df_full.to_csv(output_file, index=False)
    print(f"✓ Results saved: {output_file}")
    
    # ============================================
    # STEP 7: CLEANUP
    # ============================================
    print("\nStep 7: Cleaning up...")
    
    try:
        shutil.rmtree(temp_dir)
        print(f"✓ Removed temp directory")
    except:
        print(f"⚠️  Could not remove temp directory: {temp_dir}")
    
    # ============================================
    # STEP 8: SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("XFOIL ANALYSIS COMPLETE!")
    print("="*60)
    
    num_successful = df_results['design_id'].nunique()
    num_failed = len(dat_files) - num_successful
    
    print(f"\nResults:")
    print(f"  Total designs: {len(dat_files)}")
    print(f"  Successful: {num_successful}")
    print(f"  Failed: {num_failed}")
    print(f"  Total data points: {len(df_results)}")
    print(f"  Avg points per design: {len(df_results) / max(1, num_successful):.1f}")
    
    if len(df_results) > 0:
        print(f"\nPerformance summary:")
        print(f"  Cl range: [{df_results['cl'].min():.3f}, {df_results['cl'].max():.3f}]")
        print(f"  Cd range: [{df_results['cd'].min():.5f}, {df_results['cd'].max():.5f}]")
        print(f"  Max Cl/Cd: {df_results['cl_cd'].max():.1f}")
        
        # Best design
        best_idx = df_results['cl_cd'].idxmax()
        best_design = df_results.loc[best_idx]
        print(f"\nBest design so far:")
        print(f"  Design ID: {best_design['design_id']}")
        print(f"  Alpha: {best_design['alpha']:.1f}°")
        print(f"  Cl/Cd: {best_design['cl_cd']:.1f}")
    
    print(f"\nOutput file: {output_file}")
    print("="*60)
    print("\nNext step: Train neural network (4_train_model.py)")