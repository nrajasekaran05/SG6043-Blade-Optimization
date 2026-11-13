"""Re-evaluate exported airfoil geometries directly with NeuralFoil.

Compares what the surrogate model predicted against an independent
NeuralFoil evaluation of the exported coordinates (Re = 500,000, Mach 0).
"""
import os

import aerosandbox as asb
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts') + os.sep
RE = 500_000
ALPHAS = (2.0, 6.0)

DESIGNS = {
    'Baseline (DOE design 0)': 'DOE_airfoils/sg6043_design_0000.dat',
    'Best DOE at 6 deg (design 221)': 'DOE_airfoils/sg6043_design_0221.dat',
    'Best DOE at 2 deg (design 34)': 'sg6043_optimized_BEST.dat',
    'Bayesian optimized (200 pts)': 'sg6043_bayesian_optimized.dat',
    'Bayesian optimized (399 pts)': 'optimized_airfoil_bayesian.dat',
}


def load_dat(path):
    coords = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    return np.array(coords)


def scalar(x):
    return float(np.asarray(x).ravel()[0])


print(f"{'Design':34s}" + ''.join(f"  Cl/Cd @ {a:g} deg" for a in ALPHAS))
for name, rel in DESIGNS.items():
    airfoil = asb.Airfoil(name=name, coordinates=load_dat(DATA_DIR + rel))
    row = f'{name:34s}'
    for alpha in ALPHAS:
        aero = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=RE, mach=0.0)
        row += f"  {scalar(aero['CL']) / scalar(aero['CD']):15.1f}"
    print(row)
