#!/usr/bin/env python
"""Test script for compute_logP_f5"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import importlib
import temperatures
importlib.reload(temperatures)
from temperatures import compute_logP_f5, find_T_min, find_T_max, refine_Tmin
from model import model
from espinosa import Vt_vec

# Setup model
lambda_, g, vev, units = 6e-3, 0.75002, 500, 'MeV'
print(f"Model: lambda={lambda_}, g={g}, vev={vev} {units}")

dp = model(vev, lambda_, g, xstep=vev*1e-3, Tstep=vev*1e-6)
V = dp.DVtot
dV = dp.gradV

# Find critical temperatures
T_max, vevs_max, max_min_vals, false_min_tmax = find_T_max(V, dV, precision=1e-2, Phimax=2*vev, step_phi=vev*1e-2, tmax=2.5*vev)
T_min, vevs_min, false_min_tmin = find_T_min(V, dV, tmax=T_max, precision=1e-2, Phimax=2*vev, step_phi=vev*1e-2, max_min_vals=max_min_vals)
maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
T_min = refine_Tmin(T_min, V, dV, maxvev, log_10_precision=6)
print(f"T_max = {T_max:.2f} {units}, T_min = {T_min:.6e} {units}")

# Compute action at each temperature
true_vev, S3overT, V_min_value, phi0_min, V_exit, false_vev = {}, {}, {}, {}, {}, {}

def action_over_T(T):
    instance = Vt_vec(T, V, dV, step_phi=1e-3, precision=1e-3, vev0=maxvev, ratio_vev_step0=50)
    if instance.barrier:
        true_vev[T] = instance.true_min
        false_vev[T] = instance.phi_original_false_vev
        S3overT[T] = instance.action_over_T
        V_min_value[T] = instance.min_V
    return None

n_points = 100
temps = np.linspace(T_min if T_min else 0, T_max, n_points)
print(f"Computing action at {n_points} temperatures...")
for t in temps:
    action_over_T(t)
print(f"Computed {len(S3overT)} points with barrier")

# Run compute_logP_f5
print("\nRunning compute_logP_f5...")
try:
    result = compute_logP_f5(dp, V_min_value, S3overT, true_vev, false_vev, 
                             v_w=1, units=units, n_iterations=3, return_all=True)
    logP_f5, Temps5, R5, Gamma5, H5, V_ext5, I5, rho_f5, rho_t5, w_f5, w_t5, L5 = result
    
    print(f"\nResults:")
    print(f"  Temperature range: {Temps5.min():.2f} - {Temps5.max():.2f} {units}")
    print(f"  logP_f5 range: {np.nanmin(logP_f5):.6e} to {np.nanmax(logP_f5):.6e}")
    print(f"  V_ext range: {np.nanmin(V_ext5):.6e} to {np.nanmax(V_ext5):.6e}")
    print(f"  R range: {np.nanmin(R5):.6e} to {np.nanmax(R5):.6e}")
    print(f"  H range: {np.nanmin(H5):.6e} to {np.nanmax(H5):.6e}")
    
    print(f"\nNaN/Inf counts:")
    print(f"  logP_f5: {np.sum(~np.isfinite(logP_f5))} / {len(logP_f5)}")
    print(f"  R: {np.sum(~np.isfinite(R5))} / {len(R5)}")
    
    print(f"\nSample values:")
    print(f"{'T':>10} {'logP_f5':>15} {'V_ext':>15} {'R':>12}")
    print("-"*55)
    indices = np.linspace(0, len(Temps5)-1, 10).astype(int)
    for i in indices:
        print(f"{Temps5[i]:>10.2f} {logP_f5[i]:>15.6e} {V_ext5[i]:>15.6e} {R5[i]:>12.4e}")
        
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
