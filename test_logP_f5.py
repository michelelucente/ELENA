#!/usr/bin/env python3
"""
Test script to compare compute_logP_f5 with compute_logP_f
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add the src directory to the Python path
script_dir = Path(os.getcwd()).resolve()
sys.path.append(str(Path(script_dir / 'src').resolve()))

# Import modules
from model import model
from temperatures import find_T_min, find_T_max, refine_Tmin, compute_logP_f, compute_logP_f5
from espinosa import Vt_vec
from utils import interpolation_narrow
from temperatures import N_bubblesH

print("=" * 80)
print("TESTING compute_logP_f5 vs compute_logP_f")
print("=" * 80)

# Set up parameters as requested
lambda_ = 6e-3
g = 0.75002
vev = 500
units = "MeV"

print(f"\nParameters:")
print(f"  lambda = {lambda_}")
print(f"  g = {g}")
print(f"  vev = {vev} {units}")

# Create the model
dp = model(vev, lambda_, g, xstep=vev * 1e-3, Tstep=vev * 1e-6)
V = dp.DVtot
dV = dp.gradV

print("\nModel created successfully!")

# Find critical temperatures
print("\nFinding critical temperatures...")
T_max, vevs_max, max_min_vals, false_min_tmax = find_T_max(V, dV, precision=1e-2, Phimax=2*vev, step_phi=vev * 1e-2, tmax=2.5 * vev)
T_min, vevs_min, false_min_tmin = find_T_min(V, dV, tmax=T_max, precision=1e-2, Phimax=2*vev, step_phi=vev * 1e-2, max_min_vals=max_min_vals)

if T_max is not None and T_min is not None:
    maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
elif T_max is not None:
    maxvev = np.max(vevs_max)
elif T_min is not None:
    maxvev = np.max(vevs_min)
else:
    maxvev = None

T_min = refine_Tmin(T_min, V, dV, maxvev, log_10_precision=6) if T_min is not None else None

print(f"\nCritical temperatures:")
print(f"  T_max = {T_max:.6e} {units}")
print(f"  T_min = {T_min:.6e} {units}")

# Initialize dictionaries for storing results
true_vev = {}
S3overT = {}
V_min_value = {}
phi0_min = {}
V_exit = {}
false_vev = {}

def action_over_T(T, c_step_phi=1e-3, precision=1e-3):
    instance = Vt_vec(T, V, dV, step_phi=c_step_phi, precision=precision, vev0=maxvev, ratio_vev_step0=50)
    if instance.barrier:
        true_vev[T] = instance.true_min
        false_vev[T] = instance.phi_original_false_vev
        S3overT[T] = instance.action_over_T
        V_min_value[T] = instance.min_V
        phi0_min[T] = instance.phi0_min
        V_exit[T] = instance.V_exit
        return instance.action_over_T
    else:
        return None

# Compute tunneling action over temperature range
n_points = 100  # Use fewer points for faster testing
temperatures = np.linspace(T_min, T_max, n_points)
action_vec = np.vectorize(action_over_T)

print(f"\nComputing tunneling action for {n_points} temperature points...")
start_time = time.time()
action_vec(temperatures)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")

temperatures = np.array([T for T in temperatures if T in S3overT])
print(f"Valid temperature points: {len(temperatures)}")

# Debug S3overT
min_S3 = min(S3overT.values())
max_S3 = max(S3overT.values())
print(f"S3/T range: {min_S3} to {max_S3}")
Gamma_est = (14.0)**4 * np.exp(-min_S3)
print(f"Estimated max Gamma (at T=14, S3={min_S3}): {Gamma_est}")

# Run compute_logP_f
print("\n" + "=" * 80)
print("Running compute_logP_f...")
start_time = time.time()
logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(dp, V_min_value, S3overT, v_w=1, units=units, cum_method='None')
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.4f} seconds")

# Run compute_logP_f5
print("\nRunning compute_logP_f5...")
start_time = time.time()
result5 = compute_logP_f5(dp, V_min_value, S3overT, true_vev, false_vev, v_w=1, units=units, return_all=True)
logP_f5, Temps5, R5, Gamma5, H5, V_ext5, I5, rho_f5, rho_t5, rho_plus_p_f5, rho_plus_p_t5, delta_rho5 = result5
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.4f} seconds")

# Basic statistics
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\ncompute_logP_f:")
print(f"  Temperature range: {Temps.min():.4f} to {Temps.max():.4f} {units}")
print(f"  logP_f range: {np.nanmin(logP_f):.6e} to {np.nanmax(logP_f):.6e}")
print(f"  NaN count: {np.isnan(logP_f).sum()}")

print(f"\ncompute_logP_f5:")
print(f"  Temperature range: {Temps5.min():.4f} to {Temps5.max():.4f} {units}")
print(f"  logP_f5 range: {np.nanmin(logP_f5):.6e} to {np.nanmax(logP_f5):.6e}")
print(f"  NaN count: {np.isnan(logP_f5).sum()}")

# Compare at matched temperatures
print("\n" + "=" * 80)
print("COMPARISON AT MATCHED TEMPERATURES")
print("=" * 80)

# Find common temperature indices
print(f"\n{'T':<15} {'logP_f':>15} {'logP_f5':>15} {'Diff':>15} {'Rel Diff (%)':>15}")
print("-" * 75)

# Sample some temperatures across the range
sample_indices = np.linspace(0, len(Temps) - 1, 10, dtype=int)
for idx in sample_indices:
    T_test = Temps[idx]
    # Find matching index in Temps5
    idx5 = np.argmin(np.abs(Temps5 - T_test))
    
    val_f = logP_f[idx]
    val_f5 = logP_f5[idx5]
    diff = val_f - val_f5
    
    if val_f != 0 and not np.isnan(val_f):
        rel_diff = 100 * np.abs(diff / val_f)
    else:
        rel_diff = np.nan
    
    print(f"{T_test:<15.6f} {val_f:>15.6e} {val_f5:>15.6e} {diff:>15.6e} {rel_diff:>14.2f}%")

# Calculate for compute_logP_f
mask_Pf = ~np.isnan(logP_f)
try:
    T_perc = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.71))
    T_completion = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.01))
except:
    T_perc = np.nan
    T_completion = np.nan

nH = N_bubblesH(Temps, Gamma, logP_f, H, ratio_V)
mask_nH = ~np.isnan(nH)
try:
    T_nuc = interpolation_narrow(np.log(nH[mask_nH]), Temps[mask_nH], 0)
except:
    T_nuc = np.nan

# Calculate for compute_logP_f5
mask_Pf5 = ~np.isnan(logP_f5)
try:
    T_perc5 = interpolation_narrow(logP_f5[mask_Pf5], Temps5[mask_Pf5], np.log(0.71))
    T_completion5 = interpolation_narrow(logP_f5[mask_Pf5], Temps5[mask_Pf5], np.log(0.01))
except:
    T_perc5 = np.nan
    T_completion5 = np.nan

nH5 = N_bubblesH(Temps5, Gamma5, logP_f5, H5, R5)
mask_nH5 = ~np.isnan(nH5)
try:
    T_nuc5 = interpolation_narrow(np.log(nH5[mask_nH5]), Temps5[mask_nH5], 0)
except:
    T_nuc5 = np.nan

print("\n" + "=" * 80)
print("MILESTONE TEMPERATURES COMPARISON")
print("=" * 80)
print(f"\n{'Milestone':<20} {'compute_logP_f':>18} {'compute_logP_f5':>18} {'Difference':>15}")
print("-" * 75)
print(f"{'T_nucleation':<20} {T_nuc:>18.6f} {T_nuc5:>18.6f} {T_nuc - T_nuc5:>15.6f}")
print(f"{'T_percolation':<20} {T_perc:>18.6f} {T_perc5:>18.6f} {T_perc - T_perc5:>15.6f}")
print(f"{'T_completion':<20} {T_completion:>18.6f} {T_completion5:>18.6f} {T_completion - T_completion5:>15.6f}")

# Additional diagnostic information
print("\n" + "=" * 80)
print("ADDITIONAL DIAGNOSTICS FOR compute_logP_f5")
print("=" * 80)
print(f"\nV_ext (expected bubble volume) range: {np.nanmin(V_ext5):.6e} to {np.nanmax(V_ext5):.6e}")
print(f"I (dV_ext/dT) range: {np.nanmin(I5):.6e} to {np.nanmax(I5):.6e}")
print(f"R (thermodynamic ratio) range: {np.nanmin(R5):.6e} to {np.nanmax(R5):.6e}")
print(f"Hubble H range: {np.nanmin(H5):.6e} to {np.nanmax(H5):.6e}")
print(f"Gamma (decay rate) range: {np.nanmin(Gamma5):.6e} to {np.nanmax(Gamma5):.6e}")
print(f"\nDelta rho (latent heat related) range: {np.nanmin(delta_rho5):.6e} to {np.nanmax(delta_rho5):.6e}")
print(f"rho_f range: {np.nanmin(rho_f5):.6e} to {np.nanmax(rho_f5):.6e}")
print(f"rho_t range: {np.nanmin(rho_t5):.6e} to {np.nanmax(rho_t5):.6e}")

# Compare ratio_V (from logP_f) with R5 (from logP_f5)
print("\n" + "=" * 80)
print("THERMODYNAMIC RATIO COMPARISON (ratio_V vs R)")
print("=" * 80)
print(f"\n{'T':<15} {'ratio_V':>15} {'R5':>15} {'Diff':>15}")
print("-" * 60)

for idx in sample_indices[:5]:
    T_test = Temps[idx]
    idx5 = np.argmin(np.abs(Temps5 - T_test))
    print(f"{T_test:<15.6f} {ratio_V[idx]:>15.6e} {R5[idx5]:>15.6e} {ratio_V[idx] - R5[idx5]:>15.6e}")

print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 80)
