import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
try:
    from scipy.integrate import cumulative_simpson
except:
    pass

from utils import convert_units, s_SM
from model import model
from espinosa import Vt_vec
from dof_interpolation import g_rho

M_pl = 2.4353234600842885e+18

def find_extrema(V, dV, T, Phimax = 150, step = 1):
        phi = np.arange(step, Phimax + step, step)
        phi = phi.reshape(-1,1)
        
        v = V(phi, T)
        argmaxV = np.argmax(v)
        while (argmaxV + 1) != len(v):
            if np.isnan(v[argmaxV]):
                return [], [], []
            else:
                Phimax *= 2
                phi = np.arange(step, Phimax, step)
                phi = phi.reshape(-1,1)
                
                v = V(phi, T)
                argmaxV = np.argmax(v)

        dv = dV(phi, T)
        signs = np.sign(dv).reshape(-1)
        signs = np.round(signs).astype(int)
        sign_changes = np.diff(signs) # negative: max, positive: min
        change_indices = np.nonzero(abs(sign_changes) == 2)[0]

        filtered_change_indices = []
        previous = 0
        for idx in np.flip(change_indices):
            if idx != previous - 1:
                filtered_change_indices.append(idx)
            previous = idx
        filtered_change_indices = np.array(filtered_change_indices)
        filtered_change_indices = np.flip(filtered_change_indices)

        loc = []
        val = []
        kind = []
        for idx in filtered_change_indices:
            this_one = "max" if sign_changes[idx] < 0 else "min"
            if this_one == "max":
                extreme_location = np.interp(0, np.flip(dv[idx:idx+2]).flatten(), np.flip(phi[idx:idx+2]).flatten())
            else:
                extreme_location = np.interp(0, dv[idx:idx+2].flatten(), phi[idx:idx+2].flatten())
            potential_value = np.interp(extreme_location, phi[idx:idx+2].flatten(), v[idx:idx+2].flatten())

            kind.append(this_one)
            loc.append(extreme_location)
            val.append(potential_value)

        zeroes_indices = np.where(dv==0)[0]

        real_zeroes = []
        for idx in zeroes_indices:
            if dv[idx-1] * dv[idx+1] < 0:
                real_zeroes.append(idx)

        for idx in real_zeroes:
            kind.append("max" if dv[idx-1] > dv[idx+1] else "min")
            loc.append(phi[idx][0])
            val.append(v[idx])


        # Deals with cases where consecutive zeroes are present
        diff = np.diff(zeroes_indices)

        starts = np.where(diff > 1)[0] + 1
        ends = np.where(diff > 1)[0]

        starts = np.insert(starts, 0, 0)
        ends = np.append(ends, len(zeroes_indices) - 1)

        if len(zeroes_indices) > 1:
            sequences = [(zeroes_indices[start], zeroes_indices[end]) for start, end in zip(starts, ends)]

            filtered_sequences = [seq for seq in sequences if seq[0] != seq[1]]

            for pair in filtered_sequences:
                if dv[pair[0]-1] * dv[pair[1]+1] < 0:
                    this_one = "max" if dv[pair[0]-1] > dv[pair[1]+1] else "min"
                    if this_one == "max":
                        extreme_location = np.interp(0, dv[[pair[1]+1, pair[0]-1]].flatten(), phi[[pair[1]+1,pair[0]-1]].flatten())
                    else:
                        extreme_location = np.interp(0, dv[[pair[0]-1,pair[1]+1]].flatten(), phi[[pair[0]-1,pair[1]+1]].flatten())
                    potential_value = np.interp(extreme_location, phi[[pair[0]-1,pair[1]+1]].flatten(), v[[pair[0]-1,pair[1]+1]].flatten())
                    kind.append(this_one)
                    loc.append(extreme_location)
                    val.append(potential_value)

        kind, loc, val = np.array(kind), np.array(loc), np.array(val)

        kind = kind[np.argsort(loc)].tolist()
        val = val[np.argsort(loc)].tolist()
        loc = loc[np.argsort(loc)].tolist()

        if len(kind) > 2:
            global_min_idx = np.argmin(val)
            kind = kind[:global_min_idx+1]
            loc = loc[:global_min_idx+1]
            val = val[:global_min_idx+1]

        return kind, loc, val


def find_T_min(V, dV, precision = 1e-2, Phimax = 150, step_phi = 0.1, tmax = 250, tmin = 0, max_min_vals = None):
    if tmax is None:
        return None, None, None
        
    t0 = tmin
    dt = tmax/100
    tchange = None
    high_T_vevs = []
    counter2 = 0
    while tchange is None:
        counter2 += 1
        temps = np.arange(t0, tmax + dt, dt)
        temps[temps > tmax] = tmax
        temps[temps < dt/10] = 0
        temps = np.unique(temps)
        for T in temps:
            res = find_extrema(V, dV, T, Phimax, step_phi / counter2)
            num_extrema = len(res[0])
            idx_max = num_extrema - 2
            if max_min_vals is not None:
                if num_extrema >= 2:
                    closest_distance = float('inf')  # Initialize to infinity
                    for i in range(num_extrema - 1):
                        # Check for the "max" followed by "min"
                        if res[0][i] == 'max' and res[0][i + 1] == 'min':
                            loc_max = res[1][i]
                            loc_min = res[1][i + 1]
                            # Calculate the distance from the old point
                            distance = ((loc_max - max_min_vals[0])**2 + (loc_min - max_min_vals[1])**2) ** 0.5
                            
                            # Check if this distance is smaller than the closest found so far
                            if distance < closest_distance:
                                closest_distance = distance
                                idx_max = i


            if res[0][idx_max:idx_max+2] == ['max', 'min']:
                if len(res[0]) == 2:
                    false_min_loc, false_min_val = 0, 0
                if len(res[0]) > 2:
                    extrema_kind, extrema_loc, extrema_val = np.array(res[0][:idx_max]), np.array(res[1][:idx_max]), np.array(res[2][:idx_max])
                    try:
                        false_min_val = np.min(extrema_val[extrema_kind == 'min'])
                        false_min_loc = extrema_loc[extrema_val == false_min_val]
                        if false_min_val > 0:
                            false_min_loc, false_min_val = 0, 0
                    except:
                        false_min_loc, false_min_val = 0, 0

            if num_extrema >=2 and res[0][idx_max:idx_max+2] == ['max', 'min'] and res[2][idx_max:idx_max+2][0] > false_min_val and res[2][idx_max:idx_max+2][1] < false_min_val:
                high_T_vevs.append(res[1][idx_max+1])
                if T == 0:
                    return 0, np.array(high_T_vevs), [false_min_loc, false_min_val]
                
                if np.abs( (res[2][idx_max] - false_min_val) / (res[2][idx_max+1] - false_min_val)) <= precision:
                    return T, np.array(high_T_vevs), [false_min_loc, false_min_val]
                else:
                    tchange = T
                    t0 = T - dt
                    dt = dt/10
                    break
            else:
                if T == temps[-1]:
                    step_phi = step_phi / 10
                    break
                else:
                    continue
    return tchange, np.array(high_T_vevs), [false_min_loc, false_min_val]


def find_T_max(V, dV, precision = 1e-2, Phimax = 150, step_phi = 1, tmax = 250, tmin = 0):
    t0 = tmin
    dt = tmax/100
    tchange = None
    res = find_extrema(V, dV, tmax, Phimax, step_phi)
    counter = 0
    while res[0] != [] and counter < 10:
        counter += 1
        tmax *= 2
        res = find_extrema(V, dV, tmax, Phimax, step_phi) 

    if counter == 10:
        return None, None, None, None
    
    high_T_vevs = []
    counter2 = 0
    max_min = False
    while counter2 <= 6:
        counter2 += 1
        temps = np.arange(tmax, t0 - dt, -dt)
        temps[temps < dt/10] = 0
        temps = np.sort(np.unique(temps))[::-1]
        counter2_2ndorder_limit = 3
        for T in temps:
            res = find_extrema(V, dV, T, Phimax, step_phi / counter2)     
            if res[0] == ['min']:
                if counter2 <= 1:
                    tmax = T + dt
                    dt = dt/10
                    break

                if max_min:
                    tmax = T_max_min + dt
                    dt = dt/10
                    break

                if not max_min and counter2 <= counter2_2ndorder_limit:
                    tmax = T + dt
                    dt = dt/10
                    break

                if not max_min and counter2 > counter2_2ndorder_limit:
                    return None, None, None, None # This is a 2nd order transition

            if res[0][-2:] == ['max', 'min']:
                max_min, T_max_min = True, T
                if len(res[0]) == 2:
                    if res[2][0] < 0:
                        tmax = T + dt
                        dt = dt/10
                        step_phi = step_phi / 10
                        break

                    false_min_loc, false_min_val = 0, 0

                if len(res[0]) > 2:
                    extrema_kind, extrema_loc, extrema_val = np.array(res[0][:-2]), np.array(res[1][:-2]), np.array(res[2][:-2])
                    try:
                        false_min_val = np.min(extrema_val[extrema_kind == 'min'])
                        false_min_loc = extrema_loc[extrema_val == false_min_val]
                        false_min_loc = false_min_loc[0] if len(false_min_loc) == 1 else false_min_loc
                        if false_min_val > 0:
                            false_min_loc, false_min_val = 0, 0
                    except:
                        false_min_loc, false_min_val = 0, 0

            
            if res[0][-2:] == ['max', 'min'] and res[2][-2:][0] > false_min_val and res[2][-2:][1] < false_min_val:
                max_min, T_max_min = True, T
                high_T_vevs.append(res[1][-2:][1])
                # Barrier
                if T == 0:
                    return None, None, None, None

                if T == tmax:
                    return tmax, np.array(high_T_vevs), res[1][-2:], [false_min_loc, false_min_val]
                
                if np.abs( (res[2][-2:][1] - false_min_val) / (res[2][-2:][0] - res[2][-2:][1]) ) <= precision:
                    return T, np.array(high_T_vevs), res[1][-2:], [false_min_loc, false_min_val]
                else:
                    res_return = res
                    false_min_loc_return, false_min_val_return = false_min_loc, false_min_val
                    tchange = T
                    tmax = T + dt
                    dt = dt/10
                    break
            else:
                # No barrier
                if T == temps[-1]:
                    return None, None, None, None
                else:
                    continue
    
    if tchange is not None:
        return tchange, np.array(high_T_vevs), res_return[1][-2:], [false_min_loc_return, false_min_val_return]
    else:
        return None, None, None, None


def refine_Tmin(T_min, V_physical, dV_physical, maxvev, log_10_precision = 6):
    if T_min > 0:
        for i in range(0, log_10_precision + 1):
            k = 1
            barrier = True

            while (barrier and (1 - k * 10**(-i)) > 0) or (i==0):
                if i==0:
                    t_temp = 0
                else:
                    t_temp = T_min * (1 - k * 10**(-i))

                extrema = find_extrema(V_physical, dV_physical, t_temp, Phimax = 2 * maxvev, step = maxvev * 1e-3)

                kind, loc, val = np.array(extrema[0]), np.array(extrema[1]), np.array(extrema[2])
                
                if len(kind) < 2:
                    barrier = False
                    Tsmooth = t_temp
                else:
                    false_min = 0
                    true_min_idx = np.argmin(val)

                    if len(kind) >=2 and kind[true_min_idx] == 'min' and kind[true_min_idx -1] == 'max' and true_min_idx >= 1:
                        barrier = True
                        if len(kind) > 2 and kind[true_min_idx - 2] == 'min' and val[true_min_idx - 2] < 0:
                            false_min = val[true_min_idx - 2]
                        else:
                            false_min = 0
                    else:
                        barrier = False
                    
                    if val[true_min_idx] < false_min and val[true_min_idx - 1] > false_min and kind[true_min_idx] == 'min' and kind[true_min_idx -1] == 'max':
                        barrier = True
                    else:
                        barrier = False

                    if barrier:
                        if i==0:
                            return 0
                        else:
                            T_min = t_temp
                            k += 1

                if i == 0:
                    break

        return T_min
    else:
        return T_min


def compute_logP_f(m, V_min_value, S3overT, true_vev, false_vev, v_w, units = 'GeV', cum_method='cumulative_simpson', return_all = False):
    # Method
    # cum_method is kept for compatibility but the iterative approach uses trapezoidal rule
    
    V = m.Vtot

    # Sort Temps descending (High T to Low T) for iterative computation
    Temps = np.array(sorted(V_min_value.keys(), reverse=True))
    steps = len(Temps)
    T_step = (Temps[0] - Temps[-1]) * 1e-3 # Ensure positive step for derivative
    
    dVdT = lambda phi, T : (V(np.array([phi]), T + T_step) - V(np.array([phi]), T - T_step)) / (2. * T_step) - s_SM(T, units = units)
    d2VdT2 = lambda phi, T : (dVdT(phi, T + T_step) - dVdT(phi, T - T_step)) / (2. * T_step)
    
    # Precompute quantities that depend only on T (not on P_f)
    e_vacuum_full = np.array([-V_min_value[t] for t in Temps])
    
    e_radiation = np.pi**2 * g_rho(Temps / convert_units[units]) * Temps**4 / 30
    
    # Action and Decay width
    S3_T = np.array([S3overT[t] for t in Temps])
    Gamma_list = Temps**4 * (S3_T / (2 * np.pi))**(3/2) * np.exp(-S3_T)
    
    # V''(phi, T) / V'(phi, T) components
    # ratio_V = P_f * ratio_V_false + (1 - P_f) * ratio_V_true
    ratio_V_false = np.array([d2VdT2(false_vev[t], t) / dVdT(false_vev[t], t) for t in Temps])
    ratio_V_true = np.array([d2VdT2(true_vev[t], t) / dVdT(true_vev[t], t) for t in Temps])
    
    # Initialize arrays
    logP_f = np.zeros_like(Temps)
    H = np.zeros_like(Temps)
    ratio_V = np.zeros_like(Temps)
    
    # Initial condition at T_max (index 0)
    # Assume P_f = 1 => logP_f = 0
    logP_f[0] = 0.0
    H[0] = np.sqrt((e_vacuum_full[0] + e_radiation[0]) / 3) / (M_pl * convert_units[units])
    ratio_V[0] = ratio_V_false[0]
    
    # Accumulators for integrals
    # J: integral of ratio_V
    # M: integral of A
    # K0, K1, K2, K3: integrals of B * M^n
    J = 0.0
    M = 0.0
    K0 = 0.0
    K1 = 0.0
    K2 = 0.0
    K3 = 0.0
    
    # Previous values for trapezoidal rule
    # A = ratio_V / H * exp(J/3)
    # B = ratio_V * Gamma / H * exp(-J)
    
    A_prev = ratio_V[0] / H[0] * np.exp(J / 3.0) # J=0
    B_prev = ratio_V[0] * Gamma_list[0] / H[0] * np.exp(-J) # J=0
    
    for i in range(1, steps):
        dt = Temps[i] - Temps[i-1] # This is negative
        
        # Estimate H[i] and ratio_V[i] using previous P_f
        P_f_prev = np.exp(logP_f[i-1])
        H[i] = np.sqrt((e_vacuum_full[i] * P_f_prev + e_radiation[i]) / 3) / (M_pl * convert_units[units])
        ratio_V[i] = ratio_V_false[i] * P_f_prev + ratio_V_true[i] * (1.0 - P_f_prev)
        
        # Update J: integral of ratio_V from T_max to T_i
        dJ = 0.5 * (ratio_V[i-1] + ratio_V[i]) * dt
        J += dJ
        
        # Calculate current A and B
        A_curr = ratio_V[i] / H[i] * np.exp(J / 3.0)
        B_curr = ratio_V[i] * Gamma_list[i] / H[i] * np.exp(-J)
        
        # Update M: integral of A from T_max to T_i
        dM = 0.5 * (A_prev + A_curr) * dt
        M_prev_val = M # Store previous M for trapezoidal of K
        M += dM
        
        # Update K integrals
        # K0: integral of B
        dK0 = 0.5 * (B_prev + B_curr) * dt
        K0 += dK0
        
        # K1: integral of B * M
        dK1 = 0.5 * (B_prev * M_prev_val + B_curr * M) * dt
        K1 += dK1
        
        # K2: integral of B * M^2
        dK2 = 0.5 * (B_prev * M_prev_val**2 + B_curr * M**2) * dt
        K2 += dK2
        
        # K3: integral of B * M^3
        dK3 = 0.5 * (B_prev * M_prev_val**3 + B_curr * M**3) * dt
        K3 += dK3
        
        # Calculate L(T_i)
        # L = - [ K3 - 3 M K2 + 3 M^2 K1 - M^3 K0 ]
        L = - (K3 - 3 * M * K2 + 3 * M**2 * K1 - M**3 * K0)
        
        logP_f[i] = - 4. / 243. * np.pi * v_w**3 * L
        # Ensure logP_f is non-positive (P_f <= 1)
        logP_f[i] = np.minimum(logP_f[i], 0)
        
        # Recompute H[i] and ratio_V[i] with updated P_f
        P_f_curr = np.exp(logP_f[i])
        H[i] = np.sqrt((e_vacuum_full[i] * P_f_curr + e_radiation[i]) / 3) / (M_pl * convert_units[units])
        ratio_V[i] = ratio_V_false[i] * P_f_curr + ratio_V_true[i] * (1.0 - P_f_curr)
        
        # Update prevs for next step
        A_prev = ratio_V[i] / H[i] * np.exp(J / 3.0)
        B_prev = ratio_V[i] * Gamma_list[i] / H[i] * np.exp(-J)
    
    # Reverse arrays to return in ascending order (Low T to High T) to match original behavior
    if return_all:
        return logP_f[::-1], Temps[::-1], ratio_V[::-1], Gamma_list[::-1], H[::-1], e_vacuum_full[::-1], e_radiation[::-1]  
    else:
        return logP_f[::-1], Temps[::-1], ratio_V[::-1], Gamma_list[::-1], H[::-1]


def N_bubblesH(Temps, Gamma, logP_f, H, ratio_V):
    integrand = Gamma * np.exp(logP_f) * ratio_V / H**4
    integral = cumulative_trapezoid(np.flip(integrand), initial=0, x=np.flip(Temps))

    return 4 * np.pi / 9 * np.flip(-integral)


def R_sepH(Temps, Gamma, logP_f, H, ratio_V):
    steps = len(Temps)

    # To store result
    n = np.zeros_like(Temps)
    
    # Function for the first integral
    f_ext = Gamma * ratio_V * np.exp(logP_f) / (3 * H)
    
    for i in range(steps - 1):
        cum_ratio_V = cumulative_trapezoid(ratio_V[i:], x=Temps[i:], initial=0)
        
        f1 = f_ext[i:] * np.exp(- cum_ratio_V)
        n[i] = trapezoid(f1, x=Temps[i:])

    return n**(-1/3) * H, n**(-1/3)


def R0(T, S3_T, V_exit):
    E0V = S3_T[T] * T / 2 # this is the potential energy contribution only
    DV =  - V_exit[T] # V(np.array([phisym]), T) - V(np.array([phi0]), T)
    r0 = ((3 * E0V / (4*np.pi*DV))**(1/3.0))
    return r0[0]


def is_increasing(arr):
    return np.all(arr[:-1] <= arr[1:])

class Temperatures:
    def __init__(self, lambda_, g, physical_vev, units = 'GeV', T_step = 1e-2, refine_Tmin_precision = 2):
        self.T_max, self.T_nuc, self.T_perc, self.T_completion, self.T_min = None, None, None, None, None
        self.Vf_contracting_at_T_perc, self.Vf_contracting_at_T_completion, self.Vf_contracting_somewhere = False, False, False

        from utils import interpolation_narrow
        self.lambda_ = lambda_
        self.g = g
        self.physical_vev = physical_vev

        self.units = units
        self.T_step = T_step

        self.dp = model(self.physical_vev, self.lambda_, self.g, xstep = self.physical_vev * 1e-3, Tstep = self.physical_vev * 1e-3)
        self.V = self.dp.DVtot

        if self.V(np.array([self.physical_vev]), 0) < 0:
            self.high_vev = {}
            self.S3overT = {}
            self.V_min_value = {}
            self.phi0 = {}
            self.false_vev = {}
            self.V_exit = {}
            
            self.dV = self.dp.gradV
    
            self.T_max, self.vevs_max, self.max_min_vals, _ = find_T_max(self.V, self.dV, precision = 1e-2, Phimax = 2*self.physical_vev, step_phi = self.physical_vev * self.T_step, tmax=2.5 * self.physical_vev)
            self.T_min, self.vevs_min, _ = find_T_min(self.V, self.dV, tmax=self.T_max, precision = 1e-2, Phimax = 2*self.physical_vev, step_phi = self.physical_vev * self.T_step, max_min_vals = self.max_min_vals)

            self.maxvev = np.max(np.concatenate((self.vevs_max, self.vevs_min))) if self.T_max is not None and self.T_min is not None else None
            self.T_min  = refine_Tmin(self.T_min, self.V, self.dV, self.maxvev, log_10_precision = refine_Tmin_precision) if self.T_min is not None else None

            if self.T_max is not None and self.T_min is not None:
                x = np.linspace(self.T_min, self.T_max, 120)
                vec = np.vectorize(self.action_over_T)
                vec(x)

            if self.V_min_value != {}:
                counter = 0
                while counter <= 1:
                    if counter == 1:
                        x = np.linspace(np.nanmax([self.T_min, 0.99 * self.T_completion]), np.nanmin([self.T_max, 1.01 * self.T_nuc]), 120, endpoint = True)
                        vec(x)
                    self.logP_f, self.Temps, self.ratio_V, self.Gamma, self.H = compute_logP_f(self.dp, self.V_min_value, self.S3overT, v_w = 1, units = self.units, cum_method= 'None')
                    self.nH = N_bubblesH(self.Temps, self.Gamma, self.logP_f, self.H, self.ratio_V)
                    self.mask_nH = ~np.isnan(self.nH)
                    self.mask_logP_f = ~np.isnan(self.logP_f)

                    self.T_nuc = interpolation_narrow(self.nH[self.mask_nH], self.Temps[self.mask_nH], 1)
                    self.T_perc = interpolation_narrow(self.logP_f[self.mask_logP_f], self.Temps[self.mask_logP_f], np.log(0.71))
                    self.T_completion = interpolation_narrow(self.logP_f[self.mask_logP_f], self.Temps[self.mask_logP_f], np.log(0.01))

                    if self.T_completion is not None:
                        idx_compl = np.max([np.argmin(np.abs(self.Temps - self.T_completion)), 1])
                        idx_compl = np.min([idx_compl, len(self.Temps) - 2])

                        test_completion = np.array([self.logP_f[idx_compl - 1], self.logP_f[idx_compl], self.logP_f[idx_compl + 1]])
                        test_completion = test_completion[~np.isnan(test_completion)]
                        if not is_increasing(test_completion):
                            print("P_f not decreasing at completion temperature for", (self.lambda_, self.g), ":", np.exp(test_completion))
                            self.T_completion = None

                    if counter == 1:
                        self.d_dT_logP_f = np.gradient(self.logP_f, self.Temps)
                        self.log_at_T_perc = interpolation_narrow(self.Temps, self.d_dT_logP_f, self.T_perc)
                        self.ratio_V_at_T_perc = interpolation_narrow(self.Temps, self.ratio_V, self.T_perc)
                        self.log_at_T_completion = interpolation_narrow(self.Temps, self.d_dT_logP_f, self.T_completion)
                        self.ratio_V_at_T_completion = interpolation_narrow(self.Temps, self.ratio_V, self.T_completion)
                        self.Vf_contracting_at_T_perc = True if self.ratio_V_at_T_perc <= self.log_at_T_perc else False
                        self.Vf_contracting_at_T_completion = True if self.ratio_V_at_T_completion <= self.log_at_T_completion else False
                        mask_T = (self.Temps >= self.T_completion) & (self.Temps <= self.T_nuc)
                        self.Vf_contracting_somewhere = True if np.sum(self.d_dT_logP_f[mask_T] > self.ratio_V[mask_T]) > 0 else False
                            
                    counter += 1
                
                self.T_nuc = self.nan_to_none(self.T_nuc)
                self.T_perc = self.nan_to_none(self.T_perc)
                self.T_completion = self.nan_to_none(self.T_completion)

                if self.T_completion is not None:
                    self.action_over_T(self.T_perc)

                    self.alpha = self.alpha_th_bar(self.T_perc)[0]
                    self.alpha_inf = self.c_alpha_inf(self.T_perc)[0]
                    self.alpha_eq = self.c_alpha_eq(self.T_perc)[0]

                    if self.alpha < self.alpha_inf or self.alpha < self.alpha_eq:
                        #print(f"Non-runaway regime for {(self.lambda_, self.g)}: alpha = {self.alpha:.2e}, alpha_inf = {self.alpha_inf:.2e}, alpha_eq = {self.alpha_eq:.2e}")
                        self.T_completion = None
                    

            else:
                self.T_perc, self.T_completion, self.T_nuc = None, None, None
        else:
            self.T_max, self.T_nuc, self.T_perc, self.T_completion, self.T_min = None, None, None, None, None

        self.temperatures = [self.T_max, self.T_nuc, self.T_perc, self.T_completion, self.T_min]
        self.temperatures = [temp for temp in self.temperatures if temp is not None]
        self.consistency = all(self.temperatures[i] >= self.temperatures[i + 1] for i in range(len(self.temperatures) - 1))

    def action_over_T(self, T):
        instance = Vt_vec(T, self.V, self.dV, step_phi = 1e-3, precision = 1e-2, vev0 = self.maxvev)
        if instance.barrier:
            self.high_vev[T] = instance.vevT_original
            self.false_vev[T] = instance.phi_original_false_vev
            self.S3overT[T] = instance.action_over_T
            self.V_min_value[T] = instance.min_V
            self.phi0[T] = instance.phi0_min
            self.V_exit[T] = instance.V_exit
            return instance.action_over_T
        else:
            return None
        
    def nan_to_none(self, x):
        return x if not np.isnan(x) else None
    
    def cs2(self, T):
        return self.dp.dVdT(self.high_vev[T], T, include_radiation=True, include_SM = True) / (T * self.dp.d2VdT2(self.high_vev[T], T, include_radiation=True, include_SM = True))
    
    def alpha_th_bar(self, T):
        delta_rho = - self.V_min_value[T] -  T * (self.dp.dVdT(self.false_vev[T], T, include_radiation=True, include_SM = False,  units = self.units) - self.dp.dVdT(self.high_vev[T], T, include_radiation=True, include_SM = False,  units = self.units))
        delta_p = self.V_min_value[T] / self.cs2(T)
        wf = - T * self.dp.dVdT(self.false_vev[T], T, include_radiation=True, include_SM = True,  units = self.units)
        wf_DS = - T * self.dp.dVdT(self.false_vev[T], T, include_radiation=True, include_SM = False,  units = self.units)

        return (delta_rho - delta_p) / (3 * wf), (delta_rho - delta_p) / (3 * wf_DS)
    
    def c_alpha_inf(self, T):
        v_true = self.high_vev[T]
        v_false = self.false_vev[T]
        Dm2_photon = 3 * self.g**2 * (v_true**2 - v_false**2)
        Dm2_scalar = 3 * self.lambda_ * (v_true**2 - v_false**2) 
        numerator = (Dm2_photon + Dm2_scalar) * T**2 / 24
        rho_tot = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = True, units = self.units) ) / 4
        rho_DS = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = False, units = self.units) ) / 4
        return numerator/ rho_tot, numerator / rho_DS

    def c_alpha_eq(self, T):
        v_true = self.high_vev[T]
        v_false = self.false_vev[T]
        numerator = (self.g**2 * 3 * (self.g * (v_true - v_false)) * T**3)
        rho_tot = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = True, units = self.units) ) / 4
        rho_DS = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = False, units = self.units) ) / 4
        return numerator / rho_tot, numerator / rho_DS