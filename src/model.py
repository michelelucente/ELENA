import numpy as np
from scipy.special import kv

from cosmoTransitions import generic_potential, helper_functions
from utils import s_SM

#Function to suppress the thermal masses when necessary
def K2(x):
    x = np.asanyarray([x])
    vectorized_kn = np.vectorize(kn, otypes=[float])
    result=vectorized_kn(2,x)*x**2/2
    return  result[0]
 
#The usual JB and JF thermal functions
def T4Jb(T,mSq):
    if  T == 0:
        return 0
    else:
        return  T**4*ft.Jb_spline(mSq/(T**2))

#To plot the potential
def plotV(Tc=100):
    VV=[]
    XX=[]
    steps = 600
    Tx=Tc
    for i in range(steps):
        minZero=m.approxZeroTMin()
        trueVEV= 1.5 * m.findMinimum(X=[minZero],T=Tx)
        falseVEV=0.0#m.findMinimum(X=[0],T=Tx)
        Xx=falseVEV- i/steps * (falseVEV - trueVEV[0])
        Vx=m.Vtot([Xx],Tx)-m.Vtot([0.0],Tx)
        XX.append(Xx)
        VV.append(Vx)
        i=i+1
    return XX,VV


# Real and Imaginary units
Re1 = complex(1., 0.)
Imi = complex(0., 1.)

# Constants and other values
ciZp = 1./3.

# Useful functions
def kdelta(a,b):
    if isinstance(a, np.ndarray):
        mask = a == b
        return 1. * mask
    else:
        if a == b:
            return 1.
        else:
            return 0.



#####################################################################################################################
#                                           MODEL CLASS
#####################################################################################################################

  
#This is the definition of our model
class model(generic_potential.generic_potential):

    def init(self, w = 100, lambdaf = 0.1, gX = 0.1, xstep=0.00005,Tstep=0.00005, daisyType=3, CV_method='on-shell'):
        """
          Setting up the parameters
        """
        
        # Number of field-dimensions in the theory.
        self.Ndim = 1

        # Input parameters
        self.w = w
        self.lambdaf = lambdaf
        self.gX = gX
        
        # Important to set these parameters when we move w or if it doens't find the transition
        
        ###################
        self.x_eps = xstep
        self.T_eps = Tstep
        #self.Tmax = Tmax
        ##################
        
        # Model parameters
        
        self.muf = self.w*np.sqrt(self.lambdaf)
        self.mu2 = self.muf**2
        
        self.daisyResum = daisyType
        self.CV_method = CV_method
        self.deriv_order = 2


    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -3. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return any([np.array([X])[...,0] < - 0.5e-6])

    def V0(self, X):
        """
        This method defines the tree-level potential.
        """
        
        X = np.asanyarray(X)
        v = X[...,0]
        r = - self.muf**2 * (v**2.)/2. + self.lambdaf * (v**4.)/4.

        return r
    
    # Coleman-Weinberg potential
    def Vcw(self, X):
        
        w0, lambdaphi,gZp = self.w, self.lambdaf, self.gX
        
        X = np.asanyarray(X)
        phi = X[...,0]
        
        # Definition of parameter mu_phi
        muphi = w0 * np.sqrt(lambdaphi)
        
        # CW potential
        return (1 / (64. * np.pi**2) * ( 6 * gZp**4 * muphi**2 * phi**2 / lambdaphi - 4 * muphi**2 * (muphi**2 - 3 * lambdaphi * phi**2) 
               + 3 * gZp**4 * phi**4 * (-1.5 + np.log(lambdaphi * phi**2 / muphi**2 + kdelta(phi,0))) 
               + 0.5 * (muphi**2 - 3 * lambdaphi * phi**2)**2 * (-3. 
               + 2 * np.log(0.5 * np.abs(1 - 3 * lambdaphi * phi**2 / muphi**2) + kdelta(3 * lambdaphi * phi**2 - muphi**2,0 ))) 
               + (muphi**2 - lambdaphi * phi**2)**2 * (-1.5 
               + np.log(0.5 * np.abs(1 - lambdaphi * phi**2 / muphi**2) + kdelta(lambdaphi * phi**2 - muphi**2, 0))) ) )
               
    
    def VDaisy(self, X, T):
    
        w0, lambdaphi,gZp = self.w, self.lambdaf, self.gX
        
        X = np.asanyarray(X)
        phi = X[...,0]
        T = np.asanyarray(T, dtype=float)
        
        
        # Definition of parameter mu_phi
        muphi = w0 * np.sqrt(lambdaphi)
        
        if isinstance(T, np.ndarray):
            T[T==0] = 1e-3
        else:
            if T==0:
                T = 1e-3

        
        # Squared masses
        m2_gold = lambdaphi * phi**2 - muphi**2
        m2_higgs = 3 * lambdaphi * phi**2 - muphi**2
        m2_dp = gZp**2 * phi**2
        
        # Prefactor and other terms
        prefactor = T / 12. / np.pi
        
        first_terms = (Re1 * m2_dp)**1.5 + (Re1 * m2_gold)**1.5 + (Re1 * m2_higgs)**1.5
        
        first_mod_term = Re1 * (m2_dp + 0.5 * ciZp * gZp**2 * T**2 * (m2_dp/T**2 * kv(2,Re1 * gZp * phi / T + kdelta(phi,0)) + 2 * kdelta(phi,0)))
        first_mod_term = first_mod_term**1.5
        
        second_mod_term = Re1 * (m2_gold + 0.25 * gZp**2 * T**2 + 0.5 * lambdaphi * T**2 )
        second_mod_term = second_mod_term**1.5
        
        third_mod_term = Re1 * (m2_higgs + 0.25 * gZp**2 * T**2 + 0.5 * lambdaphi * T**2 )
        third_mod_term = third_mod_term**1.5
        
        return np.real(prefactor * (first_terms - first_mod_term - second_mod_term - third_mod_term))
    
    
    def boson_massSq(self, X, T):
        """
        This method defines the squared boson mass spectrum of this theory. It is
        returned with the respective fields' dofs and renormalization constants.
        If the daisies are resummed
        """
        X = np.array(X)
        v = X[...,0]
        g = self.gX
        mDP2 = np.array([(g**2.)*(v**2.)])

        if self.daisyResum == 1 or 2:
            # Temperature corrections for scalar (i.e. Higgs)
            # "Even" and "Odd" denote the CP-symmetry factor
            cS = self.lambdaf / 2. + g**2 / 4.
            cDP = np.real(0.5 * ciZp * g**2. * (mDP2/T**2 * kv(2,Re1 * g * v / T + kdelta(v,0)) + 2 * kdelta(v,0))) if T>0 else 0

            MSqEven = np.array([-self.mu2 + cS*T**2. + 3.*self.lambdaf*v**2.])
            MSqOdd = np.array([-self.mu2 + cS*T**2. + self.lambdaf*v**2.])

            # Temperature corrections for longitudinal gauge bosons (i.e. dark photon)
            mDP2L = mDP2 + cDP*T*T
        else:
            MSqEven = np.array([-self.mu2 + 3.*self.lambdaf*v**2.])
            MSqOdd = np.array([-self.mu2 + self.lambdaf*v**2.])
            mDP2L = mDP2 
        
        # In some particular cases, MSqEven and MSqOdd have one more dimension with just one element. This is to fix that error
        if MSqEven.ndim > mDP2.ndim:
            MSqEven = MSqEven[0]
            MSqOdd = MSqOdd[0]
            
        M = np.concatenate((MSqEven, MSqOdd, mDP2, mDP2L))
        M = np.rollaxis(M, 0, len(M.shape))
        dof = np.array([1, 1, 2, 1])
        # c_i = 3/2 for fermions and scalars, 5/6 for gauge bosons
        c = np.array([3./2., 3./2., 5./6., 5./6.])
        return M, dof, c

    def fermion_massSq(self, X):
        """
        Since fermions are not included in this theory, their mass spectrum and
        dofs are set to zero.
        """
        M = 0
        dof = 0
        return M, dof
        
    def Vct(self, X):
        """
        The counterterm lagranian is the same as the tree level lagrangian but
        with masses and couplings replaced by counter term values (i.e. here
        mu2 -> dmu2 and l -> dl). Assume potential of the form
        V = - 1/2 mu**2 h**2 + lambda/4 h**4 where h is the investigated scalar field
        """
        X = np.array(X)
        v = X[...,0]
        #r = - self.dmu2*(v**2.)/2. + self.dl*(v**4.)/4.
        return 0.
        
    
    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential is calculated by adding
        up the tree level potential, the one-loop-zero-T correction, the respective
        counter terms, and (depending on the daisy resummation scheme) the one-loop-
        temperature-dependent corrections.
        
        Parameters
        ----------
        X : array_like
            Field value(s). 
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)

        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        
        if self.CV_method == 'on-shell':
            y += self.Vcw(X)
        else:
            y += self.V1(bosons0, fermions)
            y += self.Vct(X)

        # Parwani (1992) prescription. All modes resummed.
        if self.daisyResum == 1:
            bosonsT = self.boson_massSq(X,T)
            y += self.V1T(bosonsT, fermions, T, include_radiation)
        # Carrington (1992), Arnold and Espinosa (1992) prescription. Zero modes only.
        elif self.daisyResum == 2:
            # Absolute values are a hack. Potential trustworthy only away from where m2T, m20 < 0.
            bosons0 = self.boson_massSq(X,0.*T)
            bosonsT = self.boson_massSq(X,T)
            m20, nb, c = bosons0
            m2T, nbT, cT = bosonsT
            Vdaisy = np.real(-(T/(12.*np.pi))*np.sum ( nb*(pow(m2T+0j,1.5) - pow(m20+0j,1.5)), axis=-1))
            y += self.V1T(bosons0, fermions, T, include_radiation) + Vdaisy
        # No daisy resummation
        elif self.daisyResum == 0:
            bosons0 = self.boson_massSq(X,0.*T)
            y += self.V1T(bosons0, fermions, T, include_radiation)
        elif self.daisyResum == 3:
            y += self.V1T(self.boson_massSq(X,0.*T), fermions, T, include_radiation) + self.VDaisy(X,T)

        return y

    def DVtot(self, X, T):
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.
        """
        #X0 = np.zeros(self.Ndim)
        return self.Vtot(X,T,False) - self.Vtot(X*0,T,False)  

    def gradV(self, X, T):
        """
        Find the gradient of the full effective potential.

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """
        try:
            f = self._gradV
        except:
            # Create the gradient function
            self._gradV = helper_functions.gradientFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._gradV
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        return f(X,T,False)
    
    def approxZeroTMin(self):

        return [np.array([self.w])]  #   self.w       qua ci puoi mettere un valore, cosi' va pi
    
    def energyDensity(self,X,T,include_radiation=True):
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.V1T_from_X(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT += 8*self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        
        return self.Vtot(X,T, include_radiation) - T * dVdT

    def _d1_dT(self, func, T0):
        """
        Temperature derivative with a 5-point stencil (O(h^4)).
        Falls back to a forward stencil close to T=0 to avoid negative temperatures.
        """
        T0 = np.asanyarray(T0, dtype=float)
        T_flat = T0.ravel()
        h = self.T_eps
        deriv = np.empty_like(T_flat, dtype=float)

        central_mask = T_flat >= 2 * h
        if np.any(central_mask):
            Tc = T_flat[central_mask]
            f_m2 = func(np.maximum(Tc - 2 * h, 0.0))
            f_m1 = func(np.maximum(Tc - h, 0.0))
            f_p1 = func(Tc + h)
            f_p2 = func(Tc + 2 * h)
            deriv[central_mask] = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)

        edge_mask = ~central_mask
        if np.any(edge_mask):
            Te = T_flat[edge_mask]
            f0 = func(Te)
            f1 = func(Te + h)
            f2 = func(Te + 2 * h)
            f3 = func(Te + 3 * h)
            f4 = func(Te + 4 * h)
            deriv[edge_mask] = (-25 * f0 + 48 * f1 - 36 * f2 + 16 * f3 - 3 * f4) / (12 * h)

        return deriv.reshape(T0.shape)


    def _d2_dT2(self, func, T0):
        """
        Second temperature derivative with a 5-point stencil (O(h^4)).
        Uses a forward stencil close to T=0 to avoid stepping to negative temperatures.
        """
        T0 = np.asanyarray(T0, dtype=float)
        T_flat = T0.ravel()
        h = self.T_eps
        deriv = np.empty_like(T_flat, dtype=float)

        central_mask = T_flat >= 2 * h
        if np.any(central_mask):
            Tc = T_flat[central_mask]
            f_m2 = func(np.maximum(Tc - 2 * h, 0.0))
            f_m1 = func(np.maximum(Tc - h, 0.0))
            f0 = func(Tc)
            f_p1 = func(Tc + h)
            f_p2 = func(Tc + 2 * h)
            deriv[central_mask] = (-f_p2 + 16 * f_p1 - 30 * f0 + 16 * f_m1 - f_m2) / (12 * h**2)

        edge_mask = ~central_mask
        if np.any(edge_mask):
            Te = T_flat[edge_mask]
            f0 = func(Te)
            f1 = func(Te + h)
            f2 = func(Te + 2 * h)
            f3 = func(Te + 3 * h)
            f4 = func(Te + 4 * h)
            deriv[edge_mask] = (35 * f0 - 104 * f1 + 114 * f2 - 56 * f3 + 11 * f4) / (12 * h**2)

        return deriv.reshape(T0.shape)


    def dVdT(self, X, T0, include_radiation=True, include_SM = False, units = 'GeV'):
        T0 = np.asanyarray(T0, dtype=float)
        X = [X]
        V = lambda T : self.Vtot(X, T, include_radiation=include_radiation)

        dVdT = self._d1_dT(V, T0)

        if include_SM:
            dVdT -= s_SM(T0, units=units)
        
        return dVdT
    
    
    def d2VdT2(self, X, T0, include_radiation=True, include_SM = False, units = 'GeV'):
        T0 = np.asanyarray(T0, dtype=float)
        X = [X]

        V = lambda T : self.Vtot(X, T, include_radiation=include_radiation)
        d2V = self._d2_dT2(V, T0)

        if include_SM:
            d2V -= self._d1_dT(lambda T: s_SM(T, units=units), T0)

        return d2V
    
    def e_minus_3p_div_4(self,X,T,include_radiation=True):
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.V1T_from_X(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT += 8*self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        return self.Vtot(X,T, include_radiation) - T * dVdT / 4.0
