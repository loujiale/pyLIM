r"""
Linear Inverse Model class and methods.

Sam Lillo
"""

import numpy as np
from datetime import datetime as dt,timedelta
from numpy.linalg import inv, pinv, eig, eigvals, eigh, matrix_power
from scipy.linalg import logm, expm, solve_sylvester
import pickle
import copy
from .dataset import *
from .tools import *


# %%

class Model(object):
    r"""Linear inverse forecast model.
    
    This class uses a calibration dataset to make simple linear forecasts.
    
    Notes
    -----
    Based on the LIM described by M. Newman (2013) [1].
    
    References
    ----------
    .. [1] Newman, M. (2013), An Empirical Benchmark for Decadal Forecasts of 
       Global Surface Temperature Anomalies, J. Clim., 26(14), 5260–5269, 
       doi:10.1175/JCLI-D-12-00590.1.
    ....
    """
    
    def __init__(self, tau0_data, tau1_data=None, tau1n=None,
                 fit_noise=False, max_neg_Qeval=5):
        r"""
        Parameters
        ----------
        tau0_data : ndarray
            Data for calibrating the LIM.  Expects
            a 2D MxN matrix where M (rows) represent the sampling dimension and
            N(columns) represents the feature dimension (e.g. spatial grid
            points).
        tau1_data : ndarray, optional
            Data with lag of tau=1.  Used to calculate the mapping term, G1,
            going from tau0 to tau1.  Must be the same shape as tau0_data.  If
            not provided, tau0_data is assumed to be sequential and
            nelem_in_tau1 and tau0_data is used to calculate lag covariance.
        tau1n : int, optional
            Number of time samples that span tau=1.  E.g. for daily data when
            a forecast tau is equivalent to 1 week, tau1n should be 7.
            Used if tau1_data is not provided.
        fit_noise : bool, optional
            Whether to fit the noise term from calibration data. Used for
            noise integration
        max_neg_Qeval : int, optional
            The maximum number of allowed negative eigenvalues in the Q matrix.
            Negative eigenvalues suggest inclusion of too many modes, but a few
            spurious ones are common. Defaults to 5.
        """
        print('Initializing LIM forecasting object...')

        self.tau1n = tau1n

        if tau0_data.ndim != 2:
            print(('LIM calibration data is not 2D '
                          '(Contained ndim={:d})').format(tau0_data.ndim))
            raise ValueError('Input LIM calibration data is not 2D')
        
        if tau1_data is not None:
            if not tau1_data.shape == tau0_data.shape:
                print('LIM calibration data shape mismatch. tau1: {}'
                             ' tau0: {}'.format(tau1_data.shape,
                                                tau0_data.shape))
                raise ValueError('Tau1 and Tau0 calibration data shape '
                                 'mismatch')
            self.X0 = np.matrix(tau0_data)
            self.X1 = np.matrix(tau1_data)
        else:
            self.X0 = np.matrix(tau0_data[0:-tau1n, :])
            self.X1 = np.matrix(tau0_data[tau1n:, :])  

        self._calc_m(tau=1)
        self.Q = -(self.L @ self.C0 + self.C0 @ self.L.T)
        
    def _calc_m(self, tau=1):
        r"""
        Calculate L and G for forecasting (using nomenclature
        from Newman 2013)

        Parameters
        ----------
        X0 : ndarray
            State vector at time=0.  MxN where M is number of samples, 
            and N is the number of features.
        X1 : ndarray
            State vector at time=tau.  MxN where M is number of samples, 
            and N is the number of fatures.
        tau : float
            lag time (in units of tau) that we are calculating G for.  This is 
            used to check that all modes of L are damped. Default to 1.
        """

        tau_n = tau * self.tau1n

        # Covariance matrices C(0) and C(tau)
        self.C0 = np.matmul(self.X0.T, self.X0) / (self.X0.shape[0] - 1) 
        self.Ctau = np.matmul(self.X1.T, self.X0) / (self.X1.shape[0] - 1) 
        
        # Calculate G
        self.G1 = np.matmul(self.Ctau, pinv(self.C0))
        self.Geigs = eigvals(self.G1)
        
        # Calculate L
        self.L = (1/tau_n) * logm(self.G1)
        self.Leigs = eigvals(self.L)
        
        # Check if real part of Geigs are between 0 and 1
        # Growing mode detected if real(Geigs) >1 
        # Nyquist problem if real(Geigs) <0
#        if np.any(self.Geigs.real>1):
#            raise ValueError('Eigenvalues > 1 detected in matrix G.')           
#        if np.any(Geigs.real<0):
#            raise ValueError('Eigenvalues < 0 detected in matrix G.')    
        

    def rescale_Q(self):
        
        G_evals, G_evecs = eig(self.G1)
        L_evals, L_evecs = eig(self.L)

        self.Q = -(self.L @ self.C0 + self.C0 @ self.L.T)  # Noise covariance

        # Check if Q is Hermetian
        is_adj = abs(self.Q - self.Q.H)
        tol = 1e-10
        if np.any(abs(is_adj) > tol):
            raise ValueError('Determined Q is not Hermetian (complex '
                             'conjugate transpose is equivalent.)')

        q_evals, q_evecs = eigh(self.Q)
        sort_idx = q_evals.argsort()
        q_evals = q_evals[sort_idx][::-1]
        q_evecs = q_evecs[:, sort_idx][:, ::-1]
        num_neg = (q_evals < 0).sum()

        # Check max amplitude of negative eigval of Q vs max amp of pos eigval
        # Tolerance of 10% of max amp of pos
        # Compare trace of matrix vs trace of positive eigval, no smaller than 0.9
        # Maintain variance after setting negative eigval to 0
        if num_neg > 0:
            num_left = len(q_evals) - num_neg
            
            print(f'Removing {num_neg} negative eigenvalues and rescaling {num_left} '
                            'remaining eigenvalues of Q.')
            pos_q_evals = q_evals[q_evals > 0]
            scale_factor = q_evals.sum() / pos_q_evals.sum()
            print('Q eigenvalue rescaling: {:1.2f}'.format(scale_factor))

            q_evals = q_evals[:-num_neg]*scale_factor
            q_evecs = q_evecs[:, :-num_neg]
            self.Q_rescaled = q_evecs @ np.diag(q_evals) @ q_evecs.T
            
        else:
            scale_factor = None
            self.Q_rescaled = self.Q

        return q_evals, q_evecs, scale_factor
    
    
    def C0_from_Q(self):
        
        _=self.rescale_Q()
        C0 = solve_sylvester(self.L,self.L.T,-self.Q_rescaled)
        return C0
    

    def free_run(self,duration,dtfrac,Xinit=None):
        
        q_evals,q_evecs,_=self.rescale_Q()
        
        if Xinit is None:
            Xinit = np.zeros(len(self.L))
        X = [Xinit]
        Xout = [Xinit]
    
        deltat = 1/dtfrac
        times = np.arange(0,duration,deltat)
        tnlast = 0
        timer_start = dt.now()
        for prog,tn in enumerate(times):
            # Get gaussian noise vector
            noise = np.random.normal(size=len(self.L))
            # Integration            
            dX = self.L @ X[-1] * deltat + \
            np.array(q_evecs @ np.sqrt(q_evals)).squeeze()*noise * deltat
            
            if len(X)==1:
                X.append(X[-1]+dX)
            else:
                Xtmp = X[-1]
                X[-1] = X[-2]+2*dX
                X[-2] = Xtmp
            
            if np.floor(tn)>tnlast:
                Xout.append(X[-1])
        
            tnlast = tn
            
            update_progress('Integrating free run...',(prog+1)/len(times))
        print('--> Completed free run (%.1f seconds)' \
              % (dt.now()-timer_start).total_seconds())
                
        return Xout

    
    def optimal_growth(self,norm=None,lead_time=None):
        r"""
        Returns optimal growth structure for specified final norm.
        For L2, norm=None defaults to an identity matrix 
        """
        
        if lead_time is None:
            lead_time = self.tau1n
        if norm is None:
            norm = np.identity(self.L.shape[0])
        
        g = expm(self.L*lead_time)
        A = g.T @ np.matrix(norm) @ g
        growth, init = eigh(A)
        sort_idx = growth.argsort()
        growth = growth[sort_idx][::-1]
        init = init[:, sort_idx][:, ::-1]
        return growth, init

    
    def forecast(self, t0_data, lead_time):
        r"""
        Forecast on provided data.
        
        Performs LIM forecast over the times specified by the fcst_leads. 
        Forecast can be performed by calculating G for each time 
        period or by L for a 1-tau lag and then calculating each fcst_lead G 
        from that L matrix.
        
        Parameters
        ----------
        t0_data : ndarray
            Data to forecast from.  
            Expects a 1D or 2D array where M (rows) represent the 
            initialization time dimension
            and N(columns) represents the feature dimension.
        lead_time : List<int>
            A list of forecast lead times.  Each value is interpreted as a
            tau value, for which the forecast matrix is determined as G_1^tau.
            
        Returns
        -----
        fcst_out : ndarray-like
            LIM forecasts in a list of arrays, where the first dimension 
            corresponds to each lead time. Second dimension corresponds 
            to initialization times.
        """

        if isinstance(lead_time,(int,float)):
            lead_time = (lead_time,)

        print('--> Performing LIM forecast for lead times: '
                    + str(lead_time))

        fcst_out = []
        for lt in lead_time:
            g = expm(self.L*lt)
            Xf = np.matmul(g, np.matrix(t0_data).T)
            fcst_out.append( Xf.T )

        return fcst_out
    
    def _calc_delta(self):
        delta_tau = self.C0 - self.G1 @ self.C0 @ self.G1.T
    
    def save_precalib(self,filename):
        with open(filename,'w') as f:
            pickle.dump(self,f)
        print(f'Saved pre-calibrated LIM to {filename}')
    
    def save_to_netcdf(self,filename):
        
        vardict = {"X0": {'dims':("time","n"), 'data':self.X0},
                   "X1": {'dims':("time","n"), 'data':self.X1},
                   "C0": {'dims':("n","n"), 'data':self.C0},
                   "Ctau": {'dims':("n","n"), 'data':self.Ctau},
                   "L": {'dims':("n","n"), 'data':self.L},
                   "Leigs": {'dims':("n",), 'data':self.Leigs},
                   "G1": {'dims':("n","n"), 'data':self.G1},
                   "Geigs": {'dims':("n",), 'data':self.Geigs},
                   "Q": {'dims':("n","n"), 'data':self.Q}
                   }
        coords={"n": {'dims':("n",),'data':np.arange(len(self.C0))},
                "time": {'dims':("time",),'data':np.arange(len(self.X0))}
                }
    
        save_ncds(vardict,coords,filename=filename)

