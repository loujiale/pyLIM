r"""
Verification methods for LIM.

Sam Lillo, Matt Newman, John Albers
"""

import numpy as np
import math
from numpy.linalg import inv, pinv, eig, eigvals, eigh, matrix_power
from scipy.linalg import logm, expm
import pickle
from datetime import datetime as dt,timedelta
from .model import Model
from .tools import *

class Verif(object):
    r"""Verification methods for linear inverse forecast model.
    
    This class contains various methods for verification of the LIM.
    """
    
    def __init__(self,eofobjs,eof_lim):
        self.eofobjs = {k:eofobjs[k] for k in eof_lim.keys()}
        self.varobjs = [v for name in eof_lim.keys() for v in eofobjs[name].varobjs]
#        t = [set(v.time) for v in self.varobjs]
#        self.times = np.array(sorted(list(t[0].intersection(*t[1:]))))
        self.times = self.varobjs[0].time
        self.eof_lim = eof_lim
        # for each eofobj, get time and truncated PCs, assemble into dictionary
        # then concatenate into matrix PC x T
        p = {}
        for name in eof_lim.keys():
            time = eofobjs[name].varobjs[0].time
            pcs = eofobjs[name].pc[:,:eof_lim[name]]
            p[name] = [pc for t,pc in zip(time,pcs) if t in self.times]
        self.all_data = np.concatenate([p[name] for name in p.keys()],axis=1)
    
    def kfoldval(self,lead_times,k,tau1n,average=True,prop={}):
        
        default_prop = {'zero_init':None,'zero_model':None}
        for key in default_prop.keys():
            if key not in prop.keys():
                prop[key] = default_prop[key]

        zeroInitBool = np.full((sum(self.eof_lim.values()),), False, dtype=bool)
        if prop['zero_init'] is not None:
            for name in listify(prop['zero_init']):
                out = get_varpci(self.eof_lim,name)
                zeroInitBool[out[0]:out[1]]=True

        zeroModelBool = np.full((sum(self.eof_lim.values()), sum(self.eof_lim.values())), False, dtype=bool)
        if prop['zero_model'] is not None:
            for block in prop['zero_model']:
                out = []
                for name in block:
                    if name=='all':
                        out.append(np.arange(sum(self.eof_lim.values())))
                    else:
                        out.append(np.arange(*get_varpci(self.eof_lim,name)))
                for i in out[0]:
                    for j in out[1]:
                        zeroModelBool[i,j]=True
                
        # Make lead_times into a tuple
        if isinstance(lead_times,(int,float)):
            lead_times = (lead_times,)
        
        # Get times for tau0 and tau1 data by finding the intersection of all 
        # times and times + tau1n days
        times1 = np.intersect1d(self.times,self.times+timedelta(days = tau1n))
        times0 = times1-timedelta(days = tau1n)
        
        # Get tau0 and tau1 data by taking all data and all times and matching 
        # with the corresponding times for tau0 and tau1
        tau0_data = np.array([d for d,t in zip(self.all_data,self.times) if t in times0])
        tau1_data = np.array([d for d,t in zip(self.all_data,self.times) if t in times1])
        
        # nval is the size of each validation fold
        nval = len(times0)//k
        
        all_val_fcsts = {}
        all_val_variance = {}
        for ik in range(k):
            print(f'processing fold #{ik+1}')
            
            ''' Train the model. '''     
            # Get the tau0 and tau1 data for training that is outside of the given k-fold
            tau0_train = np.concatenate([tau0_data[:ik*nval],tau0_data[(ik+1)*nval:]],axis=0)
            tau1_train = np.concatenate([tau1_data[:ik*nval],tau1_data[(ik+1)*nval:]],axis=0)
            # Train the model
            model = Model(tau0_train,tau1_train,tau1n=tau1n)
            model.L[zeroModelBool]=0
                                   
            # Get the (time independent) variance from the model
            C0 = np.matrix(model.C0)
            Gtau = {lt:expm(np.matrix(model.L)*lt) for lt in lead_times}
            Etau = {lt:(C0 - Gtau[lt] @ C0 @ Gtau[lt].T) for lt in lead_times}
            
            ''' Make the forecast. '''
            # Times for forecast validation
            times_0_val = times0[ik*nval:(ik+1)*nval]
            times_lt_val = np.intersect1d(self.times,times_0_val+timedelta(days=int(max(lead_times))))
            times_0_val = times_lt_val-timedelta(days=int(max(lead_times)))
            
            # Initialization data for forecast validation
            tau0_val = np.array([d for d,t in zip(self.all_data,self.times) if t in times_0_val])
            for row,tmp in enumerate(tau0_val):
                tmp[zeroInitBool]=0
                tau0_val[row]=tmp
            fcst = model.forecast(tau0_val,lead_time=lead_times)
            if average:
                fcst = np.mean(fcst,axis=0)
                variance = np.mean([Etau[lt] for lt in lead_times],axis=0)
            else:
                fcst = np.array(fcst).swapaxes(0,1)
                variance = np.array([Etau[lt] for lt in lead_times])
            
            print(min(times_0_val),max(times_0_val))
            # Assign forecast arrays to dictionary with keys corresponding to 
            # initialization time
            for i,t in enumerate(times_0_val):
                all_val_fcsts[t] = fcst[i]
                all_val_variance[t] = variance
        
        self.fcsts = all_val_fcsts
        self.variance = all_val_variance

        
#        return all_val_fcsts,all_val_spread
    
#    def kfoldval(self,lead_times,k,tau1n):
#        
#        if isinstance(lead_times,(int,float)):
#            lead_times = (lead_times,)
#            
#        times_1_train = np.intersect1d(self.times,self.times+timedelta(days = tau1n))
#        times_0_train = times_1_train-timedelta(days = tau1n)
#        
#        tau0_data = np.array([d for d,t in zip(self.all_data,self.times) if t in times_0_train])
#        tau1_data = np.array([d for d,t in zip(self.all_data,self.times) if t in times_1_train])
#        
#        nval = len(times_0_train)//k
#        
#        all_val_fcsts = {}
#        all_val_spread = {}
#        for ik in range(k):
#            print(f'processing fold #{ik+1}')
#            
#            ''' Train the model. '''        
#            tau0_train = np.concatenate([tau0_data[:ik*nval],tau0_data[(ik+1)*nval:]],axis=0)
#            tau1_train = np.concatenate([tau1_data[:ik*nval],tau1_data[(ik+1)*nval:]],axis=0)
#            model = Model(tau0_train,tau1_train,tau1n=tau1n)
#            
#            C0 = np.matrix(model.C0)
#            Gtau = {lt:expm(np.matrix(model.L)*lt) for lt in lead_times}
#            Etau = {lt:(C0 - Gtau[lt] @ C0 @ Gtau[lt].T) for lt in lead_times}
#            spread = np.mean([Etau[lt] for lt in lead_times],axis=0)**0.5
#            
#            ''' Make the forecast. '''
#            times_0_val = times_0_train[ik*nval:(ik+1)*nval]
#            times_lt_val = np.intersect1d(self.times,times_0_val+timedelta(days=max(lead_times)))
#            times_0_val = times_lt_val-timedelta(days=max(lead_times))
#            
#            tau0_val = np.array([d for t,d in zip(times_0_train,tau0_data) if t in times_0_val])
#            fcst = model.forecast(tau0_val,lead_time=lead_times)
#            fcst = np.mean(fcst,axis=0)
#            
#            print(min(times_lt_val),max(times_lt_val))
#            for i,t in enumerate(times_lt_val):
#                all_val_fcsts[t] = fcst[i]
#                all_val_spread[t] = spread
#            
#        return all_val_fcsts,all_val_spread


#    def __init__(self,eofobj=None,varobj=None,ds=None,fcst_pc=None,obs_pc=None,fcst_map=None,obs_map=None,model=None,tau=None,):
#        self.eofobj = eofobj
#        self.varobj = varobj
#        if ds is None and varobj is not None:
#            self.ds = varobj.ds
#        elif ds is None and varobj is None:
#            self.ds = None
#        else:
#            self.ds = ds
#        self.fcst_pc = fcst_pc
#        self.obs_pc = obs_pc
#        self.fcst = fcst_map
#        self.obs = obs_map
#        self.model = model
#        self.tau = tau
#        
#        if obs_map is None and obs_pc is not None:
#            self.obs = eof_reconstruct(eofobj,obs_pc)
#        if fcst_map is None and fcst_pc is not None:
#            self.fcst = eof_reconstruct(eofobj,fcst_pc)
#        if not isinstance(self.obs,np.ndarray):
#            self.obs = np.array(self.obs)
#        if not isinstance(self.fcst,np.ndarray):
#            self.fcst = np.array(self.fcst)
#            
#    def rho_inf_time(self,varpci):
#        C0 = np.matrix(self.model.C0)
#        Gtau = np.matrix(expm(self.model.L*self.tau))
#        
#        Etau = (C0 - Gtau @ C0 @ Gtau.T)[min(varpci):max(varpci),min(varpci):max(varpci)]
#        Ftau = [np.dot(f.T,f) for f in self.fcst_pc[:,min(varpci):max(varpci)]]
#        
#        S2 = [np.array(f.trace() / Etau.trace()) for f in Ftau]
#        
#        rho_inf = [s2 * ((s2+1)*s2)**-.5 for s2 in S2]
#        
#        return rho_inf
#        
#    def rho_inf_space(self,varpci):
#        C0 = np.matrix(self.model.C0)
#        Gtau = np.matrix(expm(self.model.L*self.tau))
#        eof = eofobj.eof_dict['eof'][:max(varpci)-min(varpci)]
#        
#        Etau = (C0 - Gtau @ C0 @ Gtau.T)[min(varpci):max(varpci),min(varpci):max(varpci)]
#        Etau = np.matrix(eof).T @ Etau @ np.matrix(eof)
#        Ftau = [np.dot(f.T,f) for f in self.fcst_pc[:,min(varpci):max(varpci)]]
#        Ftau = np.matrix(eof).T @ Ftau @ np.matrix(eof)
#        
#        S2 = [np.array(Ftau[i,i]/Etau[i,i]) for i in range(len(Ftau))]
#        
#        rho_inf = np.array([s2 * ((s2+1)*s2)**-.5 for s2 in S2])
#        rho_inf = rho_inf.reshape(len(self.ds['lat']),len(self.ds['lon']))
#        
#        return rho_inf
#    
#
#    def calc_lac(self):
#        r"""
#        Method to calculate the Local Anomaly Correlation (LAC).  Uses numexpr
#        for speed over larger datasets.
#    
#        Note: If necessary (memory concerns) in the future, the numexpr statements
#        can be extended to use pytable arrays.  Would need to provide means to
#        function, as summing over the dataset is still very slow it seems.
#    
#        Parameters
#        ----------
#        fcast : ndarray
#            Time series of forecast data. M x N where M is the temporal dimension.
#        obs : ndarray
#            Time series of observations. M x N
#    
#        Returns
#        -------
#        lac : ndarray
#            Local anomaly corellations for all locations over the time range.
#        """
#        
#        # Calculate means of data
#        f_mean = self.fcst.mean(axis=0)
#        o_mean = self.obs.mean(axis=0)
#        f_anom = self.fcst - f_mean
#        o_anom = self.obs - o_mean
#    
#        # Calculate covariance between time series at each gridpoint
#        cov = (f_anom * o_anom).sum(axis=0)
#    
#        # Calculate standardization terms
#        f_var = (f_anom**2).sum(axis=0)
#        o_var = (o_anom**2).sum(axis=0)
#        f_std = np.sqrt(f_var)
#        o_std = np.sqrt(o_var)
#    
#        std = f_std * o_std
#        lac = cov / std
#    
#        return lac
#    
#    def get_apc(self,latbounds=None,lonbounds=None):
#        
#        
#        if latbounds is not None or lonbounds is not None:
#            lat = self.ds['lat']
#            lon = self.ds['lon']
#            lons,lats = np.meshgrid(lon,lat)
#            if latbounds is None:
#                latbounds = (min(lats),max(lats))
#            if lonbounds is None:
#                lonbounds = (min(lons),max(lats))
#            if min(lonbounds)<0:
#                lons_shift = lons.copy()
#                lons_shift[lons>180] = lons[lons>180]-360
#                lons = lons_shift
#            domain = np.where((lats>=min(latbounds)) & (lats<=max(latbounds)) & \
#                              (lons>=min(lonbounds)) & (lons<=max(lonbounds)))
#            fcst = np.array([f[domain] for f in self.fcst])
#            obs = np.array([o[domain] for o in self.obs])
#        else:
#            fcst = self.fcst.reshape([self.fcst.shape[0],np.product(self.fcst.shape[1:])])
#            obs = self.obs.reshape([self.obs.shape[0],np.product(self.obs.shape[1:])])
#        
#        num = np.sum(fcst * obs,axis=1)
#        den = np.sqrt(np.sum(fcst**2,axis=1)*np.sum(obs**2,axis=1))
#        apc = num / den
#        
#        return apc
#    
#    def calc_mse(self):
#        r"""
#        Calculate mean square error
#        """
#        sq_err = (self.obs - self.fcst)**2
#        mse = sq_err.mean(axis=0)
#        return mse
#    
#    
#    def calc_ce(self):
#        r"""
#        Calculate standardized error
#        """
#        sq_err = (self.obs - self.fcst)**2
#        obs_mean = self.obs.mean(axis=0)
#        obs_var = (self.obs - obs_mean)**2
#        ce = 1 - (sq_err.sum(axis=0) / obs_var.sum(axis=0))
#        return ce
#    
#    
#    def calc_n_eff(self,data1, data2=None):
#        r"""
#        Calculate the effective degrees of freedom for data using lag-1
#        autocorrelation.
#    
#        Parameters
#        ----------
#        data1 : ndarray
#            Dataset to calculate effective degrees of freedom for.  Assumes
#            first dimension is the temporal dimension.
#        data2 : ndarray, optional
#            A second dataset to calculate the effective degrees of freedom
#            for covariances/correlations etc.
#    
#        Returns
#        -------
#        n_eff : ndarray
#            Effective degrees of freedom for input data.
#        """
#    
#        if data2 is not None:
#            assert data1.shape == data2.shape,\
#                'Data must have have same shape for combined n_eff calculation'
#    
#        # Lag-1 autocorrelation
#        r1 = self.calc_lac(data1[0:-1], data1[1:])
#        n = len(data1)
#    
#        if data2 is not None:
#            r2 = self.calc_lac(data2[0:-1], data2[1:])
#            n_eff = n*((1 - r1*r2)/(1+r1*r2))
#        else:
#            n_eff = n*((1-r1)/(1+r1))
#    
#        return n_eff
#    
#    # TODO: Implement correct significance testing
#    def calc_corr_signif(self, corr=None):
#        r"""
#        Calculate local anomaly correlation along with 95% significance.
#        """
#        assert(self.fcst.shape == self.obs.shape)
#    
#        corr_neff = self.calc_n_eff()
#        if corr is None:
#            corr = self.calc_lac()
#    
#        signif = np.empty_like(corr, dtype=np.bool)
#    
#        if True in (abs(corr) < 0.5):
#            g_idx = np.where(abs(corr) < 0.5)
#            gen_2std = 2./np.sqrt(corr_neff[g_idx])
#            signif[g_idx] = (abs(corr[g_idx]) - gen_2std) > 0
#    
#        if True in (abs(corr) >= 0.5):
#            z_idx = np.where(abs(corr) >= 0.5)
#            z = 1./2 * np.log((1 + corr[z_idx]) / (1 - corr[z_idx]))
#            z_2std = 2. / np.sqrt(corr_neff[z_idx] - 3)
#            signif[z_idx] = (abs(z) - z_2std) > 0
#    
#        # if True in ((corr_neff <= 3) & (abs(corr) >= 0.5)):
#        #     assert(False) # I have to figure out how to implement T_Test
#        #     trow = np.where((corr_neff <= 20) & (corr >= 0.5))
#        signif[corr_neff <= 3] = False
#    
#        return signif, corr
       
