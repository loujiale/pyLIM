r"""
Driver for LIM

Sam Lillo
"""

import pickle
import copy
import os
import warnings
import imp
import collections
import xarray as xr
xr.set_options(file_cache_maxsize=5)
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gc

from .model import Model
from .verif import Verif
from .plot import PlotMap
from .dataset import varDataset,eofDataset
from .tools import *

class Driver:
    
    def __init__(self):
        self.variables = {}
        self.eofobjs = {}
        

    def get_variables(self,info,read_path=None,save_path=None,save_netcdf_path=None,segmentby=None):
        
        r"""
        Load data into variable objects and save to pickle files.
        
        Parameters
        ----------
        
        read : bool
            If True, reads previously saved variable objects, else compiles data from path.
        save_netcdf_path : str
            If a str, saves netcdf files of variable with running-mean anomalies and climatology to specified directory.
            Default is None, which means netcdf files will not be written.
        segmentby : str
            If save_netcdf_path is not None, segmentby specifies how to chunk the netcdf files. Options are "all" or "year". Default is "all".

        """
        
        def add_var(key,vardict):
            self.variables[key] = varDataset(key,*self.use_vars[key][:-1],**self.use_vars[key][-1])
        
        if isinstance(info,str):
            configFile = info
            #parse configFile string
            tmp = configFile.split('/')
            fname = tmp[-1].split('.')[0]
            fpath = '/'.join(tmp[:-1])
        
            #import configFile
            fp, path, des = imp.find_module(fname,[fpath])
            namelist = imp.load_module(configFile, fp, path, des)
        
            #save variables from configFile to driver object
            self.__dict__.update(namelist.__dict__)
            fp.close()
        
        elif isinstance(info,dict):
            self.use_vars = info        

        if read_path is not None:
            for name in self.use_vars.keys():
                print(f'reading {name}')
                self.variables[name] = pickle.load( open( os.path.join(read_path,f'{name}.p'), "rb" ) )
        else:
            # Create dataset objects for each variable.
            for name in self.use_vars.keys():
                self.variables[name] = varDataset(name,*self.use_vars[name][:-1],**self.use_vars[name][-1])
                if save_path is not None:
                    pickle.dump(self.variables[name], open( os.path.join(save_path,f'{name}.p'), "wb" ) )

        if save_netcdf_path is not None:
            # Save netcdf files for each variable.
            for name in self.use_vars.keys():
                self.variables[name].save_to_netcdf(save_netcdf_path,segmentby)
        
        
    def get_eofs(self,eof_list,datebounds=None,read_path=None,save_path=None,save_netcdf_path=None):
        
        r"""
        Load data into EOF objects and save to pickle files.
        
        Parameters
        ----------

        eof_list : list
            List of strings, or tuples of strings, corresponding to the keys in self.variables.
            A tuple indicates a multivariate EOF.
        read : bool
            If True, reads previously saved eof objects, else compiles data from path.
        datebounds : tuple
            (start date, end date) as string 'm/d'
        save_netcdf_path : str
            If a str, saves netcdf files of EOF object with regridded EOF maps, PCs, and variance explained.
            Default is None, which means netcdf files will not be written.
            
        """

        if read_path is not None:
            for key in eof_list:
                print(f'reading {key} EOF')
                self.eofobjs[key] = pickle.load( open( os.path.join(read_path,'EOF_'+'+'.join(listify(key))+'.p') , "rb" ) )

        else:
            if datebounds is None:
                tmpobjs = {k:v for k,v in self.variables.items()}
            else:
                tmpobjs = {k:v.subset(datebounds) for k,v in self.variables.items()}

            #Calculate EOFs of the subset variable objects
            for key in eof_list:
                print(key)
                eofobj = eofDataset([tmpobjs[k] for k in listify(key)])
                self.eofobjs[key] = eofobj
            
                if save_path is not None:
                    eofobj.save_to_pickle(save_path)
        
        if save_netcdf_path is not None:
            for key in eof_list:
                eofobj = self.eofobjs[key]
                eofobj.save_to_netcdf(save_netcdf_path)

                
    def get_model(self,eof_trunc, tau1n, datebounds=None, yearbounds=None,\
                  load_file=None, save_file=None, save_to_netcdf=None):
        
        r"""
        Get the LIM model object.
        
        Parameters
        ----------
        eof_trunc : dict
            Dictionary with keys corresponding to EOF names and 
            values corresponding to truncation.
        tau1n : int
            Integer for training lag, in number of time steps.
        load_file : str
            Filename for pickle file containing model to load.
            Default is None, in which case a new model is trained.
        save_file : str
            Filename of pickle file to save new model to.
            Default is None, in which case the model is not saved.
        
        Returns
        -------
        model : model object
            Object of trained model
        """

        self.eof_trunc = eof_trunc
        self.tau1n = tau1n

        if load_file is None:

            varobjs = [v for name in self.eof_trunc.keys() for v in self.eofobjs[name].varobjs]
            
            times = varobjs[0].time
            
            p = {}
            for name in self.eof_trunc.keys():
                time = self.eofobjs[name].varobjs[0].time
                pcs = self.eofobjs[name].pc[:,:eof_trunc[name]]
                p[name] = [pc for t,pc in zip(time,pcs) if t in times]
            
            all_data = np.concatenate([p[name] for name in p.keys()],axis=1)
             
            # Get times for tau0 and tau1 data by finding the intersection of all 
            # times and times + tau1n days
            times1 = np.intersect1d(times,times+timedelta(days = tau1n))
            times0 = times1-timedelta(days = tau1n)
            
            # Get tau0 and tau1 data by taking all data and all times and matching 
            # with the corresponding times for tau0 and tau1
            tau0_data = np.array([d for d,t in zip(all_data,times) if t in times0])
            tau1_data = np.array([d for d,t in zip(all_data,times) if t in times1])
                
            # Train the model
            self.model = Model(tau0_data,tau1_data,tau1n=tau1n)
            
        else:
            # Read in model
            self.model = pickle.load( open(load_file, "rb" ) )
            print(f'LIM read from {load_file} and contained in self.model')

        if save_file is not None:
            pickle.dump(self.model, open(save_file,'wb'))
        print(f'LIM trained and contained in self.model')

        if save_to_netcdf is not None:
            # Save model attributes to netcdf file
            if isinstance(save_to_netcdf,str):
                self.model.save_to_netcdf(save_to_netcdf)
            else:
                raise TypeError("save_to_netcdf must be a string containing path and filename for netcdf file.")
    

    def cross_validation(self,num_folds=10,lead_times=np.arange(1,29),eof_trunc=None,\
                         average=False,fullVariance=False,save_netcdf_path=None,segmentby='day',\
                        prop={}):

        r"""
        Cross validation.
        
        Parameters
        ----------
        num_folds : int
            Number of times data is subset and model is trained. Default is 10,
            Which means each iteration trains on 90% of data, forecasts run on 10% of the data.
        lead_times : list, tuple, ndarray
            Lead times (in days) to integrate model for output. Default is first 28 lead times.
        average : bool
            Whether to average the forecasts at the specified lead times.
        fullVariance : bool
            Whether to add untruncated noise back in for spread calculation (default to False).
        save_netcdf_path : str
            Path to save netcdf files containing forecast and spread.
        segmentby : str
            If save_netcdf_path is not None, segmentby specifies how to chunk the netcdf files. 
            Options are "day", "month", "year". Default is "day".
            
        Returns
        -------
        model_F : dict
            Dictionary of model forecast output, with keys corresponding to valid time
        model_E : dict
            Dictionary of model error output, with keys corresponding to valid time
        """        
        
        
        #run cross-validation with specified state space 
        self.lead_times = lead_times
        self.model_F = {}
        self.model_E = {}
        
        V = Verif(self.eofobjs,self.eof_trunc)
        V.kfoldval(lead_times = lead_times, k = num_folds, tau1n = self.tau1n, average=average, prop=prop)
        
        for t in V.fcsts.keys():
            self.model_F[t] = V.fcsts[t]
            self.model_E[t] = V.variance[t]
        
        if save_netcdf_path is not None:
            
            print('Saving cross-validated model forecasts to netCDF files')
            
            if not os.path.isdir(save_netcdf_path):
                os.mkdir(save_netcdf_path)
            
            # Save model attributes to netcdf file
            if not isinstance(save_netcdf_path,str):
                raise TypeError("save_to_netcdf must be a string containing path and filename for netcdf file.")
                
            else:

                model_F = collections.OrderedDict(sorted(self.model_F.items()))
                model_E = collections.OrderedDict(sorted(self.model_E.items()))
                
                data_seg = {}
                if segmentby == 'day':
                    for (K,F),(_,E) in zip(model_F.items(),model_E.items()):
                        print(K)
                        # F and E are a 2darray, lt x pc
                        # pc are concatenated PCs from each EOF
                        
                        eof_lim = self.eof_trunc
                                 
                        # use pc_to_grid to get dictionaries of anomaly forecasts and spread
                        Fmap,Emap = self.pc_to_grid(F=F,E=E,regrid=True,fullVariance=fullVariance)

                        for varname,Vmap in Fmap.items():
                            varobj = self.variables[varname]
                            
                            lonvec = (varobj.longrid[~np.isnan(Vmap).all(axis=1)][:,~np.isnan(Vmap).all(axis=0)])[0,:]
                            latvec = (varobj.latgrid[~np.isnan(Vmap).all(axis=1)][:,~np.isnan(Vmap).all(axis=0)])[:,0]
                            Vmap_shrink = Vmap[~np.isnan(Vmap).all(axis=1)][:,~np.isnan(Vmap).all(axis=0)]
                                                            
                            coords = {"init_time": {'dims':('init_time',),
                                                 'data':np.array([np.double((K-dt(1800,1,1)).total_seconds()/3600)]),
                                                 'attrs':{'long_name':'Initial time',
                                                           'units':'hours since 1800-01-01 00:00:0.0'}},
                                    "lead_time": {'dims':('time',), 
                                                 'data':self.lead_times,
                                                 'attrs':{'long_name':'lead time','units':'days'}},
                                    "time": {'dims':('time',), 
                                                 'data':np.array([K+timedelta(days=int(lt)) for lt in self.lead_times]),
                                                 'attrs':{'long_name':'valid time',}},
                                    "lon": {'dims':("lon",), 
                                              'data':varobj.longrid[0,:],
                                              'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                                    "lat": {'dims':("lat",), 
                                              'data':varobj.latgrid[:,0],
                                              'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                                    }

                            vardict = {f"{varname}": {'dims':("time","lat","lon"),
                                                           'data':Vmap_shrink,
                                                           'attrs':varobj.attrs},
                                       f"{varname}_spread": {'dims':("time","lat","lon"),
                                                           'data':Emap[varname],
                                                           'attrs':varobj.attrs}
                                        }

                            save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{K:%Y%m%d}.nc'))

                        _=gc.collect()

                elif segmentby == 'month':
                    data_seg = {}
                    init_times = []
                    MONTH = min(model_F.keys()).month
                    YEAR = min(model_F.keys()).year
                    for (K,F),(_,E) in zip(model_F.items(),model_E.items()):
                        print(K)
                        # F and E are a 2darray, lt x pc
                        # pc are concatenated PCs from each EOF
                                                
                        Fmap,Emap = self.pc_to_grid(F=F,E=E,regrid=True,fullVariance=fullVariance)
                        
                        if len(data_seg)==0:
                            #very first init_time
                            data_seg = {k:v[None,:,:,:] for k,v in Fmap.items()}
                            data_seg_E = {k:v[None,:,:,:] for k,v in Emap.items()}
                            init_times.append(K)
                        elif K.month == MONTH and K.year == YEAR:
                            data_seg = {k:np.concatenate([data_seg[k],v[None,:,:,:]],axis=0) for k,v in Fmap.items()}
                            data_seg_E = {k:np.concatenate([data_seg_E[k],v[None,:,:,:]],axis=0) for k,v in Emap.items()}
                            init_times.append(K)
                        else:

                            for varname,Vmap in data_seg.items():
                                varobj = self.variables[varname]
                                Emap = data_seg_E[varname]
                                
                                testarray = Vmap[0][0]
                                lonvec = (varobj.longrid[~np.isnan(testarray).all(axis=1)][:,~np.isnan(testarray).all(axis=0)])[0,:]
                                latvec = (varobj.latgrid[~np.isnan(testarray).all(axis=1)][:,~np.isnan(testarray).all(axis=0)])[:,0]
                                Vmap_shrink = np.array([[vv[~np.isnan(vv).all(axis=1)][:,~np.isnan(vv).all(axis=0)] \
                                                        for vv in v] for v in Vmap])
                            
                                coords = {"time": {'dims':('time',),
                                                     'data':np.array(init_times),
                                                     'attrs':{'long_name':'initial time',}},
                                        "lead_time": {'dims':('lead_time',), 
                                                     'data':list(self.lead_times),
                                                     'attrs':{'long_name':'lead time','units':'days'}},
                                        "lon": {'dims':("lon",),
                                                  'data':lonvec,
                                                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                                        "lat": {'dims':("lat",),
                                                  'data':latvec,
                                                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                                                }

                                vardict = {f"{varname}": {'dims':("time","lead_time","lat","lon"),
                                                               'data':Vmap_shrink,
                                                               'attrs':varobj.attrs},
                                            }
                                           #f"{varname}_spread": {'dims':("time","lead_time","lat","lon"),
                                           #                    'data':Emap[varname],
                                           #                    'attrs':varobj.attrs}                                       
                                save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{K-timedelta(days=1):%Y%m}.nc'))
                            
                            #Update for new year
                            data_seg = {k:v[None,:,:,:] for k,v in Fmap.items()}
                            init_times = [K]
                            MONTH = K.month
                            YEAR = K.year
                            
                            _=gc.collect()

                elif segmentby == 'year':
                    data_seg = {}
                    init_times = []
                    YEAR = min(model_F.keys()).year
                    for (K,F),(_,E) in zip(model_F.items(),model_E.items()):
                        print(K)
                        # F and E are a 2darray, lt x pc
                        # pc are concatenated PCs from each EOF
                        
                        Fmap,Emap = self.pc_to_grid(F=F,E=E,regrid=True,fullVariance=fullVariance)
                        
                        if len(data_seg)==0:
                            #very first init_time
                            data_seg = {k:v[None,:,:,:] for k,v in Fmap.items()}
                            data_seg_E = {k:v[None,:,:,:] for k,v in Emap.items()}
                            init_times.append(K)
                        elif K.year == YEAR:
                            data_seg = {k:np.concatenate([data_seg[k],v[None,:,:,:]],axis=0) for k,v in Fmap.items()}
                            data_seg_E = {k:np.concatenate([data_seg_E[k],v[None,:,:,:]],axis=0) for k,v in Emap.items()}
                            init_times.append(K)
                        else:

                            for varname,Vmap in data_seg.items():
                                varobj = self.variables[varname]
                                Emap = data_seg_E[varname]
                                
                                coords = {"time": {'dims':('time',),
                                                     'data':np.array(init_times),
                                                     'attrs':{'long_name':'initial time',}},
                                        "lead_time": {'dims':('lead_time',), 
                                                     'data':self.lead_times,
                                                     'attrs':{'long_name':'lead time','units':'days'}},
                                        "lon": {'dims':("lon",), 
                                                  'data':varobj.longrid[0,:],
                                                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                                        "lat": {'dims':("lat",), 
                                                  'data':varobj.latgrid[:,0],
                                                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                                        }
                                
                                vardict = {f"{varname}": {'dims':("time","lead_time","lat","lon"),
                                                               'data':Vmap,
                                                               'attrs':varobj.attrs},
                                           f"{varname}_spread": {'dims':("time","lead_time","lat","lon"),
                                                               'data':Emap[varname],
                                                               'attrs':varobj.attrs}
                                            }                               
                                save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{YEAR}.nc'))
                            
                            #Update for new year
                            data_seg = {k:v[None,:,:,:] for k,v in Fmap.items()}
                            init_times = [K]
                            YEAR = K.year
                            
                            _=gc.collect()
                        

    def pc_to_grid(self,F=None,E=None,regrid=False,varname=None,fullVariance=False):

        r"""
        Convect PC state space vector to variable grid space.
        
        Parameters
        ----------
        F : ndarray
            Array with one or two dimensions. LAST axis must be the PC vector.
        E : ndarray
            Array with one or two dimensions. Error covariance matrix. If None, ignores.
            LAST TWO axes must be length of PC vector.
        regrid : bool
            Whether to regrid to lat/lon gridded map.
        varname : str
            If just one variable output is desired. 
            Default is None, in which case all variables are output in a dictionary.
            
        Returns
        -------
        Fmap : dict
            If F was provided. Dictionary with keys as variable names, and values are ndarrays of 
            reconstructed gridded space.
        Emap : dict
            If E was provided. Dictionary with keys as variable names, and values are ndarrays of 
            reconstructed gridded space.
        """
        
        eof_lim = self.eof_trunc
        
        Fmap, Emap = {}, {}
        
        if F is not None:
            F = np.asarray(F)
            #Reshape to (times,pcs)
            Pshape = F.shape
            if len(Pshape)==1:
                F = F[np.newaxis,:]
                Pshape = F.shape
            else:
                F = F.reshape((np.product(Pshape[:-1]),Pshape[-1]))
            i0 = 0
            for eofname,plen in eof_lim.items():
                if varname in listify(eofname) or varname is None:
                    recon = self.eofobjs[eofname].reconstruct(F[:,i0:i0+plen])
                    for vname,v in recon.items():
                        if regrid:
                            varobj = self.variables[vname]
                            v2 = np.array(list(map(varobj.regrid,v)))
                            Fmap[vname] = v2.reshape(Pshape[:-1]+v2.shape[-2:]).squeeze()
                        else:
                            Fmap[vname] = v.reshape((*Pshape[:-1],v.shape[-1])).squeeze()
                i0 += plen

        if E is not None:
            E = np.asarray(E)
            #Reshape to (times,pcs,pcs)
            Pshape = E.shape
            if len(Pshape)==2:
                E = E[np.newaxis,:,:]
                Pshape = E.shape
            else:
                E = E.reshape((np.product(Pshape[:-2]),*Pshape[-2:]))
            i0 = 0
            for eofname,plen in eof_lim.items():
                if varname in listify(eofname) or varname is None:
                    eofobj = self.eofobjs[eofname]
                    recon = eofobj.reconstruct(E[:,i0:i0+plen,i0:i0+plen],order=2)
                    if fullVariance:
                        truncStdev = {k:np.std(v,axis=0) for k,v in eofobj.reconstruct(eofobj.pc,num_eofs=plen).items()}
                        fullStdev = {v.varlabel:np.std(v.anomaly,axis=0) for v in eofobj.varobjs}
                        varScaling = {k:fullStdev[k]/truncStdev[k] for k in fullStdev.keys()}
                    else:
                        varScaling = {v.varlabel:np.ones(v.lon.shape) for v in eofobj.varobjs}
                    
                    for vname,v in recon.items():
                        if regrid:
                            varobj = self.variables[vname]
                            v = np.array(list(map(varobj.regrid,v*varScaling[vname])))
                            Emap[vname] = v.reshape(Pshape[:-2]+v.shape[-2:]).squeeze()
                        else:
                            Emap[vname] = v.reshape((*Pshape[:-2],v.shape[-1])).squeeze()*varScaling[vname]
                i0 += plen

        if E is None and F is None:
            print('both F and E inputs were None')
            return None
        
        if varname is None:
            out = tuple([x for x in (Fmap,Emap) if len(x)>0])
        else:
            out = tuple([x[varname] for x in (Fmap,Emap) if len(x)>0])

        if len(out)>1:
            return out
        else:
            return out[0]


    def get_rhoinf_time(self,varname,fcst=None,spread=None,latbounds=None,lonbounds=None,region=None):
        r"""
        Get timeseries of rho infinity.
        
        Parameters
        ----------
        varname : str
            Name of variable of interest.
        fcst : ndarray
            Forecast array with shape (time, space) or (space). Default is None, using the self.model_F output.
        spread : ndarray
            Spread array with shape (time, space) or (space). Default is None, using the self.model_E output.
        latbounds : tuple
            latitude bounds to specify spatial subdomain. Default is None, using full domain.
        lonbounds : tuple
            longitude bounds to specify spatial subdomain. Default is None, using full domain.
        region : str
            Using regionmask, can specify country to define the subdomain. Default is None.
            
        Returns
        -------
        Fmap : dict
            If F was provided. Dictionary with keys as variable names, and values are ndarrays of 
            reconstructed gridded space.
        Emap : dict
            If E was provided. Dictionary with keys as variable names, and values are ndarrays of 
            reconstructed gridded space.
        """
        
        varobj = self.variables[varname]
        lats = varobj.lat
        lons = varobj.lon
        if latbounds is None:
            latbounds = (np.amin(lats),np.amax(lats))
        if lonbounds is None:
            lonbounds = (np.amin(lons),np.amax(lons))
        if min(lonbounds)<0:
            lons_shift = lons.copy()
            lons_shift[lons>180] = lons[lons>180]-360
            lons = lons_shift
        domain = np.where((lats>=min(latbounds)) & (lats<=max(latbounds)) & \
                          (lons>=min(lonbounds)) & (lons<=max(lonbounds)))
    
        if fcst is None and spread is None:
            fcst,spread = self.pc_to_grid(F=list(self.model_F.values()),E=list(self.model_E.values()),\
                                  varname=varname,regrid=False)
        if len(np.array(fcst).shape)<2:
            fcst = np.array(fcst).reshape(1,len(fcst))
            spread = np.array(spread).reshape(1,len(spread))
        f2 = [f[domain]**2 for f in fcst]
        e2 = [e[domain]**2 for e in spread]
        
        S2 = [np.array(np.nansum(f)/np.nansum(e)) for f,e in zip(f2,e2)]
        rho_inf = np.array([s2 * ((s2+1)*s2)**-.5 for s2 in S2])
        if len(rho_inf)<2:
            rho_inf = rho_inf[0]
        return rho_inf
                    

    def plot_acc(self,varname,lead_time,year_range=None,date_range=None,rhoinf_prop={},prop={},return_array=False):
        r"""
        Calculates and plots map of anomaly correlation coefficient.
        
        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in the LIM.
        lead_time : ndarray / list / tuple
            Lead times for forecast, WILL be averaged.
        year_range : list / tuple
            Years between which (inclusive) to assess forecasts.
        date_range : list / tuple
            Dates between which (inclusive) to assess forecasts. In form "m/d".
        rhoinf_prop : dict
            Keywords include:
            * ptile_range - percentile range (min,max) of rho infinity to select forecasts.
            * latbounds - latitude bounds for rhoinf calculation. Default is full domain.
            * lonbounds - longitude bounds for rhoinf calculation. Default is full domain.
            * region - country name to specify domain for rhoinf calculation.

        save_to_path : str
            Path to save map figures to.
            Default is None, in which case figures will not be saved.
            
        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """

        if year_range is None:
            year_range = (min(self.model_F.keys()).year,max(self.model_F.keys()).year)
        if date_range is None:
            date_range = ('1/1','12/31')
        
        def lt_avg(v):
            if isinstance(lead_time,(int,float)):
                ilt = [i for i,j in enumerate(self.lead_times) if j==lead_time]
                return v[ilt].squeeze()
            else:
                ilt = [i for i,j in enumerate(self.lead_times) if j in lead_time]
                return np.mean(v[ilt],axis=0)
        
        model_F = {k:lt_avg(v) for k,v in self.model_F.items() if date_range_test(k,date_range,year_range)}
        model_E = {k:lt_avg(v) for k,v in self.model_E.items() if date_range_test(k,date_range,year_range)}

        fcst,spread = self.pc_to_grid(F=list(model_F.values()),E=list(model_E.values()),\
                              varname=varname,regrid=False)

        rhoinf_prop_default = {'ptile_range':None,'latbounds':None,'lonbounds':None,'region':None}
        rhoinf_prop = add_prop(rhoinf_prop,rhoinf_prop_default)
        rhoinf_ptile_range = rhoinf_prop['ptile_range']
        if rhoinf_ptile_range is not None:
            # do if there is a rhoinf ptile range given
            out = self.get_rhoinf_time(varname,fcst=fcst,spread=spread,latbounds=rhoinf_prop['latbounds'],\
                                 lonbounds=rhoinf_prop['lonbounds'],\
                                 region=rhoinf_prop['region'])
            rhoinf = {k:r for k,r in zip(model_F.keys(),out) if date_range_test(k,date_range,year_range)}
            rho_limits = [np.percentile(list(rhoinf.values()),p) for p in rhoinf_ptile_range]
            rhoflag = {k:(bool(r>=min(rho_limits)) & (r<=max(rho_limits))) for k,r in rhoinf.items()}
        # otherwise give all True to a date dictionary
        else:
            rhoflag = {k:True for k in model_F.keys()}

        F_verif = {k:v for k,v in zip(list(model_F.keys()),fcst) if rhoflag[k]}
        S_verif = {k:v for k,v in zip(list(model_E.keys()),spread) if rhoflag[k]}
        
        # Get observed values for verification
        varobj = self.variables[varname]
        if len(listify(lead_time))==1:
            O_time_init = np.array(varobj.time)-timedelta(days=lead_time)
            O_data = varobj.anomaly
        # elif multiple lead-times are listed for averaging, must average verification
        elif len(listify(lead_time))>1:
            xlt = max(lead_time)
            nlt = min(lead_time)
            O_time_init = (np.array(varobj.time)-timedelta(days=xlt))[xlt-nlt:]
            O_data = np.mean([varobj.anomaly[lt-nlt:len(varobj.anomaly)+lt-xlt]\
                                                  for lt in lead_time],axis=0)
            
        # Make dictionary for observations, INIT time : VERIF data
        O_verif = {t:o for t,o in zip(O_time_init,O_data) if t in F_verif.keys()}
        F_verif = {t:f for t,f in F_verif.items() if t in O_verif.keys()}
        
        LAC = calc_lac(list(F_verif.values()),list(O_verif.values()))
        
        #Set default properties
        default_cmap = {0:'violet',
                      .22:'mediumblue',
                      .35:'lightskyblue',
                      .44:'w',
                      .56:'w',
                      .65:'gold',
                      .78:'firebrick',
                      1:'violet'}
        default_title = f'{varname} {lead_time}-day ACC | {min(year_range)} – {max(year_range)} | {" – ".join(date_range)}'
        default_prop={'cmap':default_cmap,'levels':(-1*max(abs(LAC)),max(abs(LAC))),'title':default_title,\
                      'figsize':(10,6),'dpi':150,'drawcountries':True,'drawstates':True}
        prop = add_prop(prop,default_prop)
        prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])
        ax=varobj.plot_map(LAC,prop=prop)
        ax.set_title(prop['title'])
        
        if return_array:
        	return LAC

    def prep_data(self):
        
        r"""
        Compile new data, interpolate to same grid as LIM, and convert into PCs.
        If it has already been run before within the defined self object, 
        self.RT_VARS will be the same, and self.RT_PC can be updated for different LIMs with same vars,
        without having to process variables again.
        
        Parameters
        ----------

        Returns
        -------
        self.RT_VARS : dict
            dictionary with keys = variable names + "time", and values = data,
            processed to same time res and space res & domain as training variables.
        self.RT_PC : dict
            dictionary with keys = eof names, and values = truncated PCs, both according to specified limkey
        """

        if 'time' not in self.RT_VARS.keys():
            # read and process realtime data
            for name in self.RT_VARS.keys():
                self.RT_VARS[name].update(self.variables[name].process_new_data(self.RT_VARS[name]['filename'],self.RT_VARS[name]['varname']))
    
            # find all common times
            p = [v['time'] for v in self.RT_VARS.values()]
            common_times = set(p[0]).intersection(*p)
            for name,v in self.RT_VARS.items():
                ikeep = np.array(sorted([list(v['time']).index(j) for j in common_times]))
                self.RT_VARS[name]['anomaly'] = v['anomaly'][ikeep]
                self.RT_VARS[name]['time'] = v['time'][ikeep]
            self.RT_VARS['time'] = v['time'][ikeep]

        # get PCs
        eof_lim = self.eof_trunc
        self.RT_PC = {eofname:eofobj.get_pc({n:self.RT_VARS[n] for n in listify(eofname)},trunc=eof_lim[eofname]) \
                      for eofname,eofobj in self.eofobjs.items()}


    def run_forecast(self,t_init=None,lead_times=np.arange(1,29),fullVariance=False,save_netcdf_path=None):
        
        r"""
        Run forecasts initialized with self.RT_PC, with EOFs / model specified by self.RTLIMKEY
        
        Parameters
        ----------
        t_init : datetime object
            Date of initialization for the forecast.
            Default is maximum date available in self.RT_VARS['time']
        lead_times : list, tuple, or ndarray
            Lead_times in increment of data to integrate model forward. 
            Default is first 28 leads.
        save_netcdf_path : str
            Path to save netcdf files containing forecast and spread. 
            Default is None, which will not write netcdf files.
        
        Returns
        -------
        model_F : dict
            Dictionary of model forecast output, with keys corresponding to init time.
        model_E : dict
            Dictionary of model error covariance output, with keys corresponding to init time.
        """
        
        self.lead_times = [int(i) for i in lead_times]
        
        # Get the (time independent) variance from the model
        C0 = np.matrix(self.model.C0)
        Gtau = {lt:expm(np.matrix(self.model.L)*lt) for lt in lead_times}
        Etau = {lt:(C0 - Gtau[lt] @ C0 @ Gtau[lt].T) for lt in lead_times}
        
        # use LIM specified by prep_realtime_data
        eof_lim = self.eof_trunc
        init_data = np.concatenate([self.RT_PC[name] for name in eof_lim.keys()],axis=1)
        fcst = self.model.forecast(init_data,lead_time=lead_times)
        
        fcst = np.array(fcst).swapaxes(0,1)
        variance = np.array([Etau[lt] for lt in lead_times])
        
        self.model_F = {}
        self.model_E = {}
        for i,t in enumerate(self.RT_VARS['time']):
            self.model_F[t] = fcst[i]
            self.model_E[t] = variance

        if t_init is None:
            t_init = max(self.model_F.keys())
            
        if save_netcdf_path is not None:

            init_times = [t_init+timedelta(days=i-6) for i in range(7)]
            F = [self.model_F[t] for t in init_times]
            E = [self.model_E[t] for t in init_times]
            Fmap,Emap = self.pc_to_grid(F=F,E=E,regrid=True,fullVariance=fullVariance)
            
            for varname in Fmap.keys():
                varobj = self.variables[varname]
                                                
                coords = {"time": {'dims':('time',),
                                     'data':np.array(init_times),
                                     'attrs':{'long_name':'initial time',}},
                        "lead_time": {'dims':('lead_time',), 
                                     'data':self.lead_times,
                                     'attrs':{'long_name':'lead time','units':'days'}},
                        "lon": {'dims':("lon",), 
                                  'data':varobj.longrid[0,:],
                                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                        "lat": {'dims':("lat",), 
                                  'data':varobj.latgrid[:,0],
                                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                        }
                
                vardict = {f"{varname}": {'dims':("time","lead_time","lat","lon"),
                                               'data':Fmap[varname],
                                               'attrs':varobj.attrs},
                           f"{varname}_spread": {'dims':("time","lead_time","lat","lon"),
                                               'data':Emap[varname],
                                               'attrs':varobj.attrs}
                                }
                
                save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{init_times[-1]:%Y%m%d}.nc'))
                            

    def plot_map(self,varname='T2m',t_init=None,lead_times=None,fullVariance=False,save_to_path=None,prop={}):

        r"""
        Plots maps from PCs, using EOFs from self.RTLIMKEY
        
        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in the LIM.
        t_init : datetime object
            Initialization time for forecast.
        lead_times : ndarray / list / tuple
            Lead times for forecast.
        save_to_path : str
            Path to save map figures to.
            Default is None, in which case figures will not be saved.
        
        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """
        
        if t_init is None:
            t_init = max(self.model_F.keys())
        
        if lead_times is not None:
            lead_times = listify(lead_times)
            #check if lead_times is in self.lead_times
            try:
                ilt = np.array([self.lead_times.index(l) for l in lead_times])
                F_PC = np.mean(self.model_F[t_init][ilt],axis=0)
                E_PC = np.mean(self.model_E[t_init][ilt],axis=0)
            except:
                self.run_forecast(lead_times=lead_times)
                F_PC = np.mean(self.model_F[t_init],axis=0)
                E_PC = np.mean(self.model_E[t_init],axis=0)
        else:
            try:
                F_PC = np.mean(self.model_F[t_init],axis=0)
                E_PC = np.mean(self.model_E[t_init],axis=0)
            except:
                self.run_forecast(t_init,lead_times=lead_times)            
                F_PC = np.mean(self.model_F[t_init],axis=0)
                E_PC = np.mean(self.model_E[t_init],axis=0)
        
        LT_lab = '-'.join([str(int(l/7)) for l in lead_times])
        fname_lab = '-'.join([f'{int(l):03}' for l in lead_times])
        
        eof_lim = self.eof_trunc
        pci = np.cumsum([eof_lim[key] for key in eof_lim.keys()])
        vari = list(eof_lim.keys()).index(varname)
        varpci = [pci[vari]-eof_lim[varname],pci[vari]]
        varobj = self.variables[varname]
        
        # Reconstruct map from PCs
        FMAP,EMAP = self.pc_to_grid(F=F_PC,E=E_PC,varname=varname,\
                                    regrid=False,fullVariance=fullVariance)
        
        prop['extend']='both'
        ax = varobj.plot_map(FMAP, prop = prop)
        ax.set_title(f'{varname} Forecast',loc='left',fontweight='bold',fontsize=16)
        ax.set_title(f'Init: {t_init:%a %d %b %Y}\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)
        
        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_file must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}_lt{fname_lab}.png',bbox_inches='tight')
        plt.close()

        bounds = [-np.inf*np.ones(len(FMAP)),np.zeros(len(FMAP)),np.inf*np.ones(len(FMAP))]
        cat_fcst = get_probabilistic_fcst((FMAP,),(EMAP,),bounds)[0]
        probplt = cat_fcst[1]*100
        probplt[probplt<1]=1;probplt[probplt>99]=99

        # make probabilistic forecast map
        prop['levels'] = [-.1]+list(np.arange(5,100,5))+[100.1]
        prop['cbarticks']=np.arange(0,101,10)
        prop['cbarticklabels']=['Below',90,80,70,60,'',60,70,80,90,'Above']
        prop['extend']='neither'
        prop['cbar_label']=None
        ax = varobj.plot_map(probplt, prop = prop)
        #ax = plot_map(cat_fcst[1]*100,cmap = cmap, levels = levels, prop = prop)
        ax.set_title(f'{varname} Probabilistic\nForecast',loc='left',fontweight='bold',fontsize=16)
        ax.set_title(f'Init: {t_init:%a %d %b %Y}\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)
    
        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}-CAT_lt{fname_lab}.png',bbox_inches='tight')
        plt.close()

        
    def plot_timelon(self,varname,lat_bounds,t_init=None,daysback=120,save_to_file=None,prop={}):
        
        r"""
        Plots meridionally averaged data with longitude on x-axis and time on y-axis
        
        Parameters
        ----------
        varname : str
            Name of variable (consistent with use_var dictionary keys) to plot.
        lat_bounds : tuple or list
            Latitude boundaries to average data between.
        t_init : datetime object
            Time of forecast initialization. Default is maximum time in self.model_F.
        daysback : int
            Number of days prior to t_init to plot hovmoller
        save_to_file : str
            Name of file to save figure to. Default is None, which does not save the figure
        """

        #Set default properties
        default_prop={'cmap':'bwr','levels':None,'title':None,'dpi':150,'figsize':(6,10),'cbar_label':None}
        prop = add_prop(prop,default_prop)

        if t_init is None:
            t_init = max(self.RT_VARS['time'])
            
        t1 = t_init-timedelta(days=daysback)
        t2 = t_init
        
        varobj = self.variables[varname]
        
        lat_idx = np.where((varobj.latgrid>=min(lat_bounds)) & (varobj.latgrid<=max(lat_bounds)))[0]
        
#        ytime = [t for t in varobj.time if t>=t1 and t<=t2]
#        varplot = np.array([np.mean(varobj.regrid(x)[min(lat_idx):max(lat_idx)+1,:],axis=0) \
#               for t,x in zip(varobj.time,varobj.anomaly) if t in ytime])
        ytime = [t2+timedelta(days=i+1-daysback) for i in range(daysback)]
        varplot = [np.nanmean(varobj.regrid(x)[min(lat_idx):max(lat_idx)+1,:],axis=0) \
               for t,x in zip(ytime,self.RT_VARS[varname]['anomaly'][-daysback:])]
        xlon = varobj.longrid[0,:]
        
        eof_lim = self.eof_trunc
        varpci = get_varpci(eof_lim,varname)
        eofobj = self.eofobjs[varname]
                
        if self.model_F is None:
                FORECAST = None
        else:
            F = self.pc_to_grid(F=self.model_F[t2],regrid=True,varname=varname)
            Fvar = np.mean(F[:,min(lat_idx):max(lat_idx)+1,:],axis=1)
            
            FORECAST = {}
            FORECAST['dates'] = [t2+timedelta(days=i+1) for i in range(len(Fvar))]
            FORECAST['var'] = Fvar
        
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(varplot)),np.nanmax(abs(varplot)))
        cmap,clevs = get_cmap_levels(prop['cmap'],prop['levels'])
                
        fig = plt.figure(figsize=prop['figsize'],dpi=prop['dpi'])
        ax = plt.subplot()
        if prop['title'] is None:
            prop['title'] = varname
        ax.set_title(f'{prop["title"]} | {lat2str(min(lat_bounds))} – {lat2str(max(lat_bounds))}',fontsize=16)
        _=timelonplot(ax,xlon,ytime,varplot,FORECAST=FORECAST,\
                      cmap=cmap,levels=clevs,cbar_label=prop['cbar_label'])
        
        if save_to_file is None:
            plt.show()
        elif not isinstance(save_to_file,str):
            print('WARNING: save_to_file must be a string indicating the path and name of the file to save the figure to.')
        else:
            plt.savefig(save_to_file,bbox_inches='tight')
            plt.close()
            
            
    def plot_verif(self,varname='T2m',t_init=None,lead_times=None,Fmap=None,Emap=None,
                   prob_thresh=50,regMask=None,save_to_path=None,prop={}):
    
        r"""
        Plots verifying anomalies.
        
        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in self.RT_VARS.
        t_init : datetime object
            Time of forecast initialization. 
            Default is maximum time in self.RT_VARS['time'] minus maximum lead time.
        lead_times : ndarray / list / tuple
            Lead times. Forecasts and obs WILL be averaged for given leads.
        Fmap : ndarray
            Default is None, in which case forecast will be taken from self.model_F[t_init]
        Emap : ndarray
            Default is None, in which case spread will be taken from self.model_F[t_init]
        prob_thresh : int
            Probability threshold (50 to 100) above which verification statistics will be calculated and mapped.
            Default is 50 (no masking).

        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """
        
        #Set default properties
        default_prop={'cmap':None,'levels':None,'title':None,'figsize':(10,6),'dpi':200,
                      'drawcountries':True,'drawstates':True}
        prop = add_prop(prop,default_prop)        
        
        lead_times = listify(lead_times)
        if lead_times is None:
            lead_times = self.lead_times
        if t_init is None:
            t_init = max(self.RT_VARS['time'])-timedelta(days=max(lead_times))
        t_verif = np.array([t_init + timedelta(days=i) for i in lead_times])
        
        itv = np.in1d(self.RT_VARS['time'],t_verif).nonzero()[0]
        #check if all t_verif in RT_VARS, otherwise exit with error
        if len(itv)<len(listify(t_verif)):
            print(f'{t_verif} not in RT_VARS')
            sys.exit(1)

        varobj = self.variables[varname]

        #check if lead_times is in self.lead_times
        if Fmap is None and Emap is None:
            try:
                ilt = np.array([self.lead_times.index(l) for l in lead_times])
                F_PC = np.mean(self.model_F[t_init][ilt],axis=0)
                E_PC = np.mean(self.model_E[t_init][ilt],axis=0)
                outF,outE = self.pc_to_grid(F=F_PC,E=E_PC,regrid=False)
                FMAP,EMAP = outF[varname],outE[varname]
            except:
                print(f'lead times {lead_times} and/or t_init {t_init} not available in model_F.')
        else:
            if len(Fmap.shape)>1:
                FMAP = Fmap[varobj.domain]
            else:
                FMAP = Fmap
            if len(Emap.shape)>1:
                EMAP = Emap[varobj.domain]
            else:
                EMAP = Emap
        
        fname_lab = '-'.join([f'{int(l):03}' for l in lead_times])

        VMAP = np.mean(self.RT_ANOM[varname][itv],axis=0)
        
        if prop['cmap'] is None:
            prop['cmap'] = {0:'violet',22.5:'mediumblue',40:'lightskyblue',47.5:'w',52.5:'w',60:'gold',77.5:'firebrick',100:'violet'}
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(VMAP)),np.nanmax(abs(VMAP)))
        prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])
        prop['extend']='both'
        ax = varobj.plot_map(VMAP, prop = prop)
        ax.set_title(f'{varname} \nAnomaly',loc='left',fontweight='bold',fontsize=14)
        ax.set_title(f'Verification\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)
        
        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_file must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}_lt{fname_lab}_obs.png',bbox_inches='tight')
        plt.close()
        
        bounds = [-np.inf*np.ones(len(FMAP)),np.zeros(len(FMAP)),np.inf*np.ones(len(FMAP))]
        cat_fcst = get_probabilistic_fcst((FMAP,),(EMAP,),bounds)[0]    
        cat_obs = get_categorical_obs((VMAP,),bounds)[0]
        
        # make categorical verification map
        cmap,levels = get_cmap_levels(['deepskyblue','coral'],[0,1]) 
        prop['cmap'] = cmap
        prop['levels'] = [0,.5,1]
        prop['cbarticklabels']=['Below','','Above']
        prop['cbar_label']=None
        prop['extend']='both'
        ax = varobj.plot_map(cat_obs[1], prop = prop)
        ax.set_title(f'{varname} \nCategorical',loc='left',fontweight='bold',fontsize=14)
        ax.set_title(f'Verification\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)
        
        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}-CAT_lt{fname_lab}_obs.png',bbox_inches='tight')
        plt.close()
        
        # make hit/miss map
        if regMask is not None:
            prop['regMask'] = regMask
            countries = regionmask.defined_regions.natural_earth.countries_110
            regIndex = [i for i in countries.regions if countries.regions[i].name==regMask][0]
            labels = countries.mask(varobj.longrid,varobj.latgrid)
            regMask = np.where(labels.data[varobj.domain]==regIndex,True,False)
        else:
            regMask = np.ones(varobj.lon.shape).astype(bool)
        
        cmap,levels = get_cmap_levels(['lightpink','mediumseagreen'],[-1,1]) 
        prop['cmap'] = cmap
        prop['levels'] = [-1,0,1]
        prop['cbarticklabels']=['Miss','','Hit']
        prop['cbar_label']=None
        prop['extend']='both'
        hitmiss=np.sign((2*cat_obs[1]-1)*(2*cat_fcst[1]-1))
        
        for pthresh in listify(prob_thresh):
        
            validwhere = (abs(cat_fcst[1]-.5)<(pthresh/100-.5))
                        
            latwt = np.cos(np.radians(varobj.lat))
            fcst = np.array(cat_fcst).T
            obs = np.array(cat_obs).T
            
            HSS = get_heidke(fcst[regMask],obs[regMask],weights=latwt[regMask],\
                                        categorical=True)
            HSS_thresh = get_heidke(fcst[~validwhere & regMask],\
                                    obs[~validwhere & regMask],\
                                        weights=latwt[~validwhere & regMask],\
                                        categorical=True)
            
            RPSS = get_rpss(fcst,obs,weights=latwt,\
                                       categorical=False)
            RPSS_thresh = get_rpss(fcst[~validwhere & regMask],\
                                   obs[~validwhere & regMask],\
                                       weights=latwt[~validwhere & regMask],\
                                       categorical=False)
            
            mask = (pthresh/100-0.5)-(abs(cat_fcst[1]-.5))
            ax = varobj.plot_map(hitmiss, mask=mask, prop = prop)
            ax.set_title(f'{varname} \nHit/Miss >{pthresh}%',loc='left',fontweight='bold',fontsize=14)
            ax.set_title(f'Verification\n'+
                         f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                         loc='right',fontsize=14)
            ax.text( 0.03, 0.12, f'Heidke (all) = {HSS:.3f} \nHeidke (>{pthresh}%) = {HSS_thresh:.3f}'+\
                    f'\nRPSS (all) = {RPSS:.3f} \nRPSS (>{pthresh}%) = {RPSS_thresh:.3f}',
                    ha='left', va='center', transform=ax.transAxes,fontsize=12,zorder=99)
            
            if save_to_path is None:
                plt.show()
            elif not isinstance(save_to_path,str):
                print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
            else:
                plt.savefig(f'{save_to_path}/{varname}-CAT_lt{fname_lab}_hitmiss_{pthresh}.png',bbox_inches='tight')
            plt.close()
            
        return {'HSS_all':HSS,f'HSS_{pthresh}':HSS_thresh,'RPSS_all':RPSS,f'RPSS_{pthresh}':RPSS_thresh}

