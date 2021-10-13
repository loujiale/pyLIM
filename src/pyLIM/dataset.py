r"""
Read and prep data for use in the LIM

Sam Lillo
"""

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import copy
from datetime import datetime as dt,timedelta
import matplotlib as mlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from global_land_mask import globe
from scipy.ndimage import gaussian_filter as gfilt
import statistics
from scipy.interpolate import griddata,interp2d
import pickle
import xarray as xr

from .tools import *
from .plot import PlotMap

class varDataset:
    
    r"""
    Creates an instance of varDataset object based on requested files & variable.
    
    Parameters
    ----------
    path : str
        Directory containing files for concatenation.
    varname : str
        Name of variable in files.
    climo : ndarray
         
    
    Other Parameters
    ----------------
    level : float
        If file contains data for multiple levels
    climoyears : tuple
        (start year, end year) to slice data
    datebounds : tuple
        (start date, end date) as string 'm/d'
    fullseason : bool
        True to use only complete seasons, e.g. DJF, can't cutoff at year end 
    latbounds : tuple
        (south lat, north lat) to slice data
    lonbounds : tuple)
        (west lon, east lon) to slice data
    time_window : int
        Used for running_mean, number of time units.
    time_sample : str
        Integer + string, indicating number of time sample which can be 
        'D' = Day, 'W' = Week, 'M' = Month, 'Y' = Year.
        Example: '3D' would be a time sample of one ob every 3 days.
    landmask : bool
        If True, only use values over landmass.    
        
    Returns
    -------
    Dataset : object
        An instance of Dataset.
    """

    def __repr__(self):
        
        summary = ["<variable dataset for LIM>"]
        
        #Add general summary
        emdash = ' \u2014 '
        summary_keys = {'label':self.varlabel,\
                        'data path':self.datapath,\
                        'years':emdash.join([str(x) for x in self.climoyears]),\
                        'date bounds':emdash.join(self.datebounds),\
                        'lat bounds':self.latbounds,\
                        'lon bounds':self.lonbounds,\
                        'anomaly range':(np.amin(self.anomaly),np.amax(self.anomaly)),\
                        'anomaly stdev':self.anom_stdev,\
                        'averaging window':self.time_window,\
                        'time sample':self.time_sample[:-1]+' '+{'D':'day','W':'week','M':'month','Y':'year'}[self.time_sample[-1]],\
                        }
        summary_keys.update(self.attrs)
        #Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)
    
    def __init__(self,varlabel,datapath,varname,**kwargs):
        
        self.varlabel = varlabel
        self.datapath = datapath
        self.varname = varname
        # kwargs
        self.level = kwargs.pop('level',None) # vertical level 
        self.climoyears = kwargs.pop('climoyears',None) # year range (start,end)
        self.datebounds = kwargs.pop('datebounds',('1/1','12/31')) # date range (start,end) string 'm/d'
        fullseason = kwargs.pop('fullseason',True) # whether to only use full seasons, or partial seasons
        self.latbounds = kwargs.pop('latbounds',None) # latitude bounds (S,N)
        self.lonbounds = kwargs.pop('lonbounds',None) # longitude bounds (W,E)
        self.time_window = kwargs.pop('time_window',None) # number of time samples to average
        self.time_sample = kwargs.pop('time_sample','1D') # time sample string starting with a whole number, and D:day, W:week, M:month
        self.climo = kwargs.pop('climo',None) # if another climatology is to be used
        self.landmask = kwargs.pop('landmask',False) # only use values over land
        self.coarsegrain = kwargs.pop('coarsegrain',None) # spatial smoothing
        self.attrs = {}
        
        # Concatenate all files into one dataset
        filenames = sorted([join(datapath, f) for f in listdir(datapath) \
                     if isfile(join(datapath, f)) and f.endswith('.nc')])
    
        ds = self._get_ds(filenames)
        self.lat = ds['lat'][self.domain]
        self.lon = ds['lon'][self.domain]
        self.latgrid = ds['lat']
        self.longrid = ds['lon']
        
        # Make xr dataset
        vardict={"raw": {'dims':("time","location"),'data':ds['var']},
        }
        coords={
            "lon": {'dims':('location',),'data':self.lon,},
            "lat": {'dims':('location',),'data':self.lat,},
            "time": {'dims':('time',),'data':ds['time'],},
        }
        ds=xr.Dataset.from_dict({
            'coords':coords,
            'data_vars':vardict,
            'dims':[k for k,v in coords.items()],
        })
        
        # slice for years
        if self.climoyears is None:
            self.climoyears = (int(min(ds.time).dt.year),int(max(ds.time).dt.year))
        ds = ds.sel(time=slice(f"{min(self.climoyears)}-01-01", f"{max(self.climoyears)}-12-31"))
        
        # get running mean
        if self.time_window is not None:
            ds = ds.rolling(time=self.time_window).mean().dropna('time')

        # Get climatology and anomalies
        if self.climo is None:
            self._get_climo(ds)
        self.anomaly = self.get_anom(ds).raw.data
        
        self.time = np.array([dt.utcfromtimestamp(t.astype(int)*1e-9) for t in ds.time])

        if fullseason:
            firstday = int(f'{dt(2000,*[int(j) for j in self.datebounds[0].split("/")]):%j}')
            lastday = int(f'{dt(2000,*[int(j) for j in self.datebounds[1].split("/")]):%j}')
            datewhere = np.where(list(map(self.date_range_test,self.time)) & \
                                 (self.time>=dt.strptime(f'{min(self.climoyears)}/{firstday}','%Y/%j')) & \
                                 (self.time<=dt.strptime(f'{max(self.climoyears)}/{lastday}','%Y/%j')))[0]
        else:
            datewhere = np.where(list(map(self.date_range_test,self.time)))[0]

        self.time = self.time[datewhere]
        self.anomaly = self.anomaly[datewhere]

        # bulk statistics
        self.anom_stdev = np.nanstd(self.anomaly)
        self.anom_mean = np.nanmean(self.anomaly)        
        
    def _get_ds(self,filenames):
        
        ds = {}
        print(f'--> Starting to gather data for {self.varlabel}')
        timer_start = dt.now()        
        for prog,fname in enumerate(filenames):
            with xr.open_dataset(fname) as ds0:
                
                lat_name = ([s for s in ds0.variables.keys() if 'lat' in s]+[None])[0]
                lon_name = ([s for s in ds0.variables.keys() if 'lon' in s]+[None])[0]
                lev_name = ([s for s in ds0.variables.keys() if 'lev' in s or 'lv_' in s]+[None])[0]
                time_name = ([s for s in ds0.variables.keys() if 'time' in s]+[None])[0]
                var_name = self.varname
                
                self.attrs.update(ds0[var_name].attrs)
                
                ds['lat']=ds0[lat_name].data
                ds['lon']=ds0[lon_name].data%360
                if len(ds['lat'].shape)==1:
                    ds['lon'],ds['lat'] = np.meshgrid(ds['lon'],ds['lat'])
                if lev_name is not None:
                    if self.level is None:
                        self.level = ds0[lev_name].data[0]
                    else:
                        self.attrs['level']=self.level
                    ds0 = ds0.sel({lev_name:self.level})
    
                # resample time
                try:
                    ds0=ds0.resample({time_name:self.time_sample}).reduce(np.mean)
                except:
                    pass
                try:
                    ds0=ds0.resample({time_name:self.time_sample}).interpolate("linear")
                except:
                    pass
                
                tmp = ds0[time_name].data                       
                if prog==0:
                    ds['time'] = tmp
                else:
                    ds['time'] = np.append(ds['time'],tmp)
                
                try:
                    newdata = ds0.transpose(time_name,lat_name,lon_name)[var_name].data
                except:
                    newdata = ds0[var_name].data
                    
                del ds0

            if self.coarsegrain is not None:
                lonres = abs(statistics.mode(np.gradient(ds['lon'].data)[1].flatten()))
                latres = abs(statistics.mode(np.gradient(ds['lat'].data)[0].flatten()))
                lonbin = int(self.coarsegrain/lonres)
                latbin = int(self.coarsegrain/latres)
                new_lats = ds['lat'][::latbin,::lonbin]
                new_lons = ds['lon'][::latbin,::lonbin]
                newdata = newdata[:,::latbin,::lonbin]
                ds['lat']=new_lats
                ds['lon']=new_lons

            self.mapgrid = np.ones(newdata.shape[1:])*np.nan
            
            if self.latbounds is None:
                lim_S = np.amin(ds['lat']) 
                lim_N = np.amax(ds['lat'])
            else:
                lim_S = min(self.latbounds)
                lim_N = max(self.latbounds)
            if self.lonbounds is None:
                lim_W = np.amin(ds['lon']) 
                lim_E = np.amax(ds['lon'])
            else:
                lim_W = min(self.lonbounds)
                lim_E = max(self.lonbounds)
            self.zmask = np.ones(self.mapgrid.shape,dtype=bool)
            if self.landmask:
                lon_shift = ds['lon'].copy()
                lon_shift[ds['lon']>180] = ds['lon'][ds['lon']>180]-360
                self.zmask = self.zmask*globe.is_land(ds['lat'],lon_shift)
            
            self.domain = np.where((ds['lat']>=lim_S) & \
                              (ds['lat']<=lim_N) & \
                              (ds['lon']>=lim_W) & \
                              (ds['lon']<=lim_E) & \
                              self.zmask)
            
            newdata = np.array([n[self.domain] for n in newdata])

            if prog==0:
                ds['var'] = newdata
            else:
                ds['var'] = np.append(ds['var'],newdata,axis=0)
            update_progress('Gathering data',(prog+1)/len(filenames))
        print('--> Completed gathering data (%.1f seconds)' \
              % (dt.now()-timer_start).total_seconds())
        return ds

    def _get_climo(self,ds,yearbounds=None):
        if yearbounds is None:
            yearbounds = self.climoyears
        if 'D' in self.time_sample or 'W' in self.time_sample:
            climo = ds.groupby("time.dayofyear").mean("time").interp(dayofyear=np.arange(1,367))
            climo.raw.data = gfilt(climo.raw.data,[15,0])
        elif 'M' in self.time_sample:
            climo = ds.groupby("time.month").mean("time").interp(month=np.arange(1,13))
        self.climo = climo
    
    def get_anom(self,dataset,varname='raw'):
        dataset = dataset.rename({varname:'raw'})
        if 'D' in self.time_sample or 'W' in self.time_sample:
            anomaly = dataset.groupby("time.dayofyear") - self.climo
        elif 'M' in self.time_sample:
            anomaly = dataset.groupby("time.month") - self.climo
        return anomaly.rename({'raw':varname})

    def date_range_test(self,t):
        t_min,t_max = [dt(2000,*[int(j) for j in i.split('/')]) for i in self.datebounds]
        t_min += timedelta(days=-1)
        t_max += timedelta(days=1)
        if t_min<t_max:
            test1 = (t.replace(year=2000)>t_min)
            test2 = (t.replace(year=2000)<t_max)
            return test1 & test2
        else:
            test1 = (t_min<t.replace(year=2000)<dt(2001,1,1))
            test2 = (dt(2000,1,1)<=t.replace(year=2000)<t_max)
            return test1 | test2
    
    def subset(self,datebounds = ('1/1','12/31'),fullseason=True,latbounds=None,lonbounds=None):
                           
        r"""
        Creates a new instance of varDataset object subset by dates.

        Parameters
        ----------
        datebounds : tuple
            Start and end dates of season, in form "m/d".
        fullseason : bool
            If True, starts data at beginning of season. If False, starts data at beginning of first year.
            Only matters for seasons that cross Jan 1st.
        
        Returns
        -------
        newobj : object
            A new instance of varDataset.
        """
        
        newobj = copy.deepcopy(self)
        newobj.datebounds = datebounds
        if fullseason:
            datewhere = np.where(list(map(newobj.date_range_test,newobj.time)) & \
                                 (newobj.time>=dt.strptime(f'{min(self.climoyears)}/{self.datebounds[0]}','%Y/%m/%d')) & \
                                 (newobj.time<=dt.strptime(f'{max(self.climoyears)}/{self.datebounds[1]}','%Y/%m/%d')))[0]
        else:
            datewhere = np.where(list(map(newobj.date_range_test,newobj.time)))[0]

        newobj.time = newobj.time[datewhere]
        anom = newobj.anomaly[datewhere]

        if latbounds is None:
            lim_S = np.amin(self.lat) 
            lim_N = np.amax(self.lat)
        else:
            lim_S = min(latbounds)
            lim_N = max(latbounds)
        if lonbounds is None:
            lim_W = np.amin(self.lon) 
            lim_E = np.amax(self.lon)
        else:
            lim_W = min(lonbounds)
            lim_E = max(lonbounds)

        newobj.domain = np.where((self.latgrid>=lim_S) & \
                          (self.latgrid<=lim_N) & \
                          (self.longrid>=lim_W) & \
                          (self.longrid<=lim_E) & \
                          self.zmask)

        subdom = np.where((self.lat>=lim_S) & \
                          (self.lat<=lim_N) & \
                          (self.lon>=lim_W) & \
                          (self.lon<=lim_E))
        
        newobj.anomaly = np.array([n[subdom] for n in anom])

        newobj.anom_stdev = np.nanstd(newobj.anomaly)
        newobj.anom_mean = np.nanmean(newobj.anomaly)

        return newobj

    def interpSpace(self,ds=None,lat=None,lon=None,z=None):
        
        if ds is None:
            f = interp2d(lon,lat,z)
            out = np.array([f(x,y)[0] for x,y in zip(self.lon,self.lat)])
            inan = np.where((self.lon<min(lon)) | (self.lon>max(lon)) | (self.lat<min(lat)) | (self.lat>max(lat)))
            out[inan]=0
            return out
        else:
            ds = xr.concat([ds.interp(coords={'latitude':y,'longitude':x}) \
                       for y,x in zip(self.lat,self.lon)],dim='location')
            return ds

    def regrid(self,a):
        # Take 1-d vector of same length as domain 
        # and transform to original grid
        b = self.mapgrid.copy()
        b[self.domain] = a
        return b
    
    def flatten(self,a):
        # Take n-d array and flatten
        b = a[self.domain]
        return b

    def plot_map(self,z=None,time=None,ax=None,projection='dynamic',prop={}):
                           
        r"""
        Takes space vector or time, and plots a map of the data.

        Parameters
        ----------
        z : ndarray / list
            Vector with same length as number of space points in self.anomaly
        time : datetime object
            Optional to plot self.anomaly data from specific time in self.time.
        ax : axes instance
            Can pass in own axes instance to plot in existing axes.

        Other Parameters
        ----------------
        prop : dict
            Options for plotting, keywords and defaults include
            * 'cmap' - None
            * 'levels' - negative abs data max to abs data max
            * 'figsize' - (10,6)
            * 'dpi' - 150
            * 'cbarticks' - every other level, determined in get_cmap_levels
            * 'cbarticklabels' - same as cbarticks
            * 'cbar_label' - None. Optional string, or True for self.attrs['units']
            * 'extend' - 'both'
            * 'interpolate' - None. If float value, interpolates to that lat/lon grid resolution.
            * 'drawcountries' - False
            * 'drawstates' - False

        Returns
        -------
        ax : axes instance
        """
                           
        if z is None and time is None:
            z = self.anomaly[-1]
        elif z is None:
            z = self.anomaly[list(self.time).index(time)]
        
        default_prop={'cmap':None,'levels':None,'fill':True,'figsize':(10,6),'dpi':150,'cbarticks':None,'cbarticklabels':None,\
                      'cbar_label':None,'extend':'both','interpolate':None,'drawcountries':False,'drawstates':False,\
                      'contour_total':False,'addcyc':False,'res':'m','central_longitude':-90}
        prop = add_prop(prop,default_prop)
            
        if prop['cmap'] is None:
            prop['cmap'] = {0:'violet',22.5:'mediumblue',40:'lightskyblue',\
                47.5:'w',52.5:'w',60:'gold',77.5:'firebrick',100:'violet'}
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(z)),np.nanmax(abs(z)))
        prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])

        mycmap,levels = get_cmap_levels(prop['cmap'],prop['levels'])

        if prop['cbarticklabels'] is None:
            cbticks = levels[::2]
            cbticklabels = cbticks
        elif prop['cbarticks'] is None:
            cbticks = prop['levels']
            cbticklabels = prop['cbarticklabels']
        else:
            cbticks = prop['cbarticks']
            cbticklabels = prop['cbarticklabels']

        #m,addcyc = map_proj(self.lat,self.lon)
    
        if prop['interpolate'] is None:     
            zmap = self.regrid(z)
            lat = self.latgrid
            lon = self.longrid
        else:
            xMin = max([0,min(self.lon)-5])
            yMin = max([-90,min(self.lat)-5])
            xMax = min([360,max(self.lon)+5])
            yMax = min([90,max(self.lat)+5])
            
            grid_res = prop['interpolate']
            xi = np.arange(xMin, xMax+grid_res, grid_res)
            yi = np.arange(yMin, yMax+grid_res, grid_res)
            lon,lat = np.meshgrid(xi,yi)
            # grid the data.
            zLL = z[np.argmin((self.lon-xMin)**2+(self.lat-yMin)**2)]
            zLR = z[np.argmin((self.lon-xMax)**2+(self.lat-yMin)**2)]
            zUL = z[np.argmin((self.lon-xMin)**2+(self.lat-yMax)**2)]
            zUR = z[np.argmin((self.lon-xMax)**2+(self.lat-yMax)**2)]
            lonNew = np.array(list(self.lon)+[xMin,xMax,xMin,xMax])
            latNew = np.array(list(self.lat)+[yMin,yMin,yMax,yMax])
            zNew = np.array(list(z)+[zLL,zLR,zUL,zUR])
            lonNoNan = lonNew[~np.isnan(zNew)]
            latNoNan = latNew[~np.isnan(zNew)]
            zNoNan = zNew[~np.isnan(zNew)]
            zmask = np.where(np.isnan(zNew),1,0)
            zmap = griddata((lonNoNan,latNoNan), zNoNan, (xi[None,:], yi[:,None]), method='cubic')
            zmask = griddata((lonNew,latNew), zmask, (xi[None,:], yi[:,None]), method='linear')
            zmap[zmask>0.9]=np.nan

            zmap = zmap[:,np.where((xi>=min(self.lon)) & (xi<=max(self.lon)))[0]][np.where((yi>=min(self.lat)) & (yi<=max(self.lat)))[0],:]
            xi = xi[np.where((xi>=min(self.lon)) & (xi<=max(self.lon)))]
            yi = yi[np.where((yi>=min(self.lat)) & (yi<=max(self.lat)))]
            lon,lat = np.meshgrid(xi,yi)
            
        #create figure
        if ax is None:
            fig = plt.figure(figsize = prop['figsize'],dpi=prop['dpi'])
        else:
            # get the figure numbers of all existing figures
            fig_numbers = [x.num for x in mlib._pylab_helpers.Gcf.get_all_fig_managers()]
            # set figure as last figure number
            fig = plt.figure(fig_numbers[-1])
            
        # Add cyclic
        if len(lon.shape)==1 and len(lat.shape)==1:
            lons,lats = np.meshgrid(lon,lat)
        else:
            lons,lats = lon,lat
        if np.amax(lon)-np.amin(lon)>345:
            lonplt = np.concatenate([lons,lons[:,0][:,None]+360],axis=1)
            latplt = np.concatenate([lats,lats[:,0][:,None]],axis=1)
            dataplt = np.concatenate([zmap,zmap[:,0][:,None]],axis=1)
        else: 
            lonplt,latplt,dataplt = lons,lats,zmap
        
        #Fill poles
        try:
            ipole = np.where(latplt==90)
            dataplt[ipole] = np.nanmean(dataplt[tuple([ipole[0]-1,ipole[1]])])
        except:
            pass
        try:
            ipole = np.where(latplt==-90)
            dataplt[ipole] = np.nanmean(dataplt[tuple([ipole[0]-1,ipole[1]])])
        except:
            pass
        
        m = PlotMap(projection,lon=self.lon,lat=self.lat,res=prop['res'],\
                    central_longitude=prop['central_longitude'])
        m.setup_ax(ax=ax)

        if prop['fill']:
            cbmap = m.contourf(lonplt, latplt, dataplt, cmap=mycmap,levels=levels,extend=prop['extend'],zorder=0)
            if prop['interpolate'] is not None and self.landmask:
                m.fill_water(zorder=9)
        else:
            cbmap = ax.contour(lonplt, latplt, dataplt, colors='k',levels=levels)
            if prop['interpolate'] is not None and self.landmask:
                m.fill_water(zorder=9)
                
        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        if prop['drawcountries']:
            m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        if prop['drawstates']:
            m.drawstates(linewidth=0.5,color='0.25',zorder=10)

        if m.projection in ['NorthPolarStereo','SouthPolarStereo']:
            m.stereo_lat_bound()
        
        if prop['fill']:
            #plt.subplots_adjust(bottom=0.12)
            #cax = fig.add_axes([0.15,0.05,0.7,0.03])
        
            #cbar = fig.colorbar(cbmap,ticks=cbticks,cax=cax,orientation='horizontal')
            cbar = plt.colorbar(cbmap,ticks=cbticks,orientation='horizontal',ax=ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
            cbar.ax.set_xticklabels(cbticklabels)
            
            if prop['cbar_label'] is not None:
                cbar_label = prop['cbar_label']
                if cbar_label is True:
                    cbar_label = self.attrs['units']
                cbar.ax.set_xlabel(cbar_label,fontsize=14)

        return m.ax
    
    
    def save_to_netcdf(self,path,segmentby=None):
        
        data_seg = {}
        if segmentby in (None,'all'):
            running_mean = self.anomaly
            time = self.time
            data_seg['all'] = {'running_mean':running_mean,'time':time}
            
        elif segmentby == 'year':
            years = np.array([t.year for t in self.time])
            for yr in range(min(years),max(years)+1):
                idata = np.where(years==yr)
                running_mean = self.anomaly[idata]
                time = self.time[idata]
                data_seg[yr] = {'running_mean':running_mean,'time':time}                

        for K,V in data_seg.items():
            Vmap = list(map(self.regrid,V['running_mean']))
            Cmap = list(map(self.regrid,self.climo.raw.data))
            vardict = {"anomaly": {'dims':("time","lat","lon"), 
                                   'data':Vmap, 
                                   'attrs':copy.copy(self.attrs)},
                       "climo": {'dims':("doy","lat","lon"), 
                                 'data':Cmap, 
                                 'attrs':copy.copy(self.attrs)}
                       }
            coords={
                "lon": {'dims':('lon',),'data':self.longrid[0,:],
                        'attrs':{'long_name':'longitude','units':'degrees_east'}},
                "lat": {'dims':('lat',),'data':self.latgrid[:,0],
                        'attrs':{'long_name':'latitude','units':'degrees_north'}},
                "time": {'dims':('time',),'data':V['time'],
                         'attrs':{'long_name':'time'}},
                "doy": {'dims':('doy',),'data':self.climo.dayofyear.data,
                        'attrs':{'long_name':'day of the year'}},
            }
            
            additional_info = ''
            if self.time_window not in (None,1):
                additional_info += f'.boxcar{self.timewindow}'
            save_ncds(vardict,coords,filename=join(path,f'{self.varlabel}{additional_info}.{K}.nc'))
        

    def process_new_data(self,filename,varname):
        
        r"""
        Reads in netcdf file and processes the data according to the 
        specifications and climatology built in the variable object.
        """
                        
        ds = {}
        print(f'\n--> Starting to process new data for {self.varlabel}')
        timer_start = dt.now()
        
        with xr.open_dataset(filename) as ds0:
            ds0 = ds0[varname].to_dataset()
            print(ds0)
    
            lon_name = ([s for s in ds0.variables.keys() if 'lon' in s]+[None])[0]
            lat_name = ([s for s in ds0.variables.keys() if 'lat' in s]+[None])[0]
            lev_name = ([s for s in ds0.variables.keys() if 'lev' in s \
                         or 'lv_' in s or 'isobaric' in s]+[None])[0]
            time_name = ([s for s in ds0.variables.keys() if 'time' in s]+[None])[0]
            
            ds0.assign_coords({lon_name:ds0[lon_name]%360})
            if lev_name is not None:
                ds0 = ds0.sel({lev_name:self.level})
            
            ds0 = ds0.rename({lon_name:'longitude',lat_name:'latitude',time_name:'time'})
    
            # resample to variable object time_sample
            print(f'resampling time to {self.time_sample}')
            ds0=ds0.resample({'time':self.time_sample}).reduce(np.mean)
            ds0=ds0.resample({'time':self.time_sample}).interpolate("linear")

            # interpolate to variable object space
            print(f'interpolating to {self.anomaly.shape[1]} locations')
            ds0 = self.interpSpace(ds0)
            ds0 = ds0.transpose('time','location')
    
            # get rolling mean
            print(f'getting {self.time_window}x{self.time_sample} rolling mean')
            ds0 = ds0.rolling(time=self.time_window).mean().dropna('time')
        
            # get anomaly
            print('getting anomaly')
            ds['anomaly'] = self.get_anom(ds0,varname)[varname].data
            ds['time'] = np.array([dt.utcfromtimestamp(t.astype(int)*1e-9) for t in ds0[time_name].data])

        print('--> Completed processing data (%.1f seconds)' \
              % (dt.now()-timer_start).total_seconds())
        return ds



class eofDataset:

    r"""
    Creates an instance of eofDataset object based on requested files & variable.
    
    Parameters
    ----------

    Other Parameters
    ----------------
        
    Returns
    -------
    Dataset : object
        An instance of eofDataset.
    """
    
    def __repr__(self):
        
        summary = ["<EOF dataset for LIM>"]

        varobj = self.varobjs[0]
        dateb = varobj.datebounds
        yearb = [str(x) for x in varobj.climoyears]
        
        #Add general summary
        emdash = ' \u2014 '
        summary_keys = {'variables':', '.join([v.varlabel for v in self.varobjs]),\
                        'years':emdash.join(yearb),\
                        'date bounds':emdash.join(dateb),\
                        'first 5 var expl':', '.join([f'{x:.03f}' for x in self.varExplByEOF[:5]]),\
                        }
        #Add dataset summary
        summary.append("EOF Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)
    
    def __init__(self,varobjs,max_eofs=100,eof_in=None,time_extended=None,skip=1):
        self.time_extended = time_extended
        self.skip = skip
        prepped = []
        if isinstance(varobjs,(tuple,list)):
            self.varobjs = tuple(varobjs)
        else:
            self.varobjs = tuple([varobjs])
        for obj in self.varobjs:
            varstd = (obj.anomaly-obj.anom_mean)/obj.anom_stdev
            tmp = get_area_weighted(varstd,obj.lat)
            tmp = tmp.reshape(tmp.shape[0],np.product(tmp.shape[1:]))
            prepped.append(tmp)
        prepped = np.concatenate(prepped,axis=1)
        if max_eofs > 0:
            if time_extended is None:
                eof_dict = get_eofs(prepped,max_eofs,eof_in=eof_in)
            else:
                dims = prepped.shape
                a = prepped.reshape((dims[0],np.product(dims[1:])),order='F')
                prepped_ext = self._extendmat(a,time_extended,skip)
                eof_dict = get_eofs(prepped_ext,max_eofs,eof_in=eof_in)
                eof_dict['eof'] = eof_dict['eof'].reshape((len(eof_dict['eof']),*dims[1:],time_extended),order='F')
            self.__dict__.update(eof_dict)                

    def _extendmat(self,a,per,skip=1):
        a2=np.concatenate([np.ones([per*skip,a.shape[1]])*np.nan,a])
        b=np.ones([int(a.shape[0]//skip),a.shape[1]*per])*np.nan
        for i in range(b.shape[0]): #for each day
            for j in range(per): #for each per
                b[i,j*a.shape[1]:(j+1)*a.shape[1]]=a2[i*skip-j*skip+per*skip,:]
        return np.array(b)

#    def _TEeof(self,testdata,per,howmany):
#        dims = testdata.shape
#        a = testdata.reshape((dims[0],np.product(dims[1:])),order='F')
#        a_ext = self._extendmat(a,per)
#        E,expvar,Z = calceof(a_ext[per-1:],howmany)
#        EOFs = [EE.reshape(*dims[1:],per,order='F') for EE in E.T]
#        return EOFs,expvar,Z
        
    def get_pc(self,ds,trunc=None):
        
        if not isinstance(ds,dict):
            print('ds must be a dictionary')
        
        if len(self.varobjs)==1 and self.varobjs[0].varlabel not in ds.keys():
            ds = {self.varobjs[0].varlabel:ds}
        
        prepped = []
        for obj in self.varobjs:
            data = ds[obj.varlabel]
            varstd = (data['anomaly']-obj.anom_mean)/obj.anom_stdev
            tmp = get_area_weighted(varstd,obj.lat)
            tmp = tmp.reshape(tmp.shape[0],np.product(tmp.shape[1:]))
            prepped.append(tmp)
        prepped = np.concatenate(prepped,axis=1)
        
        if trunc is None:
            trunc = prepped.shape[1]
        pc = get_eofs(prepped,eof_in=self.eof[:trunc])
        
        return pc
    
    def reconstruct(self,pcs,order=1,num_eofs=None,pc_wt=None):

        r"""
        Method for reconstructing spatial vector from PCs and EOFs
        
        Parameters
        ----------
        pcs : list or tuple or ndarray
            list of floating numbers corresponding to leading PCs
        order : int
            1 = forecast PCs, 2 = error covariance matrix
        num_eofs : int
            truncation for reconstruction. Default is to retain all EOFs available.
        pc_wt : dict
            keys corresponding to PC number. Values corresponding to weight        
            
        Returns
        -------
        recon : dict
            dictionary with keys = variable label, and values = reconstructed spatial vector.
        """
        
        if not isinstance(pcs,np.ndarray):
            pcs = np.array(pcs)
        if len(pcs.shape)<(1+order):
            pcs = pcs.reshape([1]+list(pcs.shape))
        if pc_wt is not None:
            pcs = np.array([pcs[:,i-1]*pc_wt[i] for i in pc_wt.keys()]).squeeze().T
            print(pc_wt)
        if num_eofs is None:
            num_eofs = min(self.eof.shape[0],pcs.shape[-1])
        if order==1:
            recon = np.dot(pcs[:,:num_eofs],self.eof[:num_eofs, :])
        if order==2:
            recon = np.array([np.diag(np.matrix(self.eof).T[:,:num_eofs] @ p \
                           @ np.matrix(self.eof)[:num_eofs,:])**0.5 for p in pcs])

        return_var = {}
        if len(listify(self.varobjs))>1:
            i0 = 0
            for varobj in self.varobjs:
                nlen = varobj.anomaly.shape[1]
                return_var[varobj.varlabel] = recon[:,i0:i0+nlen]*varobj.anom_stdev/np.sqrt(np.cos(np.radians(varobj.lat)))
                i0 += nlen
        else:
            varobj = self.varobjs[0]
            return_var[varobj.varlabel] = recon*varobj.anom_stdev/np.sqrt(np.cos(np.radians(varobj.lat)))
        
        return return_var
    
    def plot(self,num_eofs=5,return_figs=False,prop={}):
        
        r"""
        Map EOFs and plot PC timeseries in subplots.
        Include percent variance explained.
        
        Parameters
        ----------
        num_eofs : int
            number of EOFs to plot. Default is 4
        return_figs : bool
            Return list of the figures. Default is False
            
        Returns
        -------
        List of figures
        """

        
        default_prop={'cmap':{0:'b',.45:'w',.55:'w',1:'r'},'levels':None,'fill':True,'figsize':(10,6),'dpi':150,'cbarticks':None,'cbarticklabels':None,\
                      'cbar_label':None,'extend':'both','interpolate':None,'drawcountries':False,'drawstates':False,\
                      'contour_total':False,'addcyc':False,'res':'m','central_longitude':-90}
        prop = add_prop(prop,default_prop)
                
        PCs = self.pc[:,:num_eofs].T
        expVARs = self.varExplByEOF[:num_eofs]*100
        EOFs = self.eof[:num_eofs]
        
#        TE = self.time_extended
#        if TE is None:
#            gs = gridspec.GridSpec(2, 1, height_ratios=[7,3]) 
#        else:
#            gs = gridspec.GridSpec(TE+1, 1)
#        if TE is None:
#            TE = 1
#            EOFs = np.expand_dims(self.eof[:num_eofs],-1)
#        else:
#            EOFs = self.eof[:num_eofs]
                
        gs = gridspec.GridSpec(2, len(self.varobjs), height_ratios=[7,3])
        
        figs = []
        for iEOF,(EOF,PC,VAR) in enumerate(zip(EOFs,PCs,expVARs)):

            fig=plt.figure(figsize=(9,9))
            
            i0 = 0
            for ivar,varobj in enumerate(self.varobjs):
                
                nlen = varobj.anomaly.shape[1]
                Evar = EOF[i0:i0+nlen]
                i0 +=nlen
                
                proj=PlotMap(projection='dynamic',lon=varobj.lon,lat=varobj.lat).proj
                ax=fig.add_subplot(gs[0,ivar],projection=proj)
                varobj.plot_map(Evar,ax=ax,prop=prop)
                             
                ax.set_title(f'EOF {iEOF+1} (Exp Var={VAR:.1f}%)',size=16)

                 
            ax2=fig.add_subplot(gs[1,:])
            ax2.set_title(f'PC {iEOF+1}',size=16)
            ax2.plot(self.varobjs[0].time,PC)
            yrLoc = mlib.dates.YearLocator(5)
            ax2.xaxis.set_major_locator(yrLoc)
            plt.gca().xaxis.set_major_formatter(mlib.dates.DateFormatter('%Y'))
            ax2.set_ylim([-1.1*np.max(abs(PC)),1.1*np.max(abs(PC))])
            ax2.grid()
        
            figs.append(fig)
            plt.show()
            plt.close()
        
        if return_figs:
            return figs

    def save_to_netcdf(self,path):
        
        vardict = {"pc": {'dims':("time","index"), 
                          'data':self.pc,
                          'attrs':{'long_name':'principal components'}},
                   "varexp": {'dims':("index",), 
                              'data':self.varExplByEOF,
                              'attrs':{'long_name':'variance explained by EOF','units':'fraction of 1'}}
                   }
        coords={"index": {'dims':("index",), 'data':np.arange(self.pc.shape[1])+1},
                "time": {'dims':("time",), 'data':self.varobjs[0].time}}
        eofname = []
            
        i0 = 0
        for varobj in self.varobjs:
            nlen = varobj.anomaly.shape[1]
            Emap = list(map(varobj.regrid,self.eof[:,i0:i0+nlen]))
            i0 += nlen
            
            attrs = copy.copy(varobj.attrs)
            attrs.update({'stdev':varobj.anom_stdev})
            
            vardict.update({f"eof_{varobj.varlabel}": 
                {'dims':("index","lat","lon"), 
                 'data':Emap,
                 'attrs':attrs}
                })
            
            #_{varobj.varlabel}
            coords.update({"lon": varobj.longrid[0,:],
                           "lat": varobj.latgrid[:,0]})          
            
            coords.update({"lon": {'dims':("lon",),'data':varobj.longrid[0,:],
                                       'attrs':{'long_name':'longitude','units':'degrees_east'}},
                            "lat": {'dims':("lat",),'data':varobj.latgrid[:,0],
                                      'attrs':{'long_name':'latitude','units':'degrees_north'}}
                            })
            eofname.append(varobj.varlabel)
        
        dateb = [f'{dt.strptime(b,"%m/%d"):%b%d}' for b in varobj.datebounds]
        yearb = varobj.climoyears
        fname = f'EOF_{"+".join(eofname)}_{dateb[0]}-{dateb[1]}_{min(yearb)}-{max(yearb)}.nc'
        save_ncds(vardict,coords,filename=join(path,fname))
    
    def save_to_pickle(self,path):
        
        eofname = [v.varlabel for v in self.varobjs]
        varobj = self.varobjs[0]
        dateb = [f'{dt.strptime(b,"%m/%d"):%b%d}' for b in varobj.datebounds]
        yearb = varobj.climoyears
        fname = f'EOF_{"+".join(eofname)}_{dateb[0]}-{dateb[1]}_{min(yearb)}-{max(yearb)}.p'        
        pickle.dump(self, open( os.path.join(path,fname) ,'wb'))
    

