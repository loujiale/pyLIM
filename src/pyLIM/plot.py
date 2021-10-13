"""
Plotting methods for pyLIM.

Sam Lillo
"""

import numpy as np
from scipy import interp
import warnings

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib as mlib
    import matplotlib.lines as mlines
    import matplotlib.path as mpath
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

from .tools import *

class PlotMap(object):
    
    r"""Plotting methods for linear inverse forecast model.
    
    This class contains various methods for plotting relevant information
    for Dataset and Verif.
    """
    
    def __init__(self,projection='PlateCarree',lon=None,lat=None,res='m',**kwargs):
        
        r"""
        Initialize a cartopy instance passed projection.
        
        Parameters:
        -----------
        projection
            String representing the cartopy map projection.
        ax
            Axis on which to draw on. Default is None.

        **kwargs
            Additional arguments that are passed to those associated with projection.
        """

        #create cartopy projection, if none existing
        if projection == 'dynamic':
            if lon is not None and lat is not None:
                projection = self.get_dynamic(lon,lat)
            else:
                print('must include lon and lat coordinates for dynamic projection')
        self.projection = projection
        self.proj = getattr(ccrs, projection)(**kwargs)
        self.res = res

        
    def setup_ax(self,ax=None):
        
        if ax is None:
            self.ax = plt.axes(projection=self.proj)
        else:
            self.ax = ax
        
    def stereo_lat_bound(self):
        
        #Get current axes if not specified
        self.ax = self._check_ax()
        
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        self.ax.set_boundary(circle, transform=self.ax.transAxes)

    def drawcoastlines(self,linewidths=1.2,linestyle='solid',color='k',res=None,ax=None,**kwargs):
        """
        Draws coastlines similarly to Basemap's m.drawcoastlines() function.
        
        Parameters:
        ----------------------
        linewidths
            Line width (default is 1.2)
        linestyle
            Line style to plot (default is solid)
        color
            Color of line (default is black)
        res
            Resolution of coastline. Can be a character ('l','m','h') or one of cartopy's available
            resolutions ('110m','50m','10m'). If none is specified, then the default resolution specified
            when creating an instance of Map is used.
        ax
            Axes instance, if not None then overrides the default axes instance.
        """
        
        #Get current axes if not specified
        ax = ax or self._check_ax()
        
        #Error check resolution
        if res is None: res = self.res
        res = self.check_res(res)
        
        #Draw coastlines
        coastlines = ax.add_feature(cfeature.COASTLINE.with_scale(res),linewidths=linewidths,linestyle=linestyle,edgecolor=color,**kwargs)
        
        #Return value
        return coastlines

    def drawcountries(self,linewidths=1.2,linestyle='solid',color='k',res=None,ax=None,**kwargs):
        """
        Draws country borders similarly to Basemap's m.drawcountries() function.
        
        Parameters:
        ----------------------
        linewidths
            Line width (default is 1.2)
        linestyle
            Line style to plot (default is solid)
        color
            Color of line (default is black)
        res
            Resolution of country borders. Can be a character ('l','m','h') or one of cartopy's available
            resolutions ('110m','50m','10m'). If none is specified, then the default resolution specified
            when creating an instance of Map is used.
        ax
            Axes instance, if not None then overrides the default axes instance.
        """
        
        #Get current axes if not specified
        ax = ax or self._check_ax()
        
        #Error check resolution
        if res is None: res = self.res
        res = self.check_res(res)
        
        #Draw coastlines
        countries = ax.add_feature(cfeature.BORDERS.with_scale(res),linewidths=linewidths,linestyle=linestyle,edgecolor=color,**kwargs)
        
        #Return value
        return countries
    
    def drawstates(self,linewidths=0.7,linestyle='solid',color='k',res=None,ax=None,**kwargs):
        """
        Draws state borders similarly to Basemap's m.drawstates() function.
        
        Parameters:
        ----------------------
        linewidths
            Line width (default is 0.7)
        linestyle
            Line style to plot (default is solid)
        color
            Color of line (default is black)
        res
            Resolution of state borders. Can be a character ('l','m','h') or one of cartopy's available
            resolutions ('110m','50m','10m'). If none is specified, then the default resolution specified
            when creating an instance of Map is used.
        ax
            Axes instance, if not None then overrides the default axes instance.
        """
        
        #Get current axes if not specified
        ax = ax or self._check_ax()
        
        #Error check resolution
        if res is None: res = self.res
        res = self.check_res(res)
        
        #Draw coastlines
        states = ax.add_feature(cfeature.STATES.with_scale(res),linewidths=linewidths,linestyle=linestyle,edgecolor=color,**kwargs)
        
        #Return value
        return states

    def fill_water(self,facecolor='w',zorder=None):
        ocean_mask = self.ax.add_feature(cfeature.OCEAN.with_scale('110m'),facecolor=facecolor,edgecolor='face',zorder=zorder)
        lake_mask = self.ax.add_feature(cfeature.LAKES.with_scale('110m'),facecolor=facecolor,edgecolor='face',zorder=zorder)

    def plot_lat_lon_lines(self,bounds,zorder=None):
        
        r"""
        Plots parallels and meridians that are constrained by the map bounds.
        
        Parameters:
        -----------
        bounds : list
            List containing map bounds.
        """

        #Get current axes if not specified
        self.ax = self._check_ax()
        
        #Retrieve bounds from list
        bound_w,bound_e,bound_s,bound_n = bounds
        
        new_xrng = abs(bound_w-bound_e)
        new_yrng = abs(bound_n-bound_s)
        
        #function to round to nearest number
        def rdown(num, divisor):
            return num - (num%divisor)
        def rup(num, divisor):
            return divisor + (num - (num%divisor))
        
        #Calculate parallels and meridians
        rthres = 20
        if new_yrng < 160.0 or new_xrng < 160.0:
            rthres = 10
        if new_yrng < 40.0 or new_xrng < 40.0:
            rthres = 5
        if new_yrng < 25.0 or new_xrng < 25.0:
            rthres = 2
        if new_yrng < 9.0 or new_xrng < 9.0:
            rthres = 1
        parallels = np.arange(rdown(bound_s,rthres),rup(bound_n,rthres)+rthres,rthres)
        meridians = np.arange(rdown(bound_w,rthres),rup(bound_e,rthres)+rthres,rthres)
        
        add_kwargs = {}
        if zorder is not None:
            add_kwargs = {'zorder':zorder}
        
        #Fix for dateline crossing
        if self.proj.proj4_params['lon_0'] == 180.0:
            
            #Recalculate parallels and meridians
            parallels = np.arange(rup(bound_s,rthres),rdown(bound_n,rthres)+rthres,rthres)
            meridians = np.arange(rup(bound_w,rthres),rdown(bound_e,rthres)+rthres,rthres)
            meridians2 = np.copy(meridians)
            meridians2[meridians2>180.0] = meridians2[meridians2>180.0]-360.0
            all_meridians = np.arange(0.0,360.0+rthres,rthres)
            all_parallels = np.arange(rdown(-90.0,rthres),90.0+rthres,rthres)
            
            #First call with no labels but gridlines plotted
            gl1 = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,xlocs=all_meridians,ylocs=all_parallels,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted',**add_kwargs)
            #Second call with labels but no gridlines
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,xlocs=meridians,ylocs=parallels,linewidth=0.0,color='k',alpha=0.0,linestyle='dotted',**add_kwargs)
            
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(meridians2)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

        else:
            #Add meridians and parallels
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted',**add_kwargs)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(meridians)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        #Reset plot bounds
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
        

    def contourf(self,lon,lat,data,*args,ax=None,transform=None,**kwargs):
        """
        Wrapper to matplotlib's contourf function. Assumes lat and lon arrays are passed instead
        of x and y arrays. Default data projection is ccrs.PlateCarree() unless a different
        data projection is passed.
        """
        
        #Get current axes if not specified
        ax = ax or self._check_ax()
        
        #Check transform if not specified
        if transform is None: transform = ccrs.PlateCarree()
        
        #Fill contour data
        cs = ax.contourf(lon,lat,data,*args,**kwargs,transform=transform)
        return cs
    
    def contour(self,lon,lat,data,*args,ax=None,transform=None,**kwargs):
        """
        Wrapper to matplotlib's contour function. Assumes lat and lon arrays are passed instead
        of x and y arrays. Default data projection is ccrs.PlateCarree() unless a different
        data projection is passed.
        """
        
        #Get current axes if not specified
        ax = ax or self._check_ax()
        
        #Check transform if not specified
        if transform is None: transform = ccrs.PlateCarree()
        
        #Contour data
        cs = ax.contour(lon,lat,data,*args,**kwargs,transform=transform)
        return cs    

    def get_dynamic(self,lon,lat):
        
        r"""
        Sets up a dynamic map extent with an aspect ratio of 3:2 given latitude and longitude bounds.
        
        Parameters:
        -----------
        lon : ndarray of floats
        lat : ndarray of floats
        
        Returns:
        --------
        list
            List containing new west, east, north, south map bounds, respectively.
        """

        min_lon,max_lon = np.amin(lon),np.amax(lon)
        min_lat,max_lat = np.amin(lat),np.amax(lat)
                
        if min_lat<0 and max_lat>0:
            projection = 'PlateCarree'
        elif min_lat>=0 and max_lat>80 and max_lon-min_lon>=345:
            projection = 'NorthPolarStereo'
        else:
            projection = 'PlateCarree'

        return projection

        #Get lat/lon bounds
        bound_w = min_lon+0.0
        bound_e = max_lon+0.0
        bound_s = min_lat+0.0
        bound_n = max_lat+0.0

        #Function for fixing map ratio
        def fix_map_ratio(bound_w,bound_e,bound_n,bound_s,nthres=1.45):
            xrng = abs(bound_w-bound_e)
            yrng = abs(bound_n-bound_s)
            diff = float(xrng) / float(yrng)
            if diff < nthres: #plot too tall, need to make it wider
                goal_diff = nthres * (yrng)
                factor = abs(xrng - goal_diff) / 2.0
                bound_w = bound_w - factor
                bound_e = bound_e + factor
            elif diff > nthres: #plot too wide, need to make it taller
                goal_diff = xrng / nthres
                factor = abs(yrng - goal_diff) / 2.0
                bound_s = bound_s - factor
                bound_n = bound_n + factor
            return bound_w,bound_e,bound_n,bound_s

        #First round of fixing ratio
        #bound_w,bound_e,bound_n,bound_s = fix_map_ratio(bound_w,bound_e,bound_n,bound_s,1.45)
        
        #Return map bounds
        #return bound_w,bound_e,bound_s,bound_n
    

    def check_res(self,res,counties=False):
        r"""
        Checks if a resolution string contains digits. If yes, then that value is returned
        and is passed into the cartopy "with_scale()" argument. If it's solely a string
        representing the type of resolution ('l','m','h'), then that value is converted to
        a resolution with digits depending on the type of boundary being plotted.
        
        Parameters:
        ----------------------
        res
            String representing the passed resolution
            
        Returns:
        ----------------------
        String representing the converted resolution
        """
        
        #If resolution contains digits (e.g., '50m'), assumed to be valid input and simply returned
        if any([i.isdigit() for i in res]):
            return res
        
        #Otherwise, attach numerical values to low, medium and high resolutions
        else:
            #Use options for everything but counties
            if counties == False:
                if res == 'l':
                    return '110m'
                elif res == 'h':
                    return '10m'
                else:
                    return '50m'
            #Use options for county resolutions
            else:
                if res == 'l':
                    return '20m'
                elif res == 'h':
                    return '500k'
                else:
                    return '5m'

    def _check_ax(self):
        r"""
        Adapted from Basemap - checks to see if an axis is specified, if not, returns plt.gca().
        """
        
        if self.ax is None:
            ax = plt.gca(projection=self.proj)
        else:
            ax = self.ax #(projection=self.proj)
            
        return ax
    


    def _generate_cone(self,FORECAST,grid_res = 0.02):
        
        r"""
        Generates a cone of uncertainty using forecast and spread data.
        
        Parameters:
        -----------
        forecast : dict
            Dictionary containing forecast data
        
        """
    
        #Function for interpolating between 2 times
        def temporal_interpolation(value, orig_times, target_times):
            f = interp.interp1d(orig_times,value)
            ynew = f(target_times)
            return ynew
    
        #Function for plugging small array into larger array
        def plug_array(small,large,small_coords,large_coords):
    
            small_lat = np.round(small_coords['lat'],2)
            small_lon = np.round(small_coords['lon'],2)
            large_lat = np.round(large_coords['lat'],2)
            large_lon = np.round(large_coords['lon'],2)
    
            small_minlat = min(small_lat)
            small_maxlat = max(small_lat)
            small_minlon = min(small_lon)
            small_maxlon = max(small_lon)
    
            if small_minlat in large_lat:
                minlat = np.where(large_lat==small_minlat)[0][0]
            else:
                minlat = min(large_lat)
            if small_maxlat in large_lat:
                maxlat = np.where(large_lat==small_maxlat)[0][0]
            else:
                maxlat = max(large_lat)
            if small_minlon in large_lon:
                minlon = np.where(large_lon==small_minlon)[0][0]
            else:
                minlon = min(large_lon)
            if small_maxlon in large_lon:
                maxlon = np.where(large_lon==small_maxlon)[0][0]
            else:
                maxlon = max(large_lon)
    
            large[minlat:maxlat+1,minlon:maxlon+1] = small
    
            return large
    
        #Function for finding nearest value in an array
        def findNearest(array,val):
            return array[np.abs(array - val).argmin()]
    
        #Function for adding a radius surrounding a point
        def add_radius(lats,lons,vlat,vlon,rad):
    
            #construct new array expanding slightly over rad from lat/lon center
            grid_fac = rad*4
    
            #Make grid surrounding position coordinate & radius of circle
            nlon = np.arange(findNearest(lons,vlon-grid_fac),findNearest(lons,vlon+grid_fac+grid_res),grid_res)
            nlat = np.arange(findNearest(lats,vlat-grid_fac),findNearest(lats,vlat+grid_fac+grid_res),grid_res)
            lons,lats = np.meshgrid(nlon,nlat)
            return_arr = np.zeros((lons.shape))
    
            #Calculate distance from vlat/vlon at each gridpoint
            dlat = np.subtract(lats,vlat)
            dlon = np.subtract(lons,vlon)
            dist = np.sqrt(dlat**2+dlon**2)
    
            #Mask out values less than radius
            return_arr[dist <= rad] = 1
    
            #Attach small array into larger subset array
            small_coords = {'lat':nlat,'lon':nlon}
    
            return return_arr, small_coords
        
        #--------------------------------------------------------------------
    
        cone_size = FORECAST['spread']
        cone_climo_hr = [(t-FORECAST['dates'][0]).days for t in FORECAST['dates']]
            
        fcst_lon = FORECAST['x']
        fcst_lat = FORECAST['y']
        fhr = cone_climo_hr
        t = np.array(fhr)
        interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.2)
    
        #Interpolate forecast data temporally and spatially
        interp_kind = 'linear'
        x1 = interp.interp1d(t,fcst_lon,kind=interp_kind)
        y1 = interp.interp1d(t,fcst_lat,kind=interp_kind)
        interp_fhr = interp_fhr_idx
        interp_lon = x1(interp_fhr_idx)
        interp_lat = y1(interp_fhr_idx)
        
        idxs = np.nonzero(np.in1d(np.array(fhr),np.array(cone_climo_hr)))
        temp_arr = np.array(cone_size)[idxs]
        interp_rad = np.apply_along_axis(lambda n: temporal_interpolation(n,fhr,interp_fhr),axis=0,arr=temp_arr)
    
        #Initialize 0.05 degree grid
        gridlats = np.arange(-4,4+grid_res,grid_res)
        gridlons = np.arange(-4,4+grid_res,grid_res)
        gridlons2d,gridlats2d = np.meshgrid(gridlons,gridlats)
    
        #Iterate through fhr, calculate cone & add into grid
        large_coords = {'lat':gridlats,'lon':gridlons}
        griddata = np.zeros((gridlats2d.shape))
        for i,(ilat,ilon,irad) in enumerate(zip(interp_lat,interp_lon,interp_rad)):
            temp_grid, small_coords = add_radius(gridlats,gridlons,ilat,ilon,irad)
            plug_grid = np.zeros((griddata.shape))
            plug_grid = plug_array(temp_grid,plug_grid,small_coords,large_coords)
            griddata = np.maximum(griddata,plug_grid)
    
        return_dict = {'lat':gridlats,'lon':gridlons,'lat2d':gridlats2d,'lon2d':gridlons2d,'cone':griddata,
                       'center_lon':interp_lon,'center_lat':interp_lat}
        return return_dict

    
    def _phase_space(self,ax=None,plot_type='MJO',axlim=4):

        r"""
        Generates a 2-dimensional phase space with labeling and phase 
        deliminators for either MJO or GWO.
        
        """
        
        ax.axis([-1*axlim,axlim,-1*axlim,axlim])
        ax.set_xticks(np.arange(-1*axlim,axlim+1,1))
        ax.set_yticks(np.arange(-1*axlim,axlim+1,1))
        mlib.rcParams.update({'font.size': 18,'font.serif':'Times New Roman'})
        # Draw amp-1 circle
        ax.plot(np.cos(np.linspace(0,2*np.pi,1000)),np.sin(np.linspace(0,2*np.pi,1000)),'k')
        
        for phase in range(8):
            theta=(phase+4)*np.pi/4.
            # Label phase numbers
            plt.text(0.95*axlim*np.cos(theta+np.pi/8.),0.95*axlim*np.sin(theta+np.pi/8.),
                     str(phase+1), ha='center', va='center', fontweight='bold',fontname='Georgia',fontsize=22, backgroundcolor='w')
            # Rotate for GWO plot
            theta=theta+np.pi/8.*(plot_type=='GWO')
            # Draw stage lines
            ax.plot([np.cos(theta),2*axlim*np.cos(theta)],[np.sin(theta),2*axlim*np.sin(theta)],'k--')
    
        # Add phase descriptions
        if plot_type=='MJO':
            labs=['Maritime\nContinent','Western\nPacific','West. Hem.\nand Africa','Indian\nOcean']
            ax.set_xlabel('RMM1',fontsize=16)
            ax.set_ylabel('RMM2',fontsize=16)
        if plot_type=='GWO':
            labs=['','high AAM','','low AAM']
            ax.set_xlabel('dM/dt',fontsize=16)
            ax.set_ylabel('Global relative AAM anomaly (M)',fontsize=16)
        for i,tx in enumerate(labs):
            theta=i*np.pi/2.
            txrot=[0,90,0,270][i-1]
            plt.text(0.88*axlim*np.cos(theta),0.88*axlim*np.sin(theta),tx, 
                     ha='center', va='center', rotation=txrot, backgroundcolor='w') 



