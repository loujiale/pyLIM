# LIM_CPC

A python-based linear inverse modeling suite.

**LIM_CPC** is based on the linear inverse model (LIM) described by Penland & Sardeshmukh (1995).
This package provides the machinery to both calibrate and forecast/integrate a LIM.

## Installation
LIM_CPC requires Python 3.6+

LIM_CPC can be installed by cloning the GitHub repository:

```sh
git clone https://github.com/splillo/LIM_CPC
cd LIM_CPC
python setup.py install
```

## Dependencies
- matplotlib >= 2.2.2
- numpy >= 1.14.3
- scipy >= 1.1.0
- basemap >= 1.1.0
- netCDF4 >= 1.4.2
- xarray >= 0.8.0
- global-land-mask

Create and activate python environment with dependencies. Called limcpc.
```sh
cd LIM_CPC
conda env create -f environment.yml
conda activate limcpc
```

## User Guide
<h3>namelist.py</h3>  
  
*Set LIM and data specifications*  
```python
# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
climoyears = (1979,2017)

# Variable and EOF object file prefix
VAR_FILE_PREFIX = '/path/to/files/variable_file_prefix_'
EOF_FILE_PREFIX = '/path/to/files/eof_file_prefix_'
```
  
*Set variables to save for use in LIMs.  
Dictionary of variable names, each containing a dictionary with  
'info' and 'data'. Set the elements of 'info' here.  
For each variable, 'info' contains all the input arguments for the dataset.*  
```python
use_vars = {
            'H500':
                {'info':('/path/to/H500/files/','gh',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'T2m':
                {'info':('/path/to/T2m/files/','air',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'season0':True,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True})},
            }

```
  
*Set EOF truncations for variables.  
Dictionary keys refer to the variable (or variables within a tuple for a combined EOF).  
Dictionary values refer to the respective EOF truncation.  
Keys in eof_trunc dictionary can be integers refering to month of the year, or strings simply labeling the LIM space in which the full period of the specified variables are used for EOFs and LIM training.*  

```python
eof_trunc = {
             1: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
             2: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
             3: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
             ...
             12: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
             }
```  
notice in the example above, each month is identical. You can vary the EOFs and truncations in each month. Or you can simplify the eof_trunc dictionary in this example if you want the same variables and trunctions for each month:  
```python
eof_trunc = {
             mn: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5} \
             for mn in range(1,13)
            }
```  
for a LIM in which you want to use the full period of the specified variables, use a string as the key in the eof_trunc dictionary:  
```python
eof_trunc = {
            'fullyr': {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5}
            }
```

<h3>Using the LIM driver</h3>  
  
- import LIM_CPC package and initialize the Driver with the namelist python file. The namelist file can be located anywhere. Purpose is to allow multiple namelist files to be saved, with details for different LIMs, and referenced at any time. If not in same directory, need to include full path.  
```python
from LIM_CPC import driver
LIMdriver = driver.Driver("namelist.py")
```
- compile variable data, using the information from the use_vars dictionary in namelist.py, to use for training and cross-validating a LIM. Use read=False for new variable objects, or read=True to read in previously saved pickle files containing variables objects (default). Pickle files will read / save with the path and file prefix string VAR_FILE_PREFIX set in namelist.py.
```python
LIMdriver.get_variables(read=True)
```
- save new EOF dataset objects, consistent with the variables or variable combinations listed in eof_trunc set in namelist.py. Use read=False for new EOF objects, or read=True to read in previously saved pickle files containing EOF objects (default). Pickle files will read / save with the path and file prefix string EOF_FILE_PREFIX set in namelist.py.
```python
LIMdriver.get_eofs(read=True)
```

-------  
- cross_validation — For skill assessment and hindcasts.  
```python
LIMdriver.cross_validation(num_folds=10,lead_times=np.arange(1,29))
```  
num_folds : int  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of times data is subset and model is trained. Default is 10,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Which means each iteration trains on 90% of data, forecasts run on 10% of the data.  
lead_times : list, tuple, ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Lead times (in days) to integrate model for output.  
  
*Returns*  
model_F : dict  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dictionary of model forecast output, with keys corresponding to valid time  
model_E : dict  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dictionary of model error output, with keys corresponding to valid time  


----------  
- prep_realtime_data — Compile realtime data, interpolate to same grid as LIM, and convert into PCs.  
```python
LIMdriver.prep_realtime_data()
```  
*Assigns*  
Variable data, latitude, longitude, and time to RT_VARS dictionary.

----------  
- get_model — Either read in a pickled model or train a new model, with the option to save.  
```python
LIMdriver.get_model(limkey,load_file=None,save_file=None)
```  
limkey : int or str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; int: month of data to train LIM on, str: label of LIM.  
load_file : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Filename for pickle file containing model to load.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Default is None, in which case a new model is trained.  
save_file : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Filename of pickle file to save new model to.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Default is None, in which case the model is not saved.  
  
*Returns*  
model : model object  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Object of trained model

----------  
- run_forecast — Run forecasts with LIM using real-time data, initialized from time t_init with output at the specified lead_times.  
```python
LIMdriver.run_forecast(t_init,lead_times=np.arange(1,29))
```  
t_init : datetime object  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Date of initialization for the forecast  
lead_times : list, tuple, or ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lead_times in increment of data to integrate model forward.
  
*Returns*  
model_F : dict  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dictionary of model forecast output, with keys corresponding to valid time.  
model_E : dict  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dictionary of model error output, with keys corresponding to valid time.  

----------  
- plot_map — Plots maps from PCs.  
```python
LIMdriver.plot_map(varname='T2m',t_init=None,lead_times=None,prop={})
```  
varname : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Must be a variable name in the list of keys used in the LIM.  
prop : dict  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Customization properties for plotting.  

----------  
- plot_teleconnection — Plots teleconnection timeseries analysis, forecast, and spread.  
```python
LIMdriver.plot_teleconnection(list_of_teleconnections = ['nao', 'ea', 'wp', 'epnp', 'pna', 'eawr', 'scand', 'tnh', 'poleur'],daysback=120)
```  
list_of_teleconnections : list  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; List of names (str) of teleconnections to plot.  
daysback : int  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of days prior to t_init to plot analysis.  

----------  
- plot_timelon — Plots meridionally averaged data with longitude on x-axis and time on y-axis.  
```python
LIMdriver.plot_timelon(varname='colIrr',t_init=None,daysback=120,lat_bounds=(-7.5,7.5),save_to_file=None,prop={})
```  
varname : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Name of variable (consistent with use_var dictionary keys) to plot. Default is 'colIrr'  
t_init : datetime object  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Time of forecast initialization. Default is most recent time available.  
daysback : int  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of days prior to t_init to plot hovmoller  
lat_bounds : tuple or list  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Latitude boundaries to average data between. Default is (-7.5,7.5)  
save_to_file : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Name of file to save figure to. Default is None, which does not save the figure  

----------  
- plot_timelat — Plots meridionally averaged data with longitude on x-axis and time on y-axis.  
```python
LIMdriver.plot_timelat(varname,t_init=None,daysback=120,lon_bounds=(0,360),save_to_file=None,prop={})
```  
varname : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Name of variable (consistent with use_var dictionary keys) to plot.
t_init : datetime object  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Time of forecast initialization. Default is most recent time available.  
daysback : int  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Number of days prior to t_init to plot hovmoller  
lat_bounds : tuple or list  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Latitude boundaries to average data between. Default is (-7.5,7.5)  
save_to_file : str  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Name of file to save figure to. Default is None, which does not save the figure  

