r"""
Namelist for use in training and running a LIM

Sam Lillo
"""

# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
datebounds = ('1/1','12/31')
climoyears = (1979,2017)

#Set variables to save for use in LIMs. 
#Dictionary of variable names, each containing a tuple with the path to the data, 
#the variable name in the file, and a dictionary with specifications. 
use_vars = {
            'H100':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'H500':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'SLP':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/mslp/','mslp',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'T2m':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data/JRA_t2m/','TMP_GDS0_HTGL',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True}),
            'colIrr':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/colirr/','colIrradiance',
                                        {'latbounds':(-20,20),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,}),
            'SF750':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/stream/','stream',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'SOIL':
                ('/Volumes/time machine backup/soilmoisture/','soilw',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True}),
}

extra = {
            'SOIL':
                ('/Volumes/time machine backup/soilmoisture/','soilw',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True}),
            'H100':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'H500':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'SLP':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/mslp/','mslp',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
            'T2m':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data/JRA_t2m/','TMP_GDS0_HTGL',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True}),
            'colIrr':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/colirr/','colIrradiance',
                                        {'latbounds':(-20,20),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,}),
            'SF750':
                ('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/stream/','stream',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5}),
}