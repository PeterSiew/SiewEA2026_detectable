import xarray as xr
import numpy as np
import ipdb
from importlib import reload
import xesmf as xe
import pandas as pd

### Read ERA5 data
years=range(1940,2025)
datas=[]

var='T2M'
var_name='t2m'

var='MSLP'
var_name='msl'

for year in years:
    path='/mnt/data/data_a/ERA5/%s_daily/'%var
    filename='%s_daily-%s.nc'%(var,year)
    #data=xr.open_dataset(path+filename, chunks={'time':500}, decode_times=True)['t2m']
    data=xr.open_dataset(path+filename, chunks={}, decode_times=True)[var_name]
    print(year)
    ## Select 30N to 90N
    data=data.sel(latitude=slice(90,30))
    if 'valid_time' in data.dims:
        data=data.rename({'valid_time':'time'})
    datas.append(data)

#ipdb.set_trace()
datas=xr.concat(datas,dim='time')
datas=datas.sel(time=slice('1940-01-01','2024-03-31'))
datas=datas.chunk({'time':5000})

### Start Regridding
lats = np.arange(30.5,90,1)
lons = np.arange(0.5,360,1).tolist()
ds_out = xr.Dataset({"latitude": (["latitude"],lats), "longitude": (["longitude"],lons),}) 
regridder= xe.Regridder(datas.to_dataset(name='hihi'),ds_out,"bilinear",unmapped_to_nan=True, ignore_degenerate=True) 
datas_regrid=regridder(datas)

### Save ERA5 regrid data
datas_regrid=datas_regrid.rename(var_name)
new_time=pd.date_range(start='1940-01-01', end='2024-03-31', freq='D') 
datas_regrid=datas_regrid.assign_coords({'time':new_time})
datas_regrid.to_netcdf('/mnt/data/data_a/ERA5/%s_daily/%s_daily-1940Jan_2024Mar_1x1.nc'%(var,var))

