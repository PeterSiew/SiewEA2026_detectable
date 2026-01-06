import xarray as xr
import numpy as np
import ipdb
from importlib import reload
import xesmf as xe
import datetime as dt
import pandas as pd

### Read ERA5 data
#path='/mnt/data/data_a/ERA5/T2M_daily/T2M_daily-*.nc'
#data_raw=xr.open_mfdataset(path, chunks={'time':500},decode_times=True)
years=range(1940,1950)
years=range(1940,2025)
years=[1880,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020]


datas=[]
for year in years:
    filename='/mnt/data/data_a/Berkeley_SAT_land/Complete_TAVG_Daily_LatLong1_%s.nc'%year
    data_raw=xr.open_dataset(filename, chunks={'time':50}, decode_times=False)
    data=data_raw['temperature']
    data_year=data_raw.year.values
    data_month=data_raw.month.values
    data_day=data_raw.day.values
    new_time=[]
    for i, yr in enumerate(data_year):
        new_time.append(dt.date(int(yr),int(data_month[i]),int(data_day[i])))
    new_time=pd.to_datetime(new_time)
    #new_time = pd.date_range(start='%s-%s-%s'%(, end='2021-12-31', freq='D') 
    data=data.assign_coords({'time':new_time})
    ## Select 30N to 90N
    data=data.sel(latitude=slice(30,90))
    datas.append(data)


datas=xr.concat(datas,dim='time')
datas=datas.chunk({'time':50})

### Start Regridding
lats = np.arange(30.5,90,1)
lons = np.arange(0.5,360,1).tolist()
ds_out = xr.Dataset({"latitude": (["latitude"],lats), "longitude": (["longitude"],lons),}) 
regridder= xe.Regridder(datas.to_dataset(name='hihi'),ds_out,"bilinear",unmapped_to_nan=True, ignore_degenerate=True) 
datas_regrid=regridder(datas)

### Save the data
datas_regrid=datas_regrid.rename('t2m')
datas_regrid.to_netcdf('/mnt/data/data_a/Berkeley_SAT_land/BESTSAT_daily-1880Jan_2022July_1x1.nc')

