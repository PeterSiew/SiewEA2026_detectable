import xarray as xr
import numpy as np
import ipdb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import scipy
import xesmf as xe
from multiprocessing import Process

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import sys;sys.path.insert(0, '/Users/home/siewpe/codes/temp_variability/figures/')
import create_training_tempsd as ctt
import create_timeseries as ct
import tools


if __name__ == "__main__":

    reload(tools)
    reload(ctt)

    if False: ### DAMIP experiments
        vars=['CESM2_histGHG_tas_daily_en']; vars=['CESM2_histaer_tas_daily_en']; vars=['CESM2_histBMB_tas_daily_en']; vars=['CESM2_histEE_tas_daily_en']
        vars=['CanESM5_histnat_tas_daily_en']; vars=['CanESM5_histaer_tas_daily_en']; vars=['CanESM5_histGHG_tas_daily_en']; vars=['MIROC6_histnat_tas_daily_en']
        vars=['MIROC6_histaer_tas_daily_en']; vars=['MIROC6_histGHG_tas_daily_en'] # en 42 has problem; ###
        vars=['HadGEM3_histGHG_tas_daily_en']; vars=['HadGEM3_histaer_tas_daily_en']; vars=['HadGEM3_histnat_tas_daily_en']
        ensembles=[range(1,16)] # For CESM2 DAMIP (GHG)
        ensembles=[range(1,26)] # For CanESM5 (Nat)
        ensembles=[range(1,11)] # For CanESM5 (GHG and Aer)
        ensembles=[range(1,11)] # For MIROC6 DAMIP (Aer)
        ensembles=[range(1,51)] # For MIROC6 DAMIP (GHG and nat)
        ensembles=[[i for i in range(1,61) if i not in [6,7,8,9,10]]] # For HadGEM hist-GHG and hist-aer
        ensembles=[[i for i in range(1,61) if i not in [31,32,37]]] # For HaGEM hist-nat

    if False: # SLP standard deviation
        vars=['SLP_regrid_1x1', 'cesm1_psl_daily_en','canesm2_psl_daily_en','gfdlcm3_psl_daily_en','mk360_psl_daily_en']
        vars=['ERA5_MSLP_daily']
        ensembles=[[''], range(2,41), range(1,51), range(1,21), range(1,31)]
        ensembles=[range(1,17)] # For EC-Earth (16 members)
        years = [range(1980,2022), range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100)]
        years=[range(1836,2015)]
        save_folder='training_pslsd'

    # For best_SAT, there are missing data from 1925 to 1929
    #vars=['canesm2_tas_daily_fake_en']
    vars=['ERA5_T2M_daily_regrid_1x1','T2M_regrid_1x1','best_SAT','noaa_20CR_T2M_daily']
    vars=['amip_ECHAM5_daily_climsic_tas_en', 'amip_ECHAM5_daily_tas_en']
    vars=['ERA5_T2M_daily_regrid_1x1']
    vars1=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en']
    vars2=['amip_WACCM6_daily_tas_en', 'amip_WACCM6_climsic_daily_tas_en']
    vars3=['ERA5_T2M_daily_regrid_1x1', 'BESTSAT_daily_regrid_1x1']
    vars=vars1+vars2+vars3
    vars=['cesm1_tas_daily_PI','canesm2_tas_daily_PI'] # CESM1-PI problem has some problem - missing data
    vars=['canesm2_tas_daily_PI']

    ensembles=[[''], [''], [''],[''], range(2,41),range(1,51),range(1,31),range(1,21),range(1,31),range(1,17)]
    ensembles=[range(2,3),range(1,2),range(1,2),range(1,2),range(1,2)]
    ensembles=[range(2,22),range(1,21),range(1,21),range(1,21),range(1,21)]
    ensembles = [range(1,31), range(1,51)] # ECHAM5 two sets of AMIP
    ensembles=[['']]
    ensembles=[range(1,2)]
    ensembles1=[range(1,41),range(1,51),range(1,31),range(1,21),range(1,31),range(1,17)]
    ensembles2=[range(1,31), range(1,31)] # For WACCM6 two sets of AMIP
    ensembles3=[[''],['']]
    ensembles=ensembles1+ensembles2+ensembles3
    ensembles = [[''],['']] # The CESM1 and CanESM2 PI-control
    ensembles = [['']] # The CanESM2-PI


    years=[range(1950,2100)] # for forced similation
    years=[range(1940,2023)] # for CESM2 DAMIP
    years=[range(1940,2021)] # for CanESM5 DAMIP
    years=[range(1850,2100)] # for MIROC6 DAMIP (full record)
    years=[range(1850,2020)] # for HadGEM3 DAMIP (the 2020 DJF is wrong - because it only has D without J and F in 2021)
    years = [range(1940,2022),range(1980,2020),range(1930,2021),range(1836,2015),
            range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100)] # for obs and models (tas_daily)
    years = [range(1979,2018),range(1979,2018)]
    years=[range(1940,2024)]
    years = [range(2015,3111)] 
    years = [range(1950,2100)]
    years1=[range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100),range(1950,2100)]
    years2=[range(1979,2014), range(1979,2014)] # WACCM6 experiment
    years3=[range(1940,2024), range(1940,2022)] # ERA5, BESTSAT
    years=years1+years2+years3
    years = [range(402,2201), range(2015,3111)] # Pre-industrial for CESM2 and CanESM
    years = [range(2015,3111)] # PI-control for CanESM2

    if False:
        ps=[]
        for i,var in enumerate(vars):
            ens_list=ensembles[i]
            year_list=years[i]
            p=Process(target=ctt.create_training, args=(var,ens_list,year_list))
            p.start()
            ps.append(p)
        for p in ps: # We need that for waiting
            p.join()
    else: # Debug
        for i, var in enumerate(vars):
            ctt.create_training(vars[i],ensembles[i],years[i])


def create_training(var,ensembles,years):

    lons = np.arange(-177,180,3).tolist(); lons.remove(0) 
    ds_out = xr.Dataset({"latitude": (["latitude"], np.arange(35,90,3)), "longitude": (["longitude"],lons),}) # 20 to 90N; -177 to 177

    if True:
        lons = np.arange(-179.5,180,1).tolist()
        ds_out = xr.Dataset({"latitude": (["latitude"], np.arange(34.5,90,1)), "longitude": (["longitude"],lons),}) # For a more smoother fingerprint

    mons = {'SON':[9,10,11],'DJF':[12,1,2],'NDJF':[11,12,1,2],'ONDJF':[10,11,12,1,2],'SONDJF':[9,10,11,12,1,2]}
    seasons=['DJF','SON']
    seasons=['SON','DJF','NDJF']
    seasons=['ONDJF','SONDJF']
    seasons=['SON','DJF','NDJF','ONDJF','SONDJF']
    seasons=['SON','DJF','NDJF','ONDJF','SONDJF']
    seasons=['SON','DJF','SONDJF']
    extra_name=''

    for en in ensembles:
        ## Regrid first
        data_raw=tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False)
        #ipdb.set_trace()
        data_raw=data_raw.sel(latitude=slice(30,90))
        print("Start to compute")
        #data_raw=data_raw.chunk(chunks='auto'); #data_raw = data_raw.chunk(chunks={})
        data_raw=data_raw.chunk({'time':50}) # smaller is better for ERA5
        data_raw=data_raw.compute()
        print("Finish the compute")
        regridder=xe.Regridder(data_raw.to_dataset(name='hihi'),ds_out,"bilinear",unmapped_to_nan=True, ignore_degenerate=True) 
        data_raw=regridder(data_raw)
        ## Should remove the seasonal cycle (for 1-Jan to 31-Dec data)
        #data_raw=tools.remove_seasonal_cycle_and_detrend(data_raw, detrend=False)
        data_anom=tools.remove_seasonal_cycle_simple(data_raw, detrend=False)
        print(var,en)
        for yr in years:
            for season in seasons:
                mon_mask = data_anom.time.dt.month.isin(mons[season])
                data = data_anom.sel(time=mon_mask)
                data = data.rename('training')
                if season=='SON':
                    data_sel = data.sel(time=slice('%s-09-01'%yr,'%s-11-30'%yr))
                elif season=='DJF': 
                    # Be careful that some data has up to 30 day in Feb (HadGCM in DAMIP)
                    data_sel = data.sel(time=slice('%s-12-01'%yr,'%s-02-28'%(yr+1))) 
                elif season=='NDJF':
                    data_sel = data.sel(time=slice('%s-11-01'%yr,'%s-02-28'%(yr+1)))
                elif season=='ONDJF':
                    data_sel = data.sel(time=slice('%s-10-01'%yr,'%s-02-28'%(yr+1)))
                elif season=='SONDJF':
                    data_sel = data.sel(time=slice('%s-09-01'%yr,'%s-02-28'%(yr+1)))
                if True: # for temp_var: sub-seasonal temperature variability 
                    save_folder='training_tempsd'
                    data_std=data_sel.std(dim='time')
                    data_std.to_netcdf('/mnt/data/data_a/t2m_variability_training/%s/%s/%s%s_%s%s.nc'%(save_folder,season,var,en,yr,extra_name))
                if True: # for AA:  the mean temp over the season (here we try to remove the seasonal cycle first)
                    save_folder='training_AA'
                    data_mean=data_sel.mean(dim='time')
                    data_mean.to_netcdf('/mnt/data/data_a/t2m_variability_training/%s/%s/%s%s_%s%s.nc'%(save_folder,season,var,en,yr,extra_name))
                else: # Get the heat and cold extremes (the last 5% at each grid point)
                    pass
                    #q95 = data.quantile(0.95,dim='time')
                    #q05 = data.quantile(0.05,dim='time')
                    #data_heat_extreme = xr.where(data>q95, data, np.nan).mean(dim='time')
                    #data_cold_extreme = xr.where(data<q05, data, np.nan).mean(dim='time')
                    # Save the regrid file
                    #data_heat_extreme.to_netcdf('/dx02/pyfsiew/training/training_heatextreme/%s/%s%s_%s%s.nc'
                    #                   %(season,var,en,yr,extra_name))
                    #data_cold_extreme.to_netcdf('/dx02/pyfsiew/training/training_coldextreme/%s/%s%s_%s%s.nc'
                    #                    %(season,var,en,yr,extra_name))

