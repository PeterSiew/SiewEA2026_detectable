import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt; import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import xesmf as xe
import scipy

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools


if __name__ == "__main__":

    if False: # WACCM6 Exp1 (obs ice)
        ensembles=range(1,31) # WACCAM
        ensembles=range(1,11) # WACCAM
        new_time = pd.date_range(start='1979-01-01', end='2014-12-31', freq='D')
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
        psl_ratio=1
        psl_path, tas_path = {}, {}
        for en in ensembles:
            psl_path[en]='/mnt/data/data_a/liang_greenice/waccm6/slp_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
            tas_path[en]='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
        psl_var='PSL'
        tas_var='TREFHT'
        years=[i for i in range(1979,2014)] # For WACCAM
    if False: # WACCM6 Exp2 (fixed ice)
        ensembles=range(1,31) # WACCAM
        ensembles=range(1,11) # WACCAM
        new_time = pd.date_range(start='1979-01-01', end='2014-12-31', freq='D')
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
        psl_ratio=1
        psl_path, tas_path = {}, {}
        for en in ensembles:
            psl_path[en]='/mnt/data/data_a/liang_greenice/waccm6/slp_daily/exp2_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
            tas_path[en]='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp2_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
        psl_var='PSL'
        tas_var='TREFHT'
        years=[i for i in range(1979,2014)] # For WACCAM
    if False: # CESM1
        #ensembles=range(1,17) # NCAR CESM1
        ensembles=range(1,17) # NCAR CESM1
        ensembles=range(1,4) # just for testing
        new_time = pd.date_range(start='1920-01-01', end='2100-12-31', freq='D')  
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
        psl_ratio=1
        psl_path, tas_path = {}, {}
        for en in ensembles:
            psl_path[en]='/mnt/data/data_a/CMIP5_LENS/CESM/psl_daily/psl_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
            tas_path[en]='/mnt/data/data_a/CMIP5_LENS/CESM/tas_daily/tas_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%en
        psl_var='psl'
        tas_var='tas'
        years=[i for i in range(1979,2024)] 
        early_years=[i for i in range(1979,2001)]
        late_years=[i for i in range(2001,2024)]
    if False: # CanESM2
        ensembles=range(1,17) 
        ensembles=range(1,5)
        new_time = pd.date_range(start='1950-01-01', end='2100-12-31', freq='D') # CanESM2
        mask = ~((new_time.month==2) & (new_time.day==29)); new_time = new_time[mask]
        psl_ratio=1
        psl_path, tas_path = {}, {}
        for en in ensembles:
            psl_path[en]='/mnt/data/data_a/CMIP5_LENS/CanESM2/psl_daily/psl_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%en
            tas_path[en]='/mnt/data/data_a/CMIP5_LENS/CanESM2/tas_daily/tas_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%en
        psl_var='psl'
        tas_var='tas'
        years=[i for i in range(1979,2024)] 
        early_years=[i for i in range(1979,2001)]
        late_years=[i for i in range(2001,2024)]
    if False: # Combining four models which have SLP data
        vars=['cesm1','canesm2','mk360','ecearth']; repeat_no=10
        vars=['cesm1','canesm2','mk360','ecearth']; repeat_no=2
        vars_full=np.repeat(vars,repeat_no)
        vars_ens=[[i for i in range(1,repeat_no+1)] for var in vars]; vars_ens=np.array(vars_ens).flatten()
        psl_path, tas_path = {}, {}
        ensembles=[]
        for i, var in enumerate(vars_full):
            en=i
            ensembles.append(en)
            var_en=vars_ens[i]
            if var=='cesm1':
                psl_path[en]='/mnt/data/data_a/CMIP5_LENS/CESM/psl_daily/psl_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%var_en
                tas_path[en]='/mnt/data/data_a/CMIP5_LENS/CESM/tas_daily/tas_day_CESM1-CAM5_historical_rcp85_r%si1p1_19200101-21001231.nc'%var_en
            if var=='canesm2':
                psl_path[en]='/mnt/data/data_a/CMIP5_LENS/CanESM2/psl_daily/psl_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%var_en
                tas_path[en]='/mnt/data/data_a/CMIP5_LENS/CanESM2/tas_daily/tas_day_CanESM2_historical_rcp85_r%si1p1_19500101-21001231.nc'%var_en
            if var=='mk360':
                psl_path[en]='/mnt/data/data_a/CMIP5_LENS/CSIRO-Mk3-6-0/psl_daily/psl_day_CSIRO-Mk3-6-0_historical_rcp85_r%si1p1_185001-210012.nc'%var_en
                tas_path[en]='/mnt/data/data_a/CMIP5_LENS/CSIRO-Mk3-6-0/tas_daily/tas_day_CSIRO-Mk3-6-0_historical_rcp85_r%si1p1_185001-210012.nc'%var_en
            if var=='ecearth':
                psl_path[en]='/mnt/data/data_a/CMIP5_LENS/EC-EARTH/psl_daily/psl_day_EC-EARTH_historical_rcp85_r%si1p1_1860101-21001231.nc'%var_en
                tas_path[en]='/mnt/data/data_a/CMIP5_LENS/EC-EARTH/tas_daily/tas_day_EC-EARTH_historical_rcp85_r%si1p1_1860101-21001231.nc'%var_en
        psl_var='psl'
        tas_var='tas'
        years=[i for i in range(1979,2024)] 
        early_years=[i for i in range(1979,2001)]
        late_years=[i for i in range(2001,2024)]
    if False: # for ERA5
        ensembles=['']
        new_time = pd.date_range(start='1940-01-01', end='2024-12-31', freq='D') # Acually it is not used so for
        psl_ratio=1
        psl_path, tas_path = {}, {}
        for en in ensembles:
            tas_path[en]='/mnt/data/data_a/ERA5/T2M_daily/T2M_daily-1940Jan_2024Mar_1x1.nc'
            psl_path[en]='/mnt/data/data_a/ERA5/MSLP_daily/MSLP_daily-1940Jan_2024Mar_1x1.nc'
        psl_var='msl'
        tas_var='t2m'
        years=[i for i in range(1979,2024)]
        shading_level_grid=np.linspace(-1.2,1.2,13)
        shading_level_grid=np.linspace(-0.9,0.9,13)

    ###
    mons=['09','10','11']; mon_last_day='30' #SON
    mons=['12','01','02']; mons_values=[12,1,12]; mon_last_day='28' #DJF
    mons=['09','10','11','12','01','02']; mons_values=[9,10,11,12,1,2]; mon_last_day='28' #SONDJF

    ### Read TAS, SLP data and Regrid
    lons = np.arange(-177,180,3).tolist(); lons.remove(0) 
    ds_out = xr.Dataset({"latitude": (["latitude"], np.arange(35,90,3)), "longitude": (["longitude"],lons),}) # 20 to 90N; -177 to 177 (Note that CanESM5 data only go to 87.N)
    ## Get land mask
    data = xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/DJF/%s%s_%s.nc'%('BESTSAT_daily_regrid_1x1','',1980))['training']
    land_mask = ~np.isnan(data)
    psl_data_raws, tas_data_raws={}, {}
    tasAA_data_raws={}
    chunk=2000
    for en in ensembles:
        print(tas_path[en])
        ## Read TAS data
        tas_data_raw=xr.open_mfdataset(tas_path[en], chunks={'time':None})[tas_var]
        tas_data_raw=tas_data_raw.chunk(chunks={'time':chunk})
        tas_data_raw=tas_data_raw.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%(years[-1]+1)))
        if 'lat' in tas_data_raw.dims:
            tas_data_raw=tas_data_raw.rename({'lat':'latitude', 'lon':'longitude'})
        if tas_data_raw.latitude[0].item()>tas_data_raw.latitude[-1].item(): # For ERA5 
            tas_data_raw=tas_data_raw.isel(latitude=slice(None, None, -1)) 
        tas_data_raw=tas_data_raw.sel(latitude=slice(40,90)) # To make sure that 80 latitude has value for regrid
        ## Select mon values
        mon_mask=tas_data_raw.time.dt.month.isin(mons_values)
        tas_data_raw=tas_data_raw.sel(time=mon_mask)
        ## Compute
        tas_data_raw=tas_data_raw.compute()
        ## Remove seasonal cycle
        tas_data_raw=tools.remove_seasonal_cycle_simple(tas_data_raw, detrend=False)
        ## Regrid 
        #regridder=xe.Regridder(tas_data_raw.to_dataset(name='hihi'),ds_out,"bilinear",unmapped_to_nan=True, ignore_degenerate=True) 
        regridder=xe.Regridder(tas_data_raw.to_dataset(name='hihi'),ds_out,"bilinear",ignore_degenerate=True) 
        tas_data_raw=regridder(tas_data_raw)
        ## Save the TAS data before land mask (for Arctic Amplification)
        tasAA_data_raws[en]=tas_data_raw.copy()
        ## Apply the land_mask on TAS after regridding
        tas_data_raw= xr.where(land_mask, tas_data_raw, np.nan)
        tas_data_raw=tas_data_raw.transpose('time','latitude','longitude') # Reoder back to the correct dims' order
        tas_data_raws[en]=tas_data_raw
        ## Read PSL data (no land mask)
        psl_data_raw=xr.open_mfdataset(psl_path[en], chunks={'time':None})[psl_var]
        psl_data_raw=psl_data_raw.chunk(chunks={'time':chunk})
        psl_data_raw=psl_data_raw.sel(time=slice('%s-01-01'%years[0],'%s-12-31'%(years[-1]+1)))
        if 'lat' in psl_data_raw.dims:
            psl_data_raw=psl_data_raw.rename({'lat':'latitude', 'lon':'longitude'})
        if psl_data_raw.latitude[0].item()>psl_data_raw.latitude[-1].item(): # For ERA5 
            psl_data_raw=psl_data_raw.isel(latitude=slice(None, None, -1)) 
        psl_data_raw=psl_data_raw.sel(latitude=slice(40,90))
        mon_mask=psl_data_raw.time.dt.month.isin(mons_values)
        psl_data_raw=psl_data_raw.sel(time=mon_mask)
        ## Compute
        psl_data_raw=psl_data_raw.compute()
        #print(en, psl_data_raw.min(), psl_data_raw.max())
        ## Remove seasonal cycle
        psl_data_raw=tools.remove_seasonal_cycle_simple(psl_data_raw, detrend=False)
        ## Regrid 
        #regridder=xe.Regridder(psl_data_raw.to_dataset(name='hihi'),ds_out,"bilinear",unmapped_to_nan=True, ignore_degenerate=True) 
        regridder=xe.Regridder(psl_data_raw.to_dataset(name='hihi'),ds_out,"bilinear",ignore_degenerate=True) 
        psl_data_raws[en]=regridder(psl_data_raw)

    ### Start the algorithms
    #regions=['Scan','Urals','Alaska','Newfoundland']
    regions=['Alaska','Urals'] # This is a figure for the revised letter - but not in the supporting information
    regions=['Alaska','Scan'] # This is the default - Figures 5 and 6 and 7 - the historgram
    #regions_latlon={'Scan':(55,72,5,40),'Urals':(45,60,30,60),'Alaska':(58,71,-170,-105),'Newfoundland':(45,63,-80,-55)}
    regions_latlon={'Scan':(55,72,5,41),'Alaska':(58,72,-167,-130)} # Scandanavia and Alaska (not extended, new one)
    regions_latlon={'Scan':(55,72,5,41),'Alaska':(58,72,-167,-130),'Urals':(58,73,43,82)}
    lat1, lat2, lon1, lon2 = 58,73,43,82  
    regions_xylims={'Scan':[(-30,110),(45,86)],'Alaska':[(-175,-35),(45,86)]} # Make sure they have the same size
    regions_xylims={'Scan':[(-30,110),(45,86)],'Alaska':[(-175,-35),(45,86)],'Urals':[(0,140),(45,86)]}
    shading_grids, contour_grids, contour1_grids = [],[],[]
    contour1_grids = []
    xylims_grids=[]
    coeffs={region:{en:{} for en in ensembles} for region in regions}
    tas_index_save={region:{en:{} for en in ensembles} for region in regions}
    for region in regions:
        latlon=regions_latlon[region]
        lat1,lat2,lon1,lon2=latlon[0], latlon[1], latlon[2], latlon[3]
        xylim=regions_xylims[region]
        cold_tas_data_save={en:{} for en in ensembles}
        hot_tas_data_save={en:{} for en in ensembles}
        cold_psl_data_save={en:{} for en in ensembles}
        hot_psl_data_save={en:{} for en in ensembles}
        tasAA_data_save={en:{} for en in ensembles}
        for en in ensembles:
            for year in years:
                ## Extract only SONDJF daily data (TAS)
                tas_data=tas_data_raws[en].sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-%s'%(year+1,mons[-1],mon_last_day))) # cannot used for SON with year+1
                mon_mask =tas_data.time.dt.month.isin(mons_values)
                tas_data=tas_data.sel(time=mon_mask)
                ## Pick regions for defining the TAS extreme (Here should change - do area average)
                lons=tas_data.longitude.values; lats=tas_data.latitude.values
                tas_data_idx=ct.weighted_area_average(tas_data,lat1,lat2,lon1,lon2,lons,lats)
                tas_data_idx=xr.DataArray(tas_data_idx,dims=['time'],coords={'time':tas_data.time})
                tas_index_save[region][en][year]=tas_data_idx
                ## Pick cold extreme (5% data with coldest temeprature)
                q05=tas_data_idx.quantile(0.1,dim='time')
                q95=tas_data_idx.quantile(0.9,dim='time')
                ## Get the days of these extrems 
                cold_time_sel=tas_data_idx.time[tas_data_idx<q05]
                hot_time_sel=tas_data_idx.time[tas_data_idx>q95]
                cold_tas_data_save[en][year]=tas_data.sel(time=cold_time_sel).mean(dim='time') # Taking the mean of extreme daily values
                hot_tas_data_save[en][year]=tas_data.sel(time=hot_time_sel).mean(dim='time')
                ## Apply sel day to PSL data
                psl_data=psl_data_raws[en].sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-%s'%(year+1,mons[-1],mon_last_day))) 
                mon_mask=psl_data.time.dt.month.isin(mons_values)
                psl_data=psl_data.sel(time=mon_mask)
                cold_psl_data_save[en][year]=psl_data.sel(time=cold_time_sel).mean(dim='time')
                hot_psl_data_save[en][year]=psl_data.sel(time=hot_time_sel).mean(dim='time')
                ## Get the TAS AA data (all days)
                tasAA_data=tasAA_data_raws[en].sel(time=slice('%s-%s-01'%(year,mons[0]),'%s-%s-%s'%(year+1,mons[-1],mon_last_day))) 
                mon_mask =tasAA_data.time.dt.month.isin(mons_values)
                tasAA_data=tasAA_data.sel(time=mon_mask)
                tasAA_data=tasAA_data.mean(dim='time') # Taking the seasoanl mean
                tasAA_data_save[en][year]=tasAA_data
        ## Calculate the mean (across all members and years) of SLP assoicated with cold and hot extremes
        cold_psl_data_mean=xr.concat([cold_psl_data_save[en][year] for en in ensembles for year in years],dim='en_and_year').mean(dim='en_and_year')
        hot_psl_data_mean=xr.concat([hot_psl_data_save[en][year] for en in ensembles for year in years],dim='en_and_year').mean(dim='en_and_year')
        ## Project the mean-circulation data onto their winter mean to get the circulation index
        # Set x (mean)
        x=cold_psl_data_mean-hot_psl_data_mean
        x=x.sel(latitude=slice(45,80))
        cos_lat=np.cos(x.latitude*np.pi/180)
        x=x*cos_lat
        for en in ensembles:
            for year in years:
                # Set y (individual days - mean over a seasons)
                y=(cold_psl_data_save[en][year]-hot_psl_data_save[en][year]).sel(latitude=slice(45,80))
                cos_lat=np.cos(y.latitude*np.pi/180)
                y=y*cos_lat
                coeffs[region][en][year]=np.dot(y.values.reshape(-1),x.values.reshape(-1))
        ## Calculate the trends of cold and hot extremes, and compute their difference
        cold_tas_trends=[]
        hot_tas_trends=[]
        for en in ensembles:
            ## Cold extremes
            tass=xr.concat([cold_tas_data_save[en][year] for year in years],dim='time')
            time_idx=range(tass.time.size)
            x=xr.DataArray(time_idx, dims=['time'], coords={'time':time_idx}); xmean=x.mean(dim='time')
            y=tass.assign_coords({'time':time_idx}); ymean=y.mean(dim='time')
            slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') # Unit is per K/year
            cold_tas_trends.append(slope*10)
            ## Hot extremes
            tass=xr.concat([hot_tas_data_save[en][year] for year in years],dim='time')
            time_idx=range(tass.time.size)
            x=xr.DataArray(time_idx, dims=['time'], coords={'time':time_idx}); xmean=x.mean(dim='time')
            y=tass.assign_coords({'time':time_idx}); ymean=y.mean(dim='time')
            slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') # Unit is K/year
            hot_tas_trends.append(slope*10)
        ## Average across all ensembles
        cold_tas_trends_mean=xr.concat(cold_tas_trends,dim='en').mean(dim='en')
        hot_tas_trends_mean=xr.concat(hot_tas_trends,dim='en').mean(dim='en')
        ## Calculate the trends of seasonal-mean AA
        tasAA_trends=[]
        for en in ensembles:
            tasAA=xr.concat([tasAA_data_save[en][year] for year in years],dim='time')
            time_idx=range(tasAA.time.size)
            x=xr.DataArray(time_idx, dims=['time'], coords={'time':time_idx}); xmean=x.mean(dim='time')
            y=tasAA.assign_coords({'time':time_idx}); ymean=y.mean(dim='time')
            slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') 
            tasAA_trends.append(slope*10)
        ## Average across all ensembles
        tasAA_trends_mean=xr.concat(tasAA_trends,dim='en').mean(dim='en')
        if False: ## Calulate the trends of ciruclation differences
            cold_psl_trends=[]
            hot_psl_trends=[]
            contour2_grids=[]
            for en in ensembles:
                ## Cold extremes
                psls=xr.concat([cold_psl_data_save[en][year] for year in years],dim='time')
                time_idx=range(psls.time.size)
                x=xr.DataArray(time_idx, dims=['time'], coords={'time':time_idx}); xmean=x.mean(dim='time')
                y=psls.assign_coords({'time':time_idx}); ymean=y.mean(dim='time')
                slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') # Unit is Pa/year
                cold_psl_trends.append(slope*10)
                ## Hot extremes
                psls=xr.concat([hot_psl_data_save[en][year] for year in years],dim='time')
                time_idx=range(psls.time.size)
                x=xr.DataArray(time_idx, dims=['time'], coords={'time':time_idx}); xmean=x.mean(dim='time')
                y=psls.assign_coords({'time':time_idx}); ymean=y.mean(dim='time')
                slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') # Unit is Pa/year
                hot_psl_trends.append(slope*10)
            ## Average across all ensembles
            cold_psl_trends_mean=xr.concat(cold_psl_trends,dim='en').mean(dim='en')
            hot_psl_trends_mean=xr.concat(hot_psl_trends,dim='en').mean(dim='en')
            contour2_grids.append(cold_psl_trends_mean-hot_psl_trends_mean)
        ## Create the plotting grids
        shading_grids.append(cold_tas_trends_mean-hot_tas_trends_mean)
        contour_grids.append(cold_psl_data_mean-hot_psl_data_mean)
        contour1_grids.append(tasAA_trends_mean) # this one is actually hatching grids
        xylims_grids.append(xylim)

    ###
    if True: ### Start plotting TAS and SLP trends for the regions
        row=1; col=len(shading_grids)
        grid = row*col
        contour_clevels=[np.linspace(-1500,1500,13)]*grid
        contour_clevels=[np.linspace(-3000,3000,11)]*grid
        if 'shading_level_grid' not in locals():
            shading_level_grid=np.linspace(-0.6,0.6,13) # TAS tremds for models (many members)
        shading_level_grids=[shading_level_grid]*grid # TAS tremds for models (many members)
        ###
        mask_ocean=False
        mask_ocean=True
        mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff','#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        mapcolor_grid = [cmap] * grid
        ####
        clabels_row = [''] * grid
        top_title = [''] * col
        left_title = [''] * row
        leftcorner_text = None
        import cartopy.crs as ccrs
        projection=ccrs.PlateCarree(central_longitude=0); xsize=8; ysize=2
        pval_map = None
        matplotlib.rcParams['hatch.linewidth'] = 1;matplotlib.rcParams['hatch.color'] = 'lightgray'
        pval_hatches = [[[0, 0.1, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
        pval_hatches = [None]*grid
        fill_continent=False
        if True: # Set region boxes
            region_boxes=[]
            for region in regions:
                latlon=regions_latlon[region]
                lat1,lat2,lon1,lon2=latlon[0], latlon[1], latlon[2], latlon[3]
                box=[tools.create_region_box(lat1, lat2, lon1, lon2)] 
                region_boxes.append(box[0])
        region_boxes_extra=None
        #####
        #fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
        plt.close()
        fig = plt.figure(figsize=(10,4))
        #gs = fig.add_gridspec(2,2,height_ratios=[15,1])
        gs = fig.add_gridspec(1,2)
        ax1 = fig.add_subplot(gs[0,0],projection=projection) 
        ax2 = fig.add_subplot(gs[0,1],projection=projection) 
        tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grids, clabels_row, top_titles=top_title, 
                        left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
                        region_boxes_extra=region_boxes_extra, leftcorner_text=leftcorner_text, ylim=None, xlim=None, set_xylim=xylims_grids,quiver_grids=None,
                        pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
                        contour_map_grids=contour_grids, contour_clevels=contour_clevels, contour_lw=1.5, mask_ocean=mask_ocean,
                        colorbar=False, indiv_colorbar=[False]*grid, 
                        pltf=fig,ax_all=[ax1,ax2])
        if True: ## Plot the AA trends as hatchings
            ax_all=[ax1,ax2]
            contour1_level_grids=[np.linspace(-2,2,11)]*grid
            pval_hatch_levels=[[0,1.2,1.6,1000]]*grid
            hatches=[[None,'XXX','OOO']] * grid # Mask the insignificant regions
            for i, ax in enumerate(ax_all):
                lons=contour1_grids[i].longitude.values; lats=contour1_grids[i].latitude.values
                #csf=ax.contourf(lons,lats,contour1_grids[i],contour1_level_grids[i],cmap='bwr',linewidths=1,transform=ccrs.PlateCarree())
                ax.contourf(lons, lats, contour1_grids[i], pval_hatch_levels[i],
                        hatches=hatches[i],
                        colors='none', extend='neither', transform=ccrs.PlateCarree(),zorder=10)
        if False: ## Plot the SLP trends
            ax_all=[ax1,ax2]
            contour2_level_grids=[np.linspace(-100,100,11)]*grid
            for i, ax in enumerate(ax_all):
                lons=contour2_grids[i].longitude.values; lats=contour2_grids[i].latitude.values
                csf=ax.contourf(lons,lats,contour2_grids[i],contour2_level_grids[i],linewidths=1,transform=ccrs.PlateCarree())
        ## Add color bars
        #cba = fig.add_axes([0.91, 0.1, 0.02, 0.32])
        cba = fig.add_axes([0.91, 0.36, 0.02, 0.27])
        cNorm  = matplotlib.colors.Normalize(vmin=shading_level_grids[0][0], vmax=shading_level_grids[0][-1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        #cticks = yticklabels = [round(i,1) for i in shading_level_grids[0] if i!=0]
        cticks = yticklabels = [round(i,2) for i in shading_level_grids[0]][::2]
        #print(cb1.get_ticks())
        cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='vertical', ticks=cticks, extend='both')
        cb1.ax.set_yticklabels(cticks, fontsize=10)
        ## Save figure
        fig_name='fig5_maps'
        if 'fig_add' not in locals():
            fig_add='manual'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.002, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s_%s.png'%(dt.date.today(), fig_name, fig_add), bbox_inches='tight', dpi=450, pad_inches=0.001)
    ###
    if True: # Plot circulation indices (the expanasion coefficients)
        fig = plt.figure(figsize=(8,1))
        #gs = fig.add_gridspec(2,2,height_ratios=[15,1])
        gs = fig.add_gridspec(1,2)
        #ax1 = fig.add_subplot(gs[0,0],projection=projection) 
        #ax2 = fig.add_subplot(gs[0,1],projection=projection) 
        ax3 = fig.add_subplot(gs[0,0])
        ax4 = fig.add_subplot(gs[0,1])
        axs=[ax3,ax4]
        coeffs_std={region:{en:{} for en in ensembles} for region in regions}
        ## Standardize the timeseries
        for region in regions:
            for en in ensembles:
                temp=np.array([coeffs[region][en][year] for year in years])
                temp_std=(temp-temp.mean())/temp.std()
                for i, year in enumerate(years):
                    coeffs_std[region][en][year]=temp_std[i]
        coeffs_plot={region:[] for region in regions}
        for region in regions:
            for year in years:
                temp=[coeffs_std[region][en][year] for en in ensembles]
                coeffs_plot[region].append(temp) # the first key region, second level is years; third is eensemble no.
        ## Make the plot for each region and ensemble
        #plt.close()
        #fig,axs=plt.subplots(1,len(regions),figsize=(15,2))
        x_adjs=np.linspace(-0.15,0.15,4)
        xs = np.arange(len(coeffs_plot[region]))
        for i, region in enumerate(regions):
            years=np.array(years)
            median=np.median(coeffs_plot[region],axis=1)
            slope, intercept,rvalue,pvalue,stderr,=scipy.stats.linregress(years,median)
            #axs[i].plot(years,slope*years+intercept,'k')
            bp=axs[i].boxplot(coeffs_plot[region], positions=years, showfliers=True, widths=0.2, whis=(0,100),patch_artist=True)
            for element in ['boxes', 'whiskers', 'caps']:
                plt.setp(bp[element], color='k', lw=2.5) # lw=3 in Fig.1
            for box in bp['boxes']:
                box.set(facecolor='k')
            plt.setp(bp['medians'], color='white', lw=2)
            plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor='k', markeredgecolor='k')
            ###
            #nn=4
            #axs[i].set_xticks(years[::nn])
            #xticklabels=(np.array(years)+1)[::nn]
            #axs[i].set_xticklabels(xticklabels,rotation=0,size=8)
            ## Plot the beta and pvalue
            axs[i].annotate(r"$\beta$=%s; $\rho$=%s"%(str(round(slope,3)),str(round(pvalue,3))),xy=(0.01,0.96),xycoords='axes fraction', fontsize=8.5,
                        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
            ## Set the box boundary
            for j in ['right', 'top', 'bottom','left']:
                #axs[i].spines[j].set_visible(False)
                axs[i].tick_params(axis='x', which='both',length=2,direction='in')
                axs[i].tick_params(axis='y', which='both',length=2,direction='in')
        years_xticks=[i for i in range(np.min(years),np.max(years),5)] 
        years_xticklabels=[i+1 for i in years_xticks]# 1979 would be written as 1980
        for ax in [ax3,ax4]:
            ax.set_yticks([-2,0,2])
            ax.set_yticklabels([-2,0,2],size=8)
            ax.set_xticks(years_xticks)
            ax.set_xticklabels(years_xticklabels)
            ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=1, zorder=-10)
        ax4.set_yticklabels([])
        ## Save figure
        fig_name='fig6_timeseries'
        if 'fig_add' not in locals():
            fig_add='manual'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.04, hspace=-0.75) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s_%s.png'%(dt.date.today(), fig_name, fig_add), bbox_inches='tight', dpi=450, pad_inches=0.001)


    #import ipdb
    if True: ### Plot the change of histograms for Yutian and James's interests
        regions_name={'Alaska':'Alaska','Scan':'Scandinavia'}
        regions_name={'Alaska':'Alaska','Scan':'Scandinavia','Urals':'Northern Urals'}
        from scipy.optimize import curve_fit
        def gaussian(x, mean, amplitude, standard_deviation):
            return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
        bins=np.linspace(-15,15,51)
        bins=np.linspace(-18,18,51)
        bins=np.linspace(-21,21,51)
        ## Start plotting
        plt.close()
        fig,axs=plt.subplots(1,len(regions),figsize=(6,1))
        for i, region in enumerate(regions):
            late_tass=[]
            early_tass=[]
            #ipdb.set_trace()
            for en in ensembles:
                early_tas= xr.concat([tas_index_save[region][en][yr] for yr in early_years],dim='time')
                early_tass.append(early_tas)
                late_tas=xr.concat([tas_index_save[region][en][yr] for yr in late_years],dim='time')
                late_tass.append(late_tas)
            #early_tass = xr.concat(early_tass,dim='en').values.reshape(-1)
            early_tass=np.array([j.item() for i in early_tass for j in i])
            #late_tass = xr.concat(late_tass,dim='en').values.reshape(-1)
            late_tass=np.array([j.item() for i in late_tass for j in i])
            early_tass_std = np.std(early_tass)
            late_tass_std = np.std(late_tass)
            ## Plot the early tas
            early_weight= np.ones_like(early_tass) / len(early_tass) * 100 #The sum of all bars equals to 100
            #bin_heights, bin_borders, _ = axs[i].hist(early_tass,bins=bins,edgecolor='lightgrey',fc='lightgrey',lw=0,weights=early_weight, label='Early', alpha=0.5)
            bin_heights, bin_borders, _ = axs[i].hist(early_tass,bins=bins,edgecolor='lightgrey',fc='lightgrey',lw=0,weights=None, label='Early', alpha=0.5)
            bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
            early_q05=np.percentile(early_tass,10)
            early_q95=np.percentile(early_tass,90)
            ## For q05 coloring
            idx = np.argmin(np.abs(bin_centers-early_q05))
            xx=bin_centers[0:idx]
            bar_width=np.diff(bin_borders)[0]
            bar_alpha=0.7
            bar_lw=0.05
            axs[i].bar(xx-bar_width*0.3,bin_heights[0:idx],bar_width,bottom=0, color='skyblue',edgecolor='k',lw=bar_lw,alpha=bar_alpha)
            ## For q95 coloring
            idx = np.argmin(np.abs(bin_centers-early_q95))
            xx=bin_centers[idx:]
            bar_width=np.diff(bin_borders)[0]
            axs[i].bar(xx+bar_width*0.3,bin_heights[idx:],bar_width,bottom=0, color='lightsalmon',edgecolor='k',lw=bar_lw,alpha=bar_alpha, zorder=10)
            ## Plot the curve fit line
            #popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1, 5, 1])
            #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1],5000)
            #axs[i].plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='royalblue')
            ## Plot the late tas
            late_weight= np.ones_like(late_tass) / len(late_tass) * 100 #The sum of all bars equals to 100
            #bin_heights, bin_borders, _ = axs[i].hist(late_tass,bins=bins,edgecolor='dimgray',fc='dimgray',lw=0,weights=late_weight,label='Late',alpha=0.5)
            bin_heights, bin_borders, _ = axs[i].hist(late_tass,bins=bins,edgecolor='dimgray',fc='dimgray',lw=0,weights=None,label='Late',alpha=0.5)
            bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
            ## For q05 coloring
            late_q05=np.percentile(late_tass,10)
            late_q95=np.percentile(late_tass,90)
            idx = np.argmin(np.abs(bin_centers-late_q05))
            xx=bin_centers[0:idx]
            bar_width=np.diff(bin_borders)[0]
            axs[i].bar(xx-bar_width*0.3,bin_heights[0:idx],bar_width,bottom=0, color='royalblue',edgecolor='k',lw=bar_lw,alpha=bar_alpha)
            ## For q95 coloring
            idx = np.argmin(np.abs(bin_centers-late_q95))
            xx=bin_centers[idx:]
            bar_width=np.diff(bin_borders)[0]
            axs[i].bar(xx+bar_width*0.3,bin_heights[idx:],bar_width,bottom=0, color='red',edgecolor='k',lw=bar_lw,alpha=bar_alpha, zorder=1)
            ## Plot the curve fit line
            #popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1, 2, 1])
            #x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1],5000)
            #axs[i].plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='red')
            ###
            ## Plot the standand deviation and percentile warming
            axs[i].annotate("STD: %s to %s K"%(str(round(early_tass_std,1)),str(round(late_tass_std,1))),xy=(0.01,0.95),xycoords='axes fraction', fontsize=7)
            axs[i].annotate(r"$\Delta$T:%s K"%(str(round(late_q05-early_q05,1))),xy=(0.01,0.3),xycoords='axes fraction', fontsize=6, color='royalblue')
            axs[i].annotate(r"$\Delta$T:%s K"%(str(round(late_q95-early_q95,1))),xy=(0.7,0.3),xycoords='axes fraction', fontsize=6, color='red')
            #axs[i].annotate(r"$\rho$=%s"%(str(round(corr,3))),xy=(0.01,0.95), xycoords='axes fraction', fontsize=10)
            ## Set title
            axs[i].set_title(regions_name[region],loc='left',size=9)
            ## Setup the axis
            for j in ['right', 'top']:
                axs[i].spines[j].set_visible(False)
                axs[i].tick_params(axis='x', which='both',length=2)
                axs[i].tick_params(axis='y', which='both',length=2)
            axs[i].tick_params(axis='x', direction="out", length=3, colors='black')
            axs[i].tick_params(axis='y', direction="out", length=3, colors='black')
            axs[i].set_yticks([]) if i!=0 else ""
        ## Save figure
        fig_name='fig7_tas_histogram_early_late'
        if 'fig_add' not in locals():
            fig_add='manual'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s_%s.png'%(dt.date.today(), fig_name, fig_add), bbox_inches='tight', dpi=450, pad_inches=0.001)
