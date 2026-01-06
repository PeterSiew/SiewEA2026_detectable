import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
from importlib import reload
from scipy import stats

import sys
import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import tools; reload(tools)

if __name__ == "__main__":

    if False:
        #  Plot the composite maps of the trends of T2M variability and T2M mean (that is AA) using daily data
        var = 'best_SAT'; ensembles=['']; years=range(1979,2018)
        var = 'T2M_regrid_1x1'; ensembles=['']; years=range(1980,2022)
        var = 'cesm1_snc_en'; ensembles=range(2,41); years=range(1979,2022)
        var = 'ERA5_snc'; ensembles=['']; years=range(1979,2022)
        var = 'MIROC6_histGHG_tas_daily_en'; ensembles=range(1,51); years=range(1980,2020)
        var = 'MIROC6_histaer_tas_daily_en'; ensembles=range(1,11); years=range(1980,2020)
        var = 'MIROC6_histnat_tas_daily_en'; ensembles=range(1,16); years=range(1980,2020)
        var = 'HadGEM3_histGHG_tas_daily_en'; ensembles=[i for i in range(1,61) if i not in [6,7,8,9,10]]; years=range(1980,2020)
        var = 'HadGEM3_histaer_tas_daily_en'; ensembles=[i for i in range(1,61) if i not in [6,7,8,9,10]]; years=range(1980,2020)
        var = 'HadGEM3_histnat_tas_daily_en'; ensembles=[i for i in range(1,61) if i not in [31,32,37]]; years=range(1980,2020)
        var = 'CESM2_histGHG_tas_daily_en'; ensembles=range(1,16); years=range(1980,2020)
        var = 'CESM2_histaer_tas_daily_en'; ensembles=range(1,16); years=range(1980,2020)
        var = 'CESM2_histEE_tas_daily_en'; ensembles=range(1,16); years=range(1980,2020)

        ### Models
        var = 'amip_ECHAM5_daily_tas_en'; ensembles=range(1,31); years=range(1979,2018)
        var = 'amip_ECHAM5_daily_climsic_tas_en'; ensembles=range(1,31); years=range(1979,2018)
        var = 'gfdlcm3_tas_daily_en'; ensembles=range(1,21); years=range(1980,2024)
        var = 'ecearth_tas_daily_en'; ensembles=range(1,17); years=range(1980,2024)
        var = 'amip_WACCM6_climsic_daily_tas_en'; ensembles=range(1,31); years=range(1979,2014)
        var = 'amip_WACCM6_daily_tas_en'; ensembles=range(1,31); years=range(1979,2014)
        var = 'gfdlesm2m_tas_daily_en'; ensembles=range(1,31); years=range(1980,2024)
        var = 'mk360_tas_daily_en'; ensembles=range(1,21); years=range(1980,2024)
        var = 'cesm1_tas_daily_en'; ensembles=range(2,22); years=range(1980,2024)
        var = 'canesm2_tas_daily_en'; ensembles=range(1,21); years=range(1980,2024)

        ### WACCM6
        vars=['amip_WACCM6_daily_tas_en']; corner_text=['WACCM6 obs ice (1979-2014)']
        vars=['amip_WACCM6_climsic_daily_tas_en']; corner_text=['WACCM6 clim ice (1979-2014)']
        years=[range(1979,2014)]*len(vars)
        ensembles={'amip_WACCM6_daily_tas_en':range(1,31), 'amip_WACCM6_climsic_daily_tas_en':range(1,31)}


    ### All LENS simulation together (Figure S5)
    vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en']
    vars_labels={'cesm1_tas_daily_en':'NCAR-CESM1', 'canesm2_tas_daily_en':'CCCma-CanESM2','gfdlcm3_tas_daily_en':'GFDL-CM3', 
            'gfdlesm2m_tas_daily_en':'GFDL-ESM2M', 'mk360_tas_daily_en':'CSIRO-MK360', 'ecearth_tas_daily_en':'SMHI/KNMI-EC-Earth'}
    ## Full range
    ensembles={'cesm1_tas_daily_en':range(1,41),'canesm2_tas_daily_en':range(1,51),'gfdlesm2m_tas_daily_en':range(1,31), 'gfdlcm3_tas_daily_en':range(1,21),
                'mk360_tas_daily_en':range(1,31),'ecearth_tas_daily_en':range(1,17)}; ensemble_no='hihi'
    ## 16 member each
    ensembles={'cesm1_tas_daily_en':range(1,17),'canesm2_tas_daily_en':range(1,17),'gfdlesm2m_tas_daily_en':range(1,17), 
            'gfdlcm3_tas_daily_en':range(1,17),'mk360_tas_daily_en':range(1,17),'ecearth_tas_daily_en':range(1,17),
            'amip_WACCM6_daily_tas_en':range(1,31), 'amip_WACCM6_climsic_daily_tas_en':range(1,31)}; add_MMM=True
    #years=[range(2050,2100)]*len(vars)
    years=[range(1970,2100)]*len(vars)
    corner_text=[vars_labels[var] for var in vars]

    ### Observations (Figure 2B)
    vars=['BESTSAT_daily_regrid_1x1']; corner_text=['BESTSAT (1979-2024)']
    vars=['ERA5_T2M_daily_regrid_1x1']
    ensembles={'ERA5_T2M_daily_regrid_1x1':[''],'BESTSAT_daily_regrid_1x1':['']}; add_MMM=False
    #years=[range(1979,2024)]*len(vars); corner_text=['ERA5 (1980-2024)']
    years=[range(1975,2024)]*len(vars); corner_text=['ERA5 (1976-2024)']
    years=[range(1975,2024)]*len(vars); corner_text=['']

    ### Start
    lat1, lat2= 45,90; lon1, lon2= -180, 180 # For temperature variability
    season='SON'
    season='DJF'
    season='SONDJF'
    grid=len(vars)
    if False: ### Decide what to plot
        mask_ocean=False; plotting_type='AA'
        shading_level_grid = [np.linspace(-4,4,13)]*grid # K/decade (single AA)
        shading_level_grid = [np.linspace(-2,2,13)]*grid # K/decade (ensemble AA)
    else:
        mask_ocean=True; plotting_type='tempvar'
        shading_level_grid = [np.linspace(-0.12,0.12,13)]*grid # Single member (ERA5)
        shading_level_grid = [np.linspace(-0.3,0.3,13)]*grid # Ensemble mean (T2M variability)

    ### Start extract the data and treand calculation
    data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/DJF/%s%s_%s.nc'%('BESTSAT_daily_regrid_1x1','',1980))['training']
    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
    land_mask = ~np.isnan(data)
    trends={var:{} for var in vars}
    for i, var in enumerate(vars):
        print(var)
        for en in ensembles[var]:
            datas = []
            for yr in years[i]:
                if plotting_type=='tempvar':
                    ## Temperature variability trend
                    data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    #data=xr.where(land_mask,data,np.nan)
                elif plotting_type=='AA':
                    ## Arctic warming trend
                    data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_AA/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                if False: # Turn the AA trends into dT/dy trends
                    data=data.differentiate("latitude")*10 # per 1 deg latitude (111km) to 1110km
                    pass
                datas.append(data)
            ### Calculate the trends for indiviudal members
            datas=xr.concat(datas,dim='time')
            time=np.arange(datas.time.size)
            datas=datas.assign_coords({'time':time})
            X=xr.DataArray(range(datas.time.size), dims=['time'], coords={'time':datas.time})
            Ys=datas
            results_ds = tools.linregress_xarray(Ys,X)
            trend = results_ds['slope'] * 10
            pvalue = results_ds['pvalues'] 
            pval_map = [pvalue]
            trends[var][en]=trend
    ### Calculate the ensemble average trends within each model or obs
    trends_new={}
    for var in vars:
        var_trends=[]
        for en in ensembles[var]:
            trend=trends[var][en]
            var_trends.append(trend)
        ## Ensemble mean
        trends_new[var]=xr.concat(var_trends,dim='en').mean(dim='en')
    ## Append across simulations/observations
    shading_grids=[trends_new[var] for var in vars]

    if add_MMM: # add MMM
        MMM=xr.concat([trends_new[var] for var in vars],dim='model').mean(dim='model')
        shading_grids.append(MMM)
        corner_text.append('MMM')
        shading_level_grid.append(shading_level_grid[0])
        colors={'cesm1_tas_daily_en':'red', 'canesm2_tas_daily_en':'orange', 'gfdlesm2m_tas_daily_en':'gold',
                'gfdlcm3_tas_daily_en':'pink', 'mk360_tas_daily_en':'magenta', 'ecearth_tas_daily_en':'blueviolet'}
        vars_labels={'cesm1_tas_daily_en':'NCAR-CESM1', 'canesm2_tas_daily_en':'CCCma-CanESM2','gfdlcm3_tas_daily_en':'GFDL-CM3', 
                'gfdlesm2m_tas_daily_en':'GFDL-ESM2M', 'mk360_tas_daily_en':'CSIRO-MK360', 'ecearth_tas_daily_en':'SMHI/KNMI-EC-Earth'}
        if True: # Projecting each simulation onto MMM (Figure S5)
            data = xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/DJF/%s%s_%s.nc'%('BESTSAT_daily_regrid_1x1','',1980))['training']
            lat1, lat2= 45,90; lon1, lon2= -180,180 
            data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            land_mask=~np.isnan(data)
            rmses=[]
            for var in vars:
                y=trends_new[var] 
                y=xr.where(land_mask,y,np.nan)
                cos_lat=np.cos(y.latitude*np.pi/180); y=y*cos_lat
                x=MMM
                x=xr.where(land_mask,x,np.nan)
                cos_lat=np.cos(x.latitude*np.pi/180); x=x*cos_lat
                y_mask=np.isnan(y.values.reshape(-1)); final_y=y.values.reshape(-1)[~y_mask]
                x_mask=np.isnan(x.values.reshape(-1)); final_x=x.values.reshape(-1)[~x_mask]
                #rmse=np.dot(final_y,final_x)
                rmse=np.dot(final_x,final_y)
                #rmse=tools.correlation_nan(final_y,final_x)
                #rmse=tools.rmse_nan(y.values.reshape(-1),x.values.reshape(-1)) ## rmse
                rmses.append(rmse)
            rmses_std=np.array(rmses)-np.min(rmses) / (np.max(rmses)-np.min(rmses))
            if True: # add the projection coefficents onto the left corner text
                coeffs_add = [str(round(i,1)) for i in rmses] + ['']
                corner_text_new=[]
                for i,coeff in enumerate(coeffs_add):
                    if len(coeff)!=0:
                        new_text=corner_text[i]+' (projectin this onto MMM: %s)'%coeff
                    else:
                        new_text=corner_text[i]
                    corner_text_new.append(new_text)
                corner_text=corner_text_new
            rhos=[0.62,0.65,0.45,0.78,0.55,0.68]
            if False: ### Just for testing
                plt.close()
                fig,ax1=plt.subplots(1,1,figsize=(3,3))
                #ax1.scatter(rmses,rhos)
                for i, var in enumerate(vars): 
                    ax1.scatter(rmses_std[i],rhos[i],color=colors[var],label=vars_labels[var])
                ax1.legend(bbox_to_anchor=(0.001,0.55), ncol=1, loc='lower left', frameon=True, columnspacing=1, handletextpad=0.5,fontsize=7)
                ax1.set_xlabel('Normalized\nprojection coefficient')
                ax1.set_ylabel('Correlation in the\ncross-validation test')
                ax1.set_xticks([0,0.5,1])
                ### Save Figures
                fig_name = 'how_close_to_mean_rho_in_take_one_ensemble_out'
                plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=-0.94) # hspace is the vertical
                plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

    #if False: # turn the seasonal mean warming to change of meridional temperature graident
    #    trend_en_mean=trend_en_mean.differentiate("latitude")*10 # per 1 deg latitude (111km) to 1110km

    ### Start the map plotting
    row=len(shading_grids); col = 1
    grid = row*col
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff',
                '#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b'] # 10 colors
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff','#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    contour_grids = None
    contour_clevels = None
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    leftcorner_text = corner_text
    #projection=ccrs.NorthPolarStereo(); xsize=1.5; ysize=1.5
    #projection=ccrs.Robinson(central_longitude=0); xsize=6; ysize=2
    #projection=ccrs.Mollweide(central_longitude=0); xsize=8; ysize=2
    projection=ccrs.PlateCarree(central_longitude=0); xsize=5.95; ysize=3
    xlim = [-80,120]
    xlim = [-180,180]
    ylim = (lat1+2,90) # plot from lat=47
    if True:
        #tlat1, tlat2, tlon1, tlon2 = 55,72,5,73 # Scandanavia extended
        tlat1, tlat2, tlon1, tlon2 = 55,72,5,41 # Scandanavia only
        region_boxes = [tools.create_region_box(tlat1, tlat2, tlon1, tlon2)] + [None]*10
        #tlat1, tlat2, tlon1, tlon2 = 58,75,-165,-90  # Alaska extended
        tlat1, tlat2, tlon1, tlon2 = 58,72,-167,-130  # Alaska only
        region_boxes_extra = [tools.create_region_box(tlat1, tlat2, tlon1, tlon2)] + [None]*10
    else:
        region_boxes=[None]*10
        region_boxes_extra=[None]*10
    matplotlib.rcParams['hatch.linewidth'] = 1;matplotlib.rcParams['hatch.color'] = 'lightgray'
    #pval_hatches = [[[0, 0.05, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
    pval_hatches = False
    pval_map=None
    fill_continent=False
    #####
    plt.close()
    #fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), projection=projection)
    fig = plt.figure(figsize=(col*xsize, row*ysize))
    ax_all = [fig.add_subplot(row, col, i+1, projection=projection) for i in range(grid)]
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
                    region_boxes_extra=region_boxes_extra,region_box_extra_color='red',
                    leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
                    contour_map_grids=contour_grids, contour_clevels=contour_clevels, contour_lw=1.5, mask_ocean=mask_ocean,
                    colorbar=False, indiv_colorbar=[False]*grid, ax_all=ax_all, pltf=fig)
    if False:### Add two more boxes
        ## Boxes 1
        lat1, lat2, lon1, lon2 = 45,63,-80,-55  # Newfoundland
        region_boxes = tools.create_region_box(lat1, lat2, lon1, lon2)
        lons_l, lats_l = region_boxes[0], region_boxes[1]
        ax_all[0].plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color='red', linewidth=1)
    if True: # Add an extra box for revised letter (northern Urals)
        ## Boxes 2
        lat1, lat2, lon1, lon2 = 58,73,43,82  
        region_boxes = tools.create_region_box(lat1, lat2, lon1, lon2)
        lons_l, lats_l = region_boxes[0], region_boxes[1]
        ax_all[0].plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color='red', linewidth=1)
    if True: ### Setup the coloar bar
        cba = fig.add_axes([0.91, 0.4, 0.02, 0.18])
        #cba = fig.add_axes([0.2, 0.03, 0.6, 0.03])
        cNorm  = matplotlib.colors.Normalize(vmin=shading_level_grid[0][0], vmax=shading_level_grid[0][-1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='vertical', extend='both')
        #cticks=[i for i in shading_level_grid[0]]
        #cticks=[round(i,2) for i in shading_level_grid[0] if i!=0]
        #cticks=shading_level_grid[0]
        #cb1.ax.set_yticks(cticks)
        cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=7)
        #cb1.set_label('XX', fontsize=10, rotation=0, y=-0.05, labelpad=1)
    ### Save Figures
    fig_name = 'fig2B_tempsd_or_AA_trend'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=-0.94) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)
