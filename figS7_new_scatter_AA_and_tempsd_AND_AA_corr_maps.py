import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import ipdb
import matplotlib
import cartopy.crs as ccrs
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import scipy

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import create_timeseries as ct
import tools; reload(tools)
import figX_new_scatter_AA_and_tempsd_AND_AA_corr_maps as figX; reload(figX)

if __name__ == "__main__":


    if False: # Only forced simulations (Future decades)
        vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en']
        ensembles=[range(2,18),range(1,17),range(1,17),range(1,17),range(1,17),range(1,17)]
        ensembles=[range(1,17),range(1,17),range(1,17),range(1,17),range(1,17),range(1,17)]
        years = [range(2055,2100)]*len(vars) 
        years = [range(2030,2075)]*len(vars) 
    if True: # Forced simulations + ERA5 + WACCM6 + ECHAM5
        vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en',
                    'amip_WACCM6_climsic_daily_tas_en','amip_WACCM6_daily_tas_en','amip_ECHAM5_daily_climsic_tas_en','amip_ECHAM5_daily_tas_en']
        vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en',
                    'amip_WACCM6_climsic_daily_tas_en','amip_WACCM6_daily_tas_en','ERA5_T2M_daily_regrid_1x1'] # The BSAT has no SAT in Arctic so it is not included
        ## Set ensemble members
        ensembles=[range(2,5),range(1,4),range(1,4),range(1,4),range(1,4),range(1,4),range(1,4),range(1,4),['']] # For testing
        ensembles=[range(1,17),range(1,17),range(1,17),range(1,17),range(1,17),range(1,17),range(1,31),range(1,31),['']]
        ## Set years
        years = [range(1979,2024)]*6 + [range(1979,2014)]*2 + [range(1979,2024)] 

    season='SON'
    season='DJF'
    season='SONDJF'
    var_types = ['AA','tempsd']

    ### Get the data and calculate the trend into maps
    slopes = {var: {tt:[] for tt in var_types} for i, var in enumerate(vars)}
    for i, var in enumerate(vars):
        print(var)
        for en in ensembles[i]:
            for tt in var_types:
                datas = []
                for yr in years[i]:
                    if tt in ['tempsd']:
                        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_%s/%s/%s%s_%s.nc'%(tt,season,var,en,yr))['training']
                    if tt in ['AA']:
                        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_%s/%s/%s%s_%s.nc'%(tt,season,var,en,yr))['training']
                    datas.append(data)
                datas = xr.concat(datas,dim='time')
                time=np.arange(datas.time.size)
                datas = datas.assign_coords({'time':time})
                # Calculate the trend in each grid
                x = xr.DataArray(range(datas.time.size), dims=['time'], coords={'time':datas.time}); xmean=x.mean(dim='time')
                y = datas; ymean=y.mean(dim='time')
                results_ds = tools.linregress_xarray(y, x)
                trend = results_ds['slope'] * 10
                slopes[var][tt].append(trend)

    colors={'ERA5_T2M_daily_regrid_1x1':'k', 'cesm1_tas_daily_en':'red', 'canesm2_tas_daily_en':'orange', 'gfdlesm2m_tas_daily_en':'gold',
            'gfdlcm3_tas_daily_en':'pink', 'mk360_tas_daily_en':'magenta', 'ecearth_tas_daily_en':'blueviolet',
            'amip_WACCM6_climsic_daily_tas_en':'aqua', 'amip_WACCM6_daily_tas_en':'blue',
            'amip_ECHAM5_daily_climsic_tas_en':'lime','amip_ECHAM5_daily_tas_en':'green', 'BESTSAT_daily_regrid_1x1':'k'}
    markers={'ERA5_T2M_daily_regrid_1x1':'^', 'cesm1_tas_daily_en':'o', 'canesm2_tas_daily_en':'o', 'gfdlesm2m_tas_daily_en':'o',
            'gfdlcm3_tas_daily_en':'o', 'mk360_tas_daily_en':'o', 'ecearth_tas_daily_en':'o',
            'amip_WACCM6_climsic_daily_tas_en':'d', 'amip_WACCM6_daily_tas_en':'d',
            'amip_ECHAM5_daily_climsic_tas_en':'d','amip_ECHAM5_daily_tas_en':'d', 'BESTSAT_daily_regrid_1x1':'X'}
    zorders={'ERA5_T2M_daily_regrid_1x1':5, 'cesm1_tas_daily_en':0, 'canesm2_tas_daily_en':0, 'gfdlesm2m_tas_daily_en':0,
            'gfdlcm3_tas_daily_en':0, 'mk360_tas_daily_en':0, 'ecearth_tas_daily_en':0,
            'amip_WACCM6_climsic_daily_tas_en':1, 'amip_WACCM6_daily_tas_en':1,
            'amip_ECHAM5_daily_climsic_tas_en':1,'amip_ECHAM5_daily_tas_en':1, 'BESTSAT_daily_regrid_1x1':5}
    sizes={'ERA5_T2M_daily_regrid_1x1':20, 'cesm1_tas_daily_en':15, 'canesm2_tas_daily_en':15, 'gfdlesm2m_tas_daily_en':15,
            'gfdlcm3_tas_daily_en':15, 'mk360_tas_daily_en':15, 'ecearth_tas_daily_en':15,
            'amip_WACCM6_climsic_daily_tas_en':20, 'amip_WACCM6_daily_tas_en':20,
            'amip_ECHAM5_daily_climsic_tas_en':20,'amip_ECHAM5_daily_tas_en':20, 'BESTSAT_daily_regrid_1x1':20}
    vars_labels={'ERA5_T2M_daily_regrid_1x1':'ERA5', 'cesm1_tas_daily_en':'CESM1', 'canesm2_tas_daily_en':'CanESM2',
            'gfdlcm3_tas_daily_en':'GFDL-CM3', 'gfdlesm2m_tas_daily_en':'GFDL-ESM2M', 'mk360_tas_daily_en':'CSIRO-MK360',
            'ecearth_tas_daily_en':'EC-Earth',
            'amip_WACCM6_climsic_daily_tas_en':'WACCM6 fixed ice', 'amip_WACCM6_daily_tas_en':'WACCM6 obs ice',
            'amip_ECHAM5_daily_climsic_tas_en':'ECHAM5 fixed ice','amip_ECHAM5_daily_tas_en':'ECHAM5 obs ice', 'BESTSAT_daily_regrid_1x1':'BESTSAT'}

    regions=['Alaska','Scan']
    plt.close()
    projection=ccrs.PlateCarree(central_longitude=0)
    fig = plt.figure(figsize=(7,5))
    for region in regions:
        tempsd_indices, AA_indices, corrs_mean, pvals_mean, tlatlon, Alatlon, xlims =figX.get_data_for_plotting(region, vars, slopes)
        tlat1,tlat2,tlon1,tlon2=tlatlon
        Alat1,Alat2,Alon1,Alon2=Alatlon
        if region=='Alaska':
            ax1 = fig.add_subplot(2,2,1,projection=projection) 
            ax2 = fig.add_subplot(2,2,3)
            ax1.set_title('Alaska')
            leftcorner_text=['Alaska']
            ax1.annotate(r"$\bf_{(A)}$",xy=(-0.27,0.93), xycoords='axes fraction', fontsize=15)
            #ax2.annotate(r"$\bf_{(A)}$ Alaska",xy=(-0.27,1), xycoords='axes fraction', fontsize=15)
        elif region=='Scan':
            ax1 = fig.add_subplot(2,2,2,projection=projection) 
            ax2 = fig.add_subplot(2,2,4)
            ax1.set_title('Scandinavia')
            leftcorner_text=['Scandinavia']
            ax1.annotate(r"$\bf_{(B)}$",xy=(-0.13,0.93), xycoords='axes fraction', fontsize=15)
            #ax2.annotate(r"$\bf_{(D)}$",xy=(-0.13,1), xycoords='axes fraction', fontsize=15)
        ###
        ### Start Plotting 
        row, col = 2, 2
        grid = row*col
        shading_grids=[corrs_mean*-1]
        indiv_colorbar=[True]*grid
        shading_level_grids= [np.linspace(-0.9,0.9,7)] * grid
        shading_level_grids= [np.linspace(-0.1,0.9,7)] * grid
        shading_level_grids= [np.linspace(-0.8,0.8,8)] * grid
        shading_level_grids= [[0,0.2,0.4,0.6,0.8,1]] * grid
        shading_level_grids= [[0,0.2,0.4,0.6,0.8]] * grid
        shading_level_grids= [[-0.2,0,0.2,0.4,0.6,0.8,1]] * grid
        mapcolors = ['#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']  # All red
        mapcolors = ['#eef7fa','#fff6e5','#fddbc7','#f4a582'] 
        mapcolors = ['#ffffff','#ffffff','#fddbc7','#f4a582','#d6604d','#b2182b'] # 2 white + 3 reds
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        mapcolor_grid = [cmap] * grid
        clabels_row = ['']*grid
        left_title = ['']*row
        top_title = [''] * col
        ind_titles = None
        projection=ccrs.PlateCarree(central_longitude=0); xsize=8; ysize=2
        #projection = ccrs.Stereographic(central_latitude=70, central_longitude=20)
        #xlims = (-180,180)
        ylims = (50,90)
        region_boxes = None
        region_boxes = [tools.create_region_box(tlat1, tlat2, tlon1, tlon2)] + [None]*10
        region_boxes_extra = [tools.create_region_box(Alat1, Alat2, Alon1, Alon2)] + [None]*10
        region_boxes_extra = None
        xylims = None
        matplotlib.rcParams['hatch.linewidth'] = 1;matplotlib.rcParams['hatch.color'] = 'lightgray'
        pval_hatches = [[[0, 0.01, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
        hspace=0
        if True: # Don't plot any shading in the maps - only show the region boxes
            for i in range(len(shading_grids)):
                shading_grids[i][:,:]=0
            indiv_colorbar=[False]*grid
            pvals_mean=[None]*10
            hspace=-0.2
        tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grids, clabels_row, top_titles=top_title, 
                        left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
                        region_boxes_extra=region_boxes_extra,leftcorner_text=leftcorner_text, ylim=ylims, xlim=xlims, quiver_grids=None,
                        pval_map=pvals_mean, pval_hatches=pval_hatches, fill_continent=False, coastlines=True,
                        contour_map_grids=None, contour_clevels=None, contour_lw=1.5, mask_ocean=False,
                        colorbar=False, indiv_colorbar=indiv_colorbar, ax_all=[ax1], pltf=fig, region_box_extra_color='royalblue')
        if False: ### Add indiviudal color bars
            if region=='Scan':
                cba = fig.add_axes([0.1, 0.55, 0.02, 0.22])
                cNorm = matplotlib.colors.Normalize(vmin=shading_level_grids[0][0], vmax=shading_level_grids[0][-1])
                scalarMap = matplotlib.cm.ScalarMappable(cmap=cmap)
                #scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
                cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, orientation='vertical', extend='neither')
                cticks=[i for i in shading_level_grids[0]]
                #cticks=[round(i,2) for i in shading_level_grid[0] if i!=0]
                cticks=shading_level_grids[0]
                cb1.ax.set_yticks(cticks)
                cba.yaxis.set_ticks_position('left')
                cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=7)
        ######
        ### Plot the scatter relationships
        xs,ys=[],[]
        no_of_var=len(vars)
        for i, var in enumerate(vars[-no_of_var:]):
            if len(AA_indices[var])>1:
                corr=r"($\rho$="+str(round(tools.correlation_nan(AA_indices[var],tempsd_indices[var]),2))+')'
            else:
                corr=''
            x=AA_indices[var]
            y=tempsd_indices[var]
            xs.append(x); ys.append(y)
            ax2.scatter(x, y, color=colors[var], zorder=zorders[var], s=sizes[var], alpha=0.1, marker=markers[var])
            ax2.scatter(x.mean(),y.mean(),color=colors[var], zorder=zorders[var]+1, s=sizes[var]+15, alpha=1, marker=markers[var], edgecolor='k', label='%s %s'%(vars_labels[var],corr))
        ## Get the whole correlation
        xs=[j for i in xs for j in i]
        ys=[j for i in ys for j in i]
        corr=round(tools.correlation_nan(xs,ys),2)
        ax2.annotate(r"$\rho$=%s (All)"%(str(corr)),xy=(0.01,0.95), xycoords='axes fraction', fontsize=10)
        if region=='Alaska':
            ax2.set_ylabel('Trends of day-to-day\ntemperature variability\n (K/decade)')
        ax2.set_xlabel('Trends of seasonal-mean temperature \n(K/decade)')
        #ax2.set_xlim(-0.2,2)
        #ax2.set_ylim(-0.3,0.2)
        #ax2.set_ylim(-0.4,0.2)
        ax2.legend(bbox_to_anchor=(-0.1,-0.6), ncol=2, loc='lower left', frameon=False, columnspacing=1,handletextpad=0.4, labelspacing=0.2,fontsize=7)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, zorder=-1)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, zorder=-1)
        for ax in [ax2]:
            for i in ['right', 'top']:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
            ax.tick_params(axis='x', direction="out", length=3, colors='black')
            ax.tick_params(axis='y', direction="out", length=3, colors='black')
    ### Save everything
    fig_name = 'figX_scatter_between_AA_tempsd'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=hspace) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0)


def get_data_for_plotting(region, vars, slopes):

    ### Select regions 
    if region=='Alaska':
        #tlat1, tlat2, tlon1, tlon2 = 58,75,-165,-90  # Alaska extended
        tlat1, tlat2, tlon1, tlon2 = 58,72,-167,-130  # Alaska only
        #Alat1, Alat2, Alon1, Alon2 = 65,80,-180,-140 # A: fit for Alaska only
        Alat1, Alat2, Alon1, Alon2 = 65,90,-180,180 # A: fit for Alaska only
        #xlims = (-180,180) # whole globe
        xlims = (-175,-35) # North America
    elif region=='Scan':
        #tlat1, tlat2, tlon1, tlon2 = 55,72,5,73 # Scandanavia+Europe
        #Alat1, Alat2, Alon1, Alon2 = 60,80,30,110 # A: fit for Scandavavia+Europe
        tlat1, tlat2, tlon1, tlon2 = 55,72,5,41 # Scandanavia only
        Alat1, Alat2, Alon1, Alon2 = 65,90,-180,180 # A: fit for Alaska only
        xlims = (-30,110) # Europe
    
    ### Get the indices between AA and subseasonal temperature variability change
    mask_data_raw = xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/DJF/%s%s_%s.nc'%('BESTSAT_daily_regrid_1x1','',1980))['training']
    tempsd_indices = {var:[] for var in vars}
    AA_indices = {var:[] for var in vars}
    AA_trends = {var:[] for var in vars}
    for i, var in enumerate(vars):
        ## For temp subseasonal variability index (need to apply land mask)
        tt='tempsd'
        trend_all=xr.concat(slopes[var][tt],dim='en') # along ensemble dimension
        land_mask=~np.isnan(mask_data_raw); #ocean_mask=np.isnan(data)
        land_mask= np.repeat(land_mask.values[np.newaxis,:, :],len(trend_all),axis=0)
        trend_all=xr.where(land_mask,trend_all,np.nan)
        lons=trend_all.longitude.values; lats=trend_all.latitude.values
        tempsd_index=ct.weighted_area_average(trend_all.values,tlat1,tlat2,tlon1,tlon2,lons,lats)
        tempsd_indices[var]=tempsd_index.data
        ## For AA index
        tt='AA'
        AA_trend=xr.concat(slopes[var][tt],dim='en') # along ensemble dimension
        if False: # Turn T2M trend map into meridional temperature gradient
            AA_trend=AA_trend.differentiate("latitude")
        AA_trends[var]=AA_trend # Save the maps for 1-point correlation before averaging
        AA_lons=AA_trend.longitude.values; AA_lats=AA_trend.latitude.values
        AA_index=ct.weighted_area_average(AA_trend.values,Alat1,Alat2,Alon1,Alon2,AA_lons,AA_lats)
        AA_indices[var]=AA_index.data

    ### Correlate the subseasonal variability to the AA maps (- to identify the most relavent regions)
    tempsd_index_all=[]
    AA_map_all=[]
    for i, var in enumerate(vars):
        ## Get the temp variability
        tempsd_index=tempsd_indices[var]
        tempsd_index_all.append(tempsd_index)
        ## Get the AA map
        AA_map=AA_trends[var]
        AA_map=AA_map.rename({'en':'time'})
        AA_map_all.append(AA_map)
    tempsd_index_all=[j for i in tempsd_index_all for j in i]
    tempsd_index_all=xr.DataArray(tempsd_index_all,dims=['time']) # Put back into xarray
    # Standardize the index
    ts_std=(tempsd_index_all-tempsd_index_all.mean())/tempsd_index_all.std()
    AA_map_all=xr.concat(AA_map_all,dim='time')
    results = tools.linregress_xarray(AA_map_all, tempsd_index_all, null_hypo=0)
    ### Do the correlation
    corrs_mean=results['correlation']
    pvals_mean=[results['pvalues']]

    tlatlon=(tlat1,tlat2,tlon1,tlon2)
    Alatlon=(Alat1,Alat2,Alon1,Alon2)

    return tempsd_indices,AA_indices,corrs_mean,pvals_mean, tlatlon, Alatlon, xlims

