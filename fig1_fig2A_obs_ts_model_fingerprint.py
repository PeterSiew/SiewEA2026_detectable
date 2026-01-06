import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import multiprocessing
from copy import copy
import scipy; from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import tools
import create_timeseries as ct
import fig1_fig2A_obs_ts_model_fingerprint as fig1; reload(fig1)

### For both observations and models
lat1, lat2= 44,86; lon1, lon2= -180,180 # Basically it produces the same result as before
lat1, lat2= 45,85; lon1, lon2= -180,180 # (Actually it goes from 47 to 83)
season='SON'
season='DJF'
season='SONDJF'
relative_to_clim=False
relative_to_clim=True; relative_to_clim_pi=True # the PI is relative to the long-term climatology
years_clim = range(1980,2001) # to fit waccm6 which starts 1979

### Observations
#obs_var='best_SAT'; obs_years=range(1940,2020)
#obs_var="noaa_20CR_T2M_daily"; obs_years=range(1940,2015)
obs_var='ERA5_T2M_daily_regrid_1x1'; obs_years=range(1950,2024) # last-year 2023 means 2023-2024 SONDJF
obs1_var='BESTSAT_daily_regrid_1x1'; obs1_years=range(1950,2022) # last-year 2021 means 2021-2022 SONDJF
obs_en=''; obs1_en=''

### For large-ensemble models
#model_vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','mk360_tas_daily_en'] # Only for the models with 30 ensembles
### Default: All six models
model_vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en'] 

### Set ensembles
if True: ## Including Ec-Earth
    ## Full range (Figure S3)
    ensembles={'cesm1_tas_daily_en':range(1,41),'canesm2_tas_daily_en':range(1,51),'gfdlesm2m_tas_daily_en':range(1,31), 'gfdlcm3_tas_daily_en':range(1,21),
                'mk360_tas_daily_en':range(1,31),'ecearth_tas_daily_en':range(1,17)}; ensemble_no='hihi'
    ## Enesemble size=16 (equal number of members)
    ensembles={'cesm1_tas_daily_en':range(1,17),'canesm2_tas_daily_en':range(1,17),'gfdlesm2m_tas_daily_en':range(1,17), 'gfdlcm3_tas_daily_en':range(1,17),
                'mk360_tas_daily_en':range(1,17),'ecearth_tas_daily_en':range(1,17)}; ensemble_no=16
else: # for quick tesing
    ensembles={'cesm1_tas_daily_en':range(2,5),'canesm2_tas_daily_en':range(1,4),'gfdlesm2m_tas_daily_en':range(1,4), 
            'gfdlcm3_tas_daily_en':range(1,4),'mk360_tas_daily_en':range(1,4),'ecearth_tas_daily_en':range(1,4)}; ensemble_no=3
    print('Only run for 4 ensembles - quick test')

### Set years
pi_years=range(2015,3110) # For CanEM2-PI
model_years={var:range(1950,1951) for var in model_vars} # for testing
model_years={var:range(1950,2100) for var in model_vars} # Full range
model_yr=range(1950,2100) # Model and obs years for plotting

### Set WACCM6
add_waccm=True
if True:  # Using WACCM6
    waccm_years=range(1979,2014); waccm_ens=range(1,31)
    waccm_vars=['amip_WACCM6_daily_tas_en','amip_WACCM6_climsic_daily_tas_en']
else: # Using ECHAm5
    waccm_years=range(1979,2018); waccm_ens=range(1,31)
    waccm_vars=['amip_ECHAM5_daily_tas_en', 'amip_ECHAM5_daily_climsic_tas_en']

if __name__ == "__main__":

    ### Set Alpha for Ridge Regression
    if 'save_Y_test_trues_predicts' not in locals():
        # Go via here when this script is directly run
        ridge_alpha=60000
        ridge_alpha=20000 # Defailt
        fig1_obs=True
        fig2A_fingerprint=True
        save_Y_test_trues_predicts=False
        print('Reading alpha from main script, alpha=%s'%ridge_alpha)
        print('Reading ensemble from main script, ensemble_no=%s'%ensemble_no)
    else: ## Use by FigS1_best_alpha.py (the parameter are set by there)
        print('Reading alpha from other script, alpha=%s'%ridge_alpha)
        print('Reading ensemble from other script, ensemble_no=%s'%ensemble_no)

    Y_predict_obs,Y_predict_obs1,coeffs,Y_lens_raw,Y_lens_forced,train_records,data_pass,Y_obs_raw,Y_obs1_raw,Y_predict_pi,Y_predict_waccm,Y_test_trues,Y_test_predicts=fig1.model_ridge(
                                                                                                            model_vars,ensembles,ridge_alpha=ridge_alpha)
    #Y_predict_obs has a shape of 72 (1950-2022)
    #Y_lens_raw is the original Y_timeseries (6 model x 16 members x 150 years) 
    #Y_lens_forced has a shape of 14400 (6 model x 16 members x 150 years) - it is the forced average of individal models
    #Y_predict_pi has a shape of 1095 (1 model x 1095 years)
    #Y_predict_waccm has two keys (varying ice versus fixed ice), 30 members, and 35 years (1979-2014)
    #Y_test_trues and Y_test_predicts are take-one-out validation (which are used by the another script - figS1_best_alpha.py & fig3_4_scatter_validation_ridge_alpha.py)

    if save_Y_test_trues_predicts:
        np.save('./alphas/Y_test_trues_alpha%s_ens%s.npy'%(ridge_alpha,ensemble_no),Y_test_trues)
        np.save('./alphas/Y_test_predicts_alpha%s_ens%s.npy'%(ridge_alpha,ensemble_no),Y_test_predicts)
        np.save('./alphas/Y_lens_raw_alpha%s_ens%s.npy'%(ridge_alpha,ensemble_no),Y_lens_raw)
        np.save('./alphas/train_records_alpha%s_ens%s.npy'%(ridge_alpha,ensemble_no),train_records)
        print('Saving Y_true and Y_predict for alpha=%s and ensemble_no=%s'%(ridge_alpha,ensemble_no))
        print('')

    if fig1_obs: ### plot the obseved forced esimate (Figure 3; upper and lower panels)
        plt.close()
        alpha=0.6
        #fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6.5,8))
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6.5,7),height_ratios=[3,3,1.5])
        # For ERA5 raw timeseries
        ax1.plot(obs_years, Y_obs_raw, 'black', alpha=1, lw=1, label='ERA5 raw', zorder=10)
        # For forced estimate for ERA5
        ax1.plot(obs_years, Y_predict_obs, color='brown', lw=2,label='ERA5 forced',zorder=50)
        # For internal variability component (residual) from ERA5
        Y_obs_internal=Y_obs_raw-Y_predict_obs
        ax1.plot(obs_years, Y_obs_internal, 'gainsboro', linestyle='-',alpha=1, lw=1, label='ERA5 internal variability', zorder=20)
        if True: # Plot BEST
            ax1.plot(obs1_years, Y_predict_obs1, color='violet', lw=0.8,label='Berkeley-Earth forced',zorder=50)
        if True: # For model (shading)
            en_ts=Y_lens_forced.reshape(-1,len(model_yr)) # it has a shape of 96,150
            en_ts=en_ts.reshape(len(model_vars),ensemble_no,len(model_yr)).mean(axis=1) # mean across ensemble
            ts_min=np.percentile(en_ts,0,axis=0)
            ts_max=np.percentile(en_ts,100,axis=0)
            ax1.fill_between(model_yr, ts_min, ts_max, alpha=0.3, fc='forestgreen', label='Large-ensemble simulations')
        if False: # Plot the ensemble-mean
            ax1.plot(model_yr,en_ts.mean(axis=0),color='green',label='Enesemble average (6 models; 16 members each)')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        # For WACCM6
        if add_waccm:
            colors={waccm_vars[0]:'darkorange', waccm_vars[1]:'royalblue'}
            zorders={waccm_vars[0]:5, waccm_vars[1]:3}
            labels={waccm_vars[0]:'WACCM6 (observed sea ice)', waccm_vars[1]:'WACCM6 (climatological sea ice)'}
            for wvar in waccm_vars:
                waccm_mean=np.mean([Y_predict_waccm[wvar][en] for en in waccm_ens],axis=0)
                waccm_max=np.percentile([Y_predict_waccm[wvar][en] for en in waccm_ens],95,axis=0)
                waccm_min=np.percentile([Y_predict_waccm[wvar][en] for en in waccm_ens],5,axis=0)
                ax1.plot(waccm_years, waccm_mean, color=colors[wvar],lw=2, zorder=49)
                ax1.fill_between(waccm_years, waccm_min, waccm_max, alpha=alpha, fc=colors[wvar], zorder=zorders[wvar], label=labels[wvar])
        ax1.legend(bbox_to_anchor=(0,-0.03), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=10)
        nn=20
        ax1.set_xticks(model_yr[::nn])
        ax1.set_xticklabels([yr+1 for yr in model_yr[::nn]],rotation=0)
        ax1.set_xlim(model_yr[0],model_yr[-1])
        ax1.set_ylabel('Day-to-day\ntemperature\nvaraibility (K)')
        ax1.set_ylim(-1.6,0.6)
        ###
        ### Get the 30-year trend analysis (Figure 1; lower panel)
        period=20 
        period=40
        period=30 ### Default
        ## Forced observed ts (raw)
        slopes_obs_raw={}
        for i, yr in enumerate(obs_years):
            ts_sel=Y_obs_raw[i:i+period]
            if len(ts_sel)!=period:
                continue
           # Get the trend of the timeseries
            slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
            slopes_obs_raw[yr]=slope*10 # Change per decade
        ## For observed ts (forced) - For ERA5
        slopes_obs_forced={}
        for i, yr in enumerate(obs_years):
            ts_sel=Y_predict_obs[i:i+period]
            if len(ts_sel)!=period:
                continue
           # Get the trend of the timeseries
            slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
            slopes_obs_forced[yr]=slope*10 # Change per decade
        ## For observed ts (forced) - For BEST
        slopes_obs1_forced={}
        for i, yr in enumerate(obs1_years):
            ts_sel=Y_predict_obs1[i:i+period]
            if len(ts_sel)!=period:
                continue
           # Get the trend of the timeseries
            slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
            slopes_obs1_forced[yr]=slope*10 # Change per decade
        ## Forced observed ts (internal)
        slopes_obs_internal={}
        for i, yr in enumerate(obs_years):
            ts_sel=Y_obs_internal[i:i+period]
            if len(ts_sel)!=period:
                continue
           # Get the trend of the timeseries
            slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
            slopes_obs_internal[yr]=slope*10 # Change per decade
        ## For WACCM6 model
        slopes_waccm = {wvar:{en:{} for en in waccm_ens} for wvar in waccm_vars}
        for wvar in waccm_vars:
            for en in waccm_ens:
                for i, yr in enumerate(waccm_years):
                    ts_sel=Y_predict_waccm[wvar][en][i:i+period]
                    if len(ts_sel)!=period:
                       continue
                    slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                    slopes_waccm[wvar][en][yr]=slope*10 
        ## Get the slope for the large-ensemble simulations
        slopes_lens = {i:{} for i in range(len(model_vars))}
        for m in range(len(model_vars)):
            for i, yr in enumerate(model_yr): ## Is this correct?
                ts_sel=en_ts[m][i:i+period]
                if len(ts_sel)!=period:
                   continue
                slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                slopes_lens[m][yr]=slope*10 # Change per decade
        ## Get the slope for the PI trend
        slopes_pi=[]
        for i in range(len(Y_predict_pi)):
            ts_sel=Y_predict_pi[i:i+period]
            if len(ts_sel)!=period:
                continue
            slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
            slopes_pi.append(slope*10) # Change per decade
        ### Start plotting ax2
        if False: ## For LENS
            ys=[]
            for i in range(len(model_vars)):
                x=[*slopes_lens[i].keys()]
                y=[*slopes_lens[i].values()]
                ys.append(y)
            ys=np.array(ys)
            ymin=np.percentile(ys,0,axis=0)
            ymax=np.percentile(ys,100,axis=0)
            ax2.fill_between(x, ymin, ymax, alpha=0.3, fc='tomato')
        ## For waccm6
        if period==30:
            for wvar in waccm_vars:
                ys=[]
                for en in waccm_ens:
                    x=[*slopes_waccm[wvar][en].keys()]
                    y=[*slopes_waccm[wvar][en].values()]
                    ys.append(y)
                #ax2.plot(x,y,color=colors[wvar], zorder=10,lw=2)
                ys=np.array(ys)
                ymin=np.percentile(ys,5,axis=0)
                ymax=np.percentile(ys,95,axis=0)
                #ymean=np.percentile(ys,50,axis=0)
                ymean=np.mean(ys,axis=0)
                ax2.fill_between(x, ymin, ymax, alpha=alpha, fc=colors[wvar])
                ax2.plot(x, ymean, color=colors[wvar],lw=1)
        ## For obs raw (raw)
        x=[*slopes_obs_raw.keys()]
        y=[*slopes_obs_raw.values()]
        ax2.plot(x,y,color='black', zorder=10,lw=1)
        ## For obs (forced-ERA5)
        x=[*slopes_obs_forced.keys()]
        y=[*slopes_obs_forced.values()]
        ax2.plot(x,y,color='brown', zorder=10,lw=2)
        if period==30:
            ax2.fill_between(x, np.array(y)-0.0328, np.array(y)+0.0328, alpha=0.3, fc='brown',zorder=-10, label='Estimated error for ERA5 forced')
        ## For obs (forced - BEST)
        x=[*slopes_obs1_forced.keys()]
        y=[*slopes_obs1_forced.values()]
        ax2.plot(x,y,color='violet', zorder=10,lw=1)
        ## Set legend
        ax2.legend(bbox_to_anchor=(0.05,0.75), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=9)
        ## For obs (internal)
        x=[*slopes_obs_internal.keys()]
        y=[*slopes_obs_internal.values()]
        ax2.plot(x,y,color='gainsboro', zorder=10,lw=1)
        ## For PI (use the x from observations)
        bp2=ax2.boxplot([slopes_pi], positions=[x[-1]-0.5], showfliers=True, widths=0.5, whis=[5,95], patch_artist=True)
        for bp in [bp2]:
            for element in ['boxes', 'whiskers', 'caps']:
                plt.setp(bp[element], color='k', lw=2.5) # lw=3 in Fig.1
            for box in bp['boxes']:
                box.set(facecolor='k')
            plt.setp(bp['medians'], color='white', lw=2)
            plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor='k', markeredgecolor='k')
        # Set hline for PI
        ax2.axhline(y=np.percentile(slopes_pi,5), color='gray', linestyle='--', linewidth=1)
        ax2.axhline(y=np.percentile(slopes_pi,95), color='gray', linestyle='--', linewidth=1)
        ## Set others setting
        ax2.set_xlim(x[0],x[-1])
        ## Horizional lines at 0
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        ### Set axis
        xticks=[*slopes_obs_forced.keys()]
        xticklabels=[str(i+1)+'-\n'+str(i+period)[:] for i in xticks]
        nn=5
        ax2.set_xticks(xticks[::nn])
        ax2.set_xticklabels(xticklabels[::nn],rotation=0)
        if period==30:
            ax2.set_ylim(-0.15,0.12) # for 30-year trend
        elif period in [20,40]:
            ax2.set_ylim(-0.2,0.2) # for 20-year and 40-year trend
        ax2.set_xlim(xticks[0],xticks[-1])
        ax2.set_ylabel('%s-year trend\n(K/decade)'%period)
        ###
        ## Figure 1C - the relative importance (under fig1_obs)
        years_new=sorted([*slopes_obs_raw])
        forced_rels={}
        internal_rels={}
        for yr in years_new:
            ## if they have same sign, do the calculation
            if (np.sign(slopes_obs_raw[yr])==np.sign(slopes_obs_forced[yr])) & (np.sign(slopes_obs_raw[yr])==np.sign(slopes_obs_internal[yr])): 
                rel=slopes_obs_forced[yr]/slopes_obs_raw[yr]*100
                forced_rels[yr]=rel.item()
                rel=slopes_obs_internal[yr]/slopes_obs_raw[yr]*100
                internal_rels[yr]=rel.item()
            else: 
                ### Don't do the calculation
                #rel=np.nan
                #forced_rels[yr]=rel
                #internal_rels[yr]=rel
                ### Still do the calculation even they have different signs
                rel=slopes_obs_forced[yr]/slopes_obs_raw[yr]*100
                forced_rels[yr]=rel.item()
                rel=slopes_obs_internal[yr]/slopes_obs_raw[yr]*100
                internal_rels[yr]=rel.item()
        print('The average relative importance of the forced signal after 1976 is: ')
        print(np.nanmean([forced_rels[yr] for yr in range(1976,1995)]))
        print('The average relative importance of the internal variability component after 1976 is: ')
        print(np.nanmean([internal_rels[yr] for yr in range(1976,1995)]))
        ## Start the plotting
        x=np.arange(len(years_new))
        #bar_width=0.35
        #ax3.bar(x,[forced_rels[yr] for yr in years_new],bar_width,color='brown',edgecolor='brown',label='ERA5 forced')
        #bar_width=0.1
        #ax3.bar(x,[internal_rels[yr] for yr in years_new],bar_width,color='gainsboro',edgecolor='gainsboro',bottom=[forced_rels[yr] for yr in years_new],label='ERA5 internal variability')
        ax3.plot(x,[forced_rels[yr] for yr in years_new],color='brown',label='ERA5 forced',lw=2)
        ax3.plot(x,[internal_rels[yr] for yr in years_new],color='gainsboro',label='ERA5 internal variability',lw=2)
        #ax3.plot(x,np.array([internal_rels[yr] for yr in years_new])+np.array([forced_rels[yr] for yr in years_new]),color='black',label='total')
        xticks=x
        ax3.set_xticks(xticks[::nn])
        ax3.set_xticklabels(xticklabels[::nn],rotation=0)
        ax3.set_xlim(x[0],x[-1])
        #ax3.set_yticks([-200,-100,0,100,200])
        #ax3.legend(bbox_to_anchor=(0,0.9), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=7)
        ax3.set_yticks([-100,0,100,200])
        ax3.set_ylabel("Relative\nimportance (%)")
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        ## 
        ### Set the axis
        for ax in [ax1,ax2,ax3]:
            for i in ['right', 'top']:
                ax.spines[i].set_visible(False)
                ax.tick_params(axis='x', which='both',length=2)
                ax.tick_params(axis='y', which='both',length=2)
            ax.tick_params(axis='x', direction="out", length=3, colors='black')
            ax.tick_params(axis='y', direction="out", length=3, colors='black')
        ax1.annotate(r"$\bf_{(A)}$",xy=(-0.19,0.96), xycoords='axes fraction', fontsize=13)
        ax2.annotate(r"$\bf_{(B)}$",xy=(-0.19,0.96), xycoords='axes fraction', fontsize=13)
        ax3.annotate(r"$\bf_{(C)}$",xy=(-0.19,0.96), xycoords='axes fraction', fontsize=13)
        fig_name = 'fig1ABC_forced_timeseries_for_obs'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.3) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)


    ######
    ### Get the forced fingerprint - the beta coefficients with a large Alphas (Figure 2A)
    if fig2A_fingerprint: 
        if False:
            # restore the coeffs (where land grids are removed)
            land_grids=~np.isnan(data_pass).values.reshape(-1)
            new_coeffs=np.zeros(land_grids.shape)
            new_coeffs[land_grids==True]=coeffs[0:-1]
            new_coeffs[land_grids==False]=np.nan
            shading_grids = [data_pass.copy(data=new_coeffs.reshape(data_pass.shape))] #0:-1 is because the last grid is for intercept.
            # Consider the beta coeffcients when predicting both forced and internal components
        else:
            shading_grids = [data_pass.copy(data=coeffs[0:-1].reshape(data_pass.shape))] 
        col=1; row=len(shading_grids)
        ### Start the map plotting
        mask_ocean=False
        mask_ocean=True
        grid = row*col
        mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff','#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
        cmap= matplotlib.colors.ListedColormap(mapcolors)
        mapcolor_grid = [cmap] * grid
        shading_level_grid = [np.linspace(-0.0006,0.0006,13)]*grid # For 1x1 grid
        shading_level_grid = [np.linspace(-0.06,0.06,13)]*grid  # For 3x3 grid, alpha=20
        shading_level_grid = [np.linspace(-0.006,0.006,13)]*grid  # For 3x3 grid
        contour_grids = None
        contour_clevels = None
        clabels_row = [''] * grid
        top_title = [''] * col
        left_title = [''] * row
        leftcorner_text = ['Model fingerprint']
        leftcorner_text = ['']
        import cartopy.crs as ccrs
        projection=ccrs.PlateCarree(central_longitude=0); xsize=7; ysize=3
        xlim = [-80,120]
        xlim = [-180,180]
        ylim = (lat1+2,90)
        pval_map = None
        matplotlib.rcParams['hatch.linewidth'] = 1;matplotlib.rcParams['hatch.color'] = 'lightgray'
        pval_hatches = [[[0, 0.1, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
        fill_continent=False
        ### Add two more boxes
        if True:
            tlat1, tlat2, tlon1, tlon2 = 55,72,5,41 # Scandanavia only
            region_boxes = [tools.create_region_box(tlat1, tlat2, tlon1, tlon2)] + [None]*10
            tlat1, tlat2, tlon1, tlon2 = 58,72,-167,-130  # Alaska only
            region_boxes_extra = [tools.create_region_box(tlat1, tlat2, tlon1, tlon2)] + [None]*10
        else:
            region_boxes=[None]*10
            region_boxes_extra=[None]*10
        ###
        plt.close()
        fig = plt.figure(figsize=(col*xsize, row*ysize))
        ax1 = fig.add_subplot(row, col, 1, projection=projection) 
        tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                        left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
                        region_boxes_extra=region_boxes_extra,
                        leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,region_box_extra_color='red',
                        pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
                        contour_map_grids=contour_grids, contour_clevels=contour_clevels, contour_lw=1.5, mask_ocean=mask_ocean,
                        colorbar=False, indiv_colorbar=[False]*grid, ax_all=[ax1], pltf=fig)
        if True: # Add an extra box for revised letter (northern Urals)
            ## Boxes 2
            lat1, lat2, lon1, lon2 = 58,73,43,82  
            region_boxes = tools.create_region_box(lat1, lat2, lon1, lon2)
            lons_l, lats_l = region_boxes[0], region_boxes[1]
            ax1.plot(lons_l, lats_l, transform=ccrs.PlateCarree(), color='red', linewidth=1)
        if True: ### Setup the coloar bar
            cba = fig.add_axes([0.91, 0.39, 0.02, 0.2])
            cNorm  = matplotlib.colors.Normalize(vmin=shading_level_grid[0][0], vmax=shading_level_grid[0][-1])
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
            cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='vertical', extend='both')
            cb1.ax.set_yticklabels(cb1.ax.get_yticklabels(), fontsize=7)
        fig_name = 'fig2A_model_fingerprints'
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)



    ###
    if False: ### Plot the timeseries for PI (Figure S2)
        plt.close()
        fig, ax1= plt.subplots(1,1,figsize=(6,1.5))
        ax1.plot(pi_years,Y_predict_pi,color='black',lw=0.5,label='PI forced estimate')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_xticks(pi_years[::200])
        ax1.set_xlim(pi_years[0],pi_years[-1])
        ax1.set_xlabel('PI years')
        fig_name = 'figS2_forced_timeseries_for_PPI'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

    if False: ### Get the changing slopes over time following Blackport et al. 2021 (fixed the ending time of the timeseries to 2023/24)
        #obs_years_temp=range(1960,2024) # st_yr is fixed (always 1979); end_yr is changing
        obs_years_temp=range(1950,2010) # st_yr is chaning; end_yr is fixed (always 2024)
        ## Get the slope for org. timeseries
        slopes_obs_raw={}
        for i, st_yr in enumerate(obs_years_temp):
            idx=obs_years.index(st_yr)
            temp_ts=Y_obs_raw[idx:]
            #temp_ts=Y_obs_raw[0:idx]
            slope, __, __, __, __ = stats.linregress(range(len(temp_ts)),temp_ts)
            slopes_obs_raw[st_yr]=slope*10 # Change per decade
        ## Get the slope of forced compoennt
        slopes_obs_forced={}
        slopes_pi={}
        st_yrs=[]
        for i, st_yr in enumerate(obs_years_temp):
            idx=obs_years.index(st_yr)
            #temp_ts=Y_predict_obs[0:idx] # from 1950 to that year
            temp_ts=Y_predict_obs[idx:] # from that year to 2024
            slope, __, __, __, __ = stats.linregress(range(len(temp_ts)),temp_ts)
            slopes_obs_forced[st_yr]=slope*10
            ## Get the slope for the PI trend
            period=len(temp_ts)
            st_yrs.append(st_yr)
            if period not in slopes_pi:
                slopes_pi[st_yr]=[]
            for j in range(len(Y_predict_pi)):
                ts_sel=Y_predict_pi[j:j+period]
                if len(ts_sel)!=period:
                    continue
                slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                slopes_pi[st_yr].append(slope*10) # Change per decade
        ###
        plt.close()
        fig, ax1= plt.subplots(1,1,figsize=(6,1.5))
        ax1.plot(obs_years_temp,[slopes_obs_raw[yr] for yr in obs_years_temp],color='black',lw=0.5,label='ERA5 total')
        ax1.plot(obs_years_temp,[slopes_obs_forced[yr] for yr in obs_years_temp],color='brown',lw=0.5,label='ERA5 forced')
        q05s,q95s=[],[]
        for yr in obs_years_temp:
            q05=np.percentile(slopes_pi[yr],5)
            q95=np.percentile(slopes_pi[yr],95)
            q05s.append(q05); q95s.append(q95)
        ax1.plot(obs_years_temp,q05s,color='red')
        ax1.plot(obs_years_temp,q95s,color='red')
        #bp2=ax2.boxplot([slopes_pi[yr] for yr in obs_years_temp], positions=obs_years_temp, showfliers=True, widths=0.5, whis=[5,95], patch_artist=True)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        fig_name = 'fig_moving_trend'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2) # hspace is the vertical
        plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

        

def model_ridge(vars,ensembles,ridge_alpha=100):
    #print(vars)

    ### Get land mask
    data = xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/DJF/%s%s_%s.nc'%('BESTSAT_daily_regrid_1x1','',1980))['training']
    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
    land_mask=~np.isnan(data)

    ### Get the climatology of models
    print('Start reading CMIP5 large-ensemble models')
    climatology_models = {var:{en:[] for en in ensembles[var]} for i, var in enumerate(vars)}
    if relative_to_clim: ### Create an anomaly relative to a certain period for each ensemble member
        for i, var in enumerate(vars):
            for en in ensembles[var]:
                datas=[]
                for yr in years_clim:
                    data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    datas.append(data)
                climatology_models[var][en] = xr.concat(datas, dim='time').mean(dim='time')
    ### Start reading models
    X_raw,Y_raw=[],[]
    records=[]
    for var in vars:
        print(var)
        for en in ensembles[var]:
            for yr in model_years[var]: # Read the data first (to avoid repeating the reading process)
                data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                if relative_to_clim:
                    data = data-climatology_models[var][en]
                    #data = data-climatology_obs
                data = xr.where(land_mask, data, np.nan)
                data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                # Get the X
                cos_lats=np.cos(np.radians(data.latitude)) 
                X_raw.append((data*cos_lats).values.reshape(-1))
                # Get the Y (index)
                lons=data.longitude.values; lats=data.latitude.values
                index=ct.weighted_area_average(data.values[np.newaxis,:,:],lat1,lat2,lon1,lon2,lons,lats)
                Y_raw.append(index.item())
                records.append((var,en,yr))
    X_raw=np.array(X_raw); Y_raw=np.array(Y_raw); records=np.array(records)
    data_pass=copy(data)
    ## Set the ensemble mean in Y
    Y_forced=np.zeros(Y_raw.shape)
    Y_residual=np.zeros(Y_raw.shape)
    for var in vars:
        for yr in model_years[var]: 
            # Capture the ensemble mean for each year for each model
            idx = (records[:,0]==var) & (records[:,2]==str(yr))
            Y_mean=Y_raw[idx].mean()
            Y_forced[idx]=Y_mean # has a size of 20
            Y_residual[idx]=Y_raw[idx]-Y_mean
    if False: # Combine Y_forced and Y_residual for Y_prediction
        Y_forced=np.column_stack((Y_forced,Y_residual)) # We don't call it Y_raw here is it already exists
    else:
        Y_forced=Y_forced

    ### For observations
    print('Start reading Observations')
    ## Get the climatology for observations
    if relative_to_clim:
        datas=[]
        for yr in years_clim:
            data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,obs_var,obs_en,yr))['training']
            data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            datas.append(data)
        climatology_obs = xr.concat(datas, dim='time').mean(dim='time')
    ## Start reading observations
    X_obs_raw, Y_obs_raw = [], []
    for yr in obs_years:
        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,obs_var,obs_en,yr))['training']
        data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        if relative_to_clim:
            data=data-climatology_obs
        data = xr.where(land_mask, data, np.nan)
        data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        cos_lats=np.cos(np.radians(data.latitude)) 
        X_obs_raw.append((data*cos_lats).values.reshape(-1))
        # Get the Y (index)
        lons=data.longitude.values; lats=data.latitude.values
        index=ct.weighted_area_average(data.values[np.newaxis,:,:],lat1,lat2,lon1,lon2,lons,lats)
        Y_obs_raw.append(index.item())
    X_obs_raw=np.array(X_obs_raw)
    Y_obs_raw=np.array(Y_obs_raw)

    ## Get the climatology for observations 1 (BEST)
    if relative_to_clim:
        datas=[]
        for yr in years_clim:
            data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,obs1_var,obs1_en,yr))['training']
            data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            datas.append(data)
        climatology_obs1 = xr.concat(datas, dim='time').mean(dim='time')
    ## Start reading observations
    X_obs1_raw, Y_obs1_raw = [], []
    for yr in obs1_years:
        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,obs1_var,obs1_en,yr))['training']
        data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        if relative_to_clim:
            data=data-climatology_obs1
        data = xr.where(land_mask, data, np.nan)
        data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        cos_lats=np.cos(np.radians(data.latitude)) 
        X_obs1_raw.append((data*cos_lats).values.reshape(-1))
        # Get the Y (index)
        lons=data.longitude.values; lats=data.latitude.values
        index=ct.weighted_area_average(data.values[np.newaxis,:,:],lat1,lat2,lon1,lon2,lons,lats)
        Y_obs1_raw.append(index.item())
    X_obs1_raw=np.array(X_obs1_raw)
    Y_obs1_raw=np.array(Y_obs1_raw)

    ### preindustrial control
    print('Start reading PI-control')
    #pi_var='cesm1_tas_daily_PI'; pi_years=range(1700,2200)
    pi_var='canesm2_tas_daily_PI'
    pi_en=''
    if relative_to_clim_pi: 
        if False: # Relative to the LENS (CanESM2)
            climatology_pi=xr.concat([climatology_models['canesm2_tas_daily_en'][en] for en in ensembles['canesm2_tas_daily_en']],dim='en').mean(dim='en')
        else: # Relative to the PI climatology
            datas=[]
            for yr in pi_years:
                data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,pi_var,pi_en,yr))['training']
                data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                datas.append(data)
            climatology_pi=xr.concat(datas, dim='time').mean(dim='time')
    ### Start reading the preindustrial control
    X_pi_raw, Y_pi_raw = [], []
    for yr in pi_years:
        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,pi_var,pi_en,yr))['training']
        data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        if relative_to_clim_pi:
            data=data-climatology_pi
        ## Get the X (weighted maps)
        data = xr.where(land_mask, data, np.nan)
        data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
        cos_lats=np.cos(np.radians(data.latitude)) 
        X_pi_raw.append((data*cos_lats).values.reshape(-1))
        # Get the Y (index)
        lons=data.longitude.values; lats=data.latitude.values
        index=ct.weighted_area_average(data.values[np.newaxis,:,:],lat1,lat2,lon1,lon2,lons,lats)
        Y_pi_raw.append(index.item())
    X_pi_raw=np.array(X_pi_raw)
    Y_pi_raw=np.array(Y_pi_raw)

    ### Start the WACCM6 model
    if add_waccm:
        print('Start reading WACCM6 simulation')
        ### Get the climatology of models
        if relative_to_clim: ### Create an anomaly relative to a certain period for each ensemble member
            climatology_waccm={var:{} for var in waccm_vars}
            for var in waccm_vars:
                for en in waccm_ens:
                    datas=[]
                    for yr in years_clim:
                        data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                        data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                        datas.append(data)
                    climatology_waccm[var][en]=xr.concat(datas, dim='time').mean(dim='time')
        X_waccm_raw = {var:{} for var in waccm_vars}
        Y_waccm_raw = {var:{} for var in waccm_vars}
        for var in waccm_vars:
            for en in waccm_ens:
                X_waccm_append, Y_waccm_append = [], []
                for yr in waccm_years:
                    data=xr.open_dataset('/mnt/data/data_a/t2m_variability_training/training_tempsd/%s/%s%s_%s.nc'%(season,var,en,yr))['training']
                    data=data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    if relative_to_clim:
                        data = data-climatology_waccm[var][en]
                    data = xr.where(land_mask, data, np.nan)
                    data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    cos_lats=np.cos(np.radians(data.latitude)) 
                    X_waccm_append.append((data*cos_lats).values.reshape(-1))
                    # Get the Y (index)
                    lons=data.longitude.values; lats=data.latitude.values
                    index=ct.weighted_area_average(data.values[np.newaxis,:,:],lat1,lat2,lon1,lon2,lons,lats)
                    Y_waccm_append.append(index.item())
                X_waccm_raw[var][en]=np.array(X_waccm_append)
                Y_waccm_raw[var][en]=np.array(Y_waccm_append)
    else:
        X_waccm_raw=None
        Y_waccm_raw=None

    ### remove the land grids (all nan values) - is is better to them as 0
    if False: 
        mask=~np.isnan(X_raw[0])
        X_raw=X_raw[:,mask]
        X_obs_raw=X_obs_raw[:,mask]
        X_obs1_raw=X_obs1_raw[:,mask]
        X_pi_raw=X_pi_raw[:,mask]
        for var in waccm_vars:
            for en in waccm_ens:
                X_waccm_raw[var][en]=X_waccm_raw[var][en][:,mask]

    #### Standardize the data - for observations and all models
    ## For models
    X_mean=X_raw.mean(axis=0); X_std=X_raw.std(axis=0)
    X_models=(X_raw-X_mean)/X_std
    Y_mean=Y_forced.mean(axis=0); Y_std=Y_forced.std(axis=0)
    Y_models=(Y_forced-Y_mean)/Y_std
    ## For obs and PI (using the X_mean and X_std obtained from models)
    X_obs=(X_obs_raw-X_mean) / X_std
    X_obs1=(X_obs1_raw-X_mean) / X_std
    X_pi=(X_pi_raw-X_mean) / X_std
    ## For WACCM
    if add_waccm:
        X_waccm={var:{} for var in waccm_vars}
        for var in waccm_vars:
            for en in waccm_ens:
                X_waccm[var][en]=(X_waccm_raw[var][en]-X_mean) / X_std

    #### Add a single column for intercept after standardsize
    X_models=np.column_stack((X_models,np.ones(X_models.shape[0]))) # For training
    X_obs=np.column_stack((X_obs,np.ones(X_obs.shape[0])))
    X_obs1=np.column_stack((X_obs1,np.ones(X_obs1.shape[0])))
    X_pi=np.column_stack((X_pi,np.ones(X_pi.shape[0])))
    if add_waccm:
        for var in waccm_vars:
            for en in waccm_ens:
                X_waccm[var][en]=np.column_stack((X_waccm[var][en],np.ones(X_waccm[var][en].shape[0])))

    ### Replace nan data by 0
    if True: 
        X_models[np.isnan(X_models)]=0
        X_obs[np.isnan(X_obs)]=0
        X_obs1[np.isnan(X_obs1)]=0
        X_pi[np.isnan(X_pi)]=0
        if add_waccm:
            for var in waccm_vars:
                for en in waccm_ens:
                    X_waccm[var][en][np.isnan(X_waccm[var][en])] = 0
    
    ### Setup the Ridge regression training
    Ridge_regress=False # ANN
    Ridge_regress=True # Ridge Regression
    if Ridge_regress:
        print("Start the Ridge Regression")
        clf = Ridge(alpha=ridge_alpha).fit(X_models,Y_models)
        coeffs = clf.coef_
    else:
        print("Start the ANN")
        ### ANN setting
        solver='adam' 
        ann_alpha=0.001; learning_rate=0.01
        batch_size=200
        ann_nodes=(10,10)
        random_no=1
        ann_shuffle=True 
        early_stop=True
        clf=MLPRegressor(hidden_layer_sizes=ann_nodes, learning_rate_init=learning_rate,solver=solver,early_stopping=early_stop,
                learning_rate='constant',n_iter_no_change=10,batch_size=batch_size,max_iter=1000, alpha=ann_alpha, shuffle=ann_shuffle, 
                        tol=1e-5,verbose=False,random_state=random_no*np.random.randint(1000)).fit(X_models,Y_models) # tol=1e-5
        coeffs=True
    ### Start the prediction
    Y_predict_obs=clf.predict(X_obs)*Y_std + Y_mean
    Y_predict_obs1=clf.predict(X_obs1)*Y_std + Y_mean
    Y_predict_pi=clf.predict(X_pi)*Y_std + Y_mean
    Y_predict_waccm={var:{} for var in waccm_vars}
    if add_waccm:
        for var in waccm_vars:
            for en in waccm_ens:
                Y_predict_waccm[var][en]=clf.predict(X_waccm[var][en])*Y_std + Y_mean
    else:
        Y_predict_waccm=None

    ### Take-one-simulation-out for training and validation
    Y_test_trues = {}; Y_test_predicts = {}
    if True: 
        print("Starting the Take-one-model-out-simulaiton")
        for var in vars:
            var_mask_test=var==records[:,0]
            idx_test=var_mask_test.nonzero()[0]
            var_mask_train=~var_mask_test
            idx_train=var_mask_train.nonzero()[0]
            ## Select them according to idx
            ## The raw model X is called X_raw; the raw model Y is called Y_forced (they are not standardized)
            X_test=X_raw[idx_test]; Y_test=Y_forced[idx_test]
            X_train=X_raw[idx_train]; Y_train=Y_forced[idx_train]
            ## Standardize
            X_train_mean=X_train.mean(axis=0); Y_train_mean=Y_train.mean(axis=0)
            X_train_std=X_train.std(axis=0); Y_train_std=Y_train.std(axis=0)
            X_train_standard=(X_train-X_train_mean)/X_train_std
            Y_train_standard=(Y_train-Y_train_mean)/Y_train_std
            X_test_standard=(X_test-X_train_mean)/X_train_std
            ## Replace nan data by 0
            X_train_standard[np.isnan(X_train_standard)]=0
            X_test_standard[np.isnan(X_test_standard)]=0
            ## Add an intercept
            X_train_standard=np.column_stack((X_train_standard,np.ones(X_train_standard.shape[0])))
            X_test_standard=np.column_stack((X_test_standard,np.ones(X_test_standard.shape[0])))
            ## Do the prediction
            if Ridge_regress:
                clf = Ridge(alpha=ridge_alpha).fit(X_train_standard,Y_train_standard)
            else:
                learning_rate=0.1
                clf=MLPRegressor(hidden_layer_sizes=ann_nodes, learning_rate_init=learning_rate,solver=solver,early_stopping=early_stop,
                        learning_rate='constant',n_iter_no_change=10,batch_size=batch_size,max_iter=1000, alpha=ann_alpha, shuffle=ann_shuffle, 
                                tol=1e-5,verbose=False,random_state=random_no*np.random.randint(1000)).fit(X_train_standard,Y_train_standard) # tol=1e-5
            Y_test_predict=clf.predict(X_test_standard)*Y_train_std+Y_train_mean
            Y_test_predicts[var]=Y_test_predict
            Y_test_trues[var]=Y_test
    elif False: ### Take one-ensemble out simulation (not we are using this)
        for var in vars:
            print("Starting the take-one-ensemble-out-simulaiton %s"%var)
            for en in ensembles[var]:
                en=str(en)
                var_mask_test=var==records[:,0]
                en_mask_test=en==records[:,1]
                idx_test=(var_mask_test&en_mask_test).nonzero()[0]
                idx_train=np.array([i for i in range(len(records)) if i not in idx_test])
                #idx_train=np.arange(len(records)) # Just for testing
                ## Select them according to idx
                X_test=X_raw[idx_test]; Y_test=Y_forced[idx_test]
                X_train=X_raw[idx_train]; Y_train=Y_forced[idx_train]
                ## Standardize
                X_train_mean=X_train.mean(axis=0); X_train_std=X_train.std(axis=0)
                Y_train_mean=Y_train.mean(axis=0); Y_train_std=Y_train.std(axis=0)
                X_train_standard=(X_train-X_train_mean)/X_train_std
                Y_train_standard=(Y_train-Y_train_mean)/Y_train_std
                X_test_standard=(X_test-X_train_mean)/X_train_std
                ## Replace d ata by 0
                X_train_standard[np.isnan(X_train_standard)]=0
                X_test_standard[np.isnan(X_test_standard)]=0
                ## Add an intercept
                X_train_standard=np.column_stack((X_train_standard,np.ones(X_train_standard.shape[0])))
                X_test_standard=np.column_stack((X_test_standard,np.ones(X_test_standard.shape[0])))
                ## Do the prediction
                if Ridge_regress:
                    clf = Ridge(alpha=ridge_alpha).fit(X_train_standard,Y_train_standard)
                else:
                    clf=MLPRegressor(hidden_layer_sizes=ann_nodes, learning_rate_init=learning_rate,solver=solver,early_stopping=False,
                            learning_rate='constant',n_iter_no_change=10,batch_size=batch_size,max_iter=1000, alpha=ann_alpha, shuffle=ann_shuffle, 
                                    tol=1e-5,verbose=False,random_state=random_no*np.random.randint(1000)).fit(X_train_standard,Y_train_standard) # tol=1e-5
                Y_test_predict=clf.predict(X_test_standard)*Y_train_std+Y_train_mean
                Y_test_predicts[var+en]=Y_test_predict
                Y_test_trues[var+en]=Y_test

    return Y_predict_obs, Y_predict_obs1, coeffs, Y_raw, Y_forced, records, data_pass, Y_obs_raw, Y_obs1_raw, Y_predict_pi, Y_predict_waccm, Y_test_trues, Y_test_predicts

def backup():

    #Y_predict_obs, coeffs, Y_forced, records, Y_predict_pi, Y_predict_waccm = {}, {}, {}, {}, {}, {}
    #Y_test_trues, Y_test_predicts={}, {}
    #for var in var_types:
    #    Y_predict_obs[var],coeffs[var], Y_forced[var],records[var],data_pass,Y_obs_raw,Y_predict_pi[var],Y_predict_waccm[var],Y_test_trues[var],Y_test_predicts[var]=fig1.model_ridge(var_types[var])

    ### models
    var_types={'ghg':['MIROC6_histGHG_tas_daily_en','HadGEM3_histGHG_tas_daily_en','CanESM5_histGHG_tas_daily_en'],
        'aer':['MIROC6_histaer_tas_daily_en','HadGEM3_histaer_tas_daily_en','CanESM5_histaer_tas_daily_en'],
        'nat':['MIROC6_histnat_tas_daily_en','HadGEM3_histnat_tas_daily_en','CanESM5_histnat_tas_daily_en'],
        #'hist':['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en']}
        'hist':['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en']}
    var_types={'hist':var_types['hist']}
    ###
    ghg_vars_en={'MIROC6_histGHG_tas_daily_en':[i for i in range(1,51) if i not in [42]],
                'HadGEM3_histGHG_tas_daily_en':[i for i in range(1,26) if i not in [6,7,8,9,10]],
                'CanESM5_histGHG_tas_daily_en':range(1,11)}
    aer_vars_en={'MIROC6_histaer_tas_daily_en':range(1,11),'HadGEM3_histaer_tas_daily_en':[i for i in range(1,26) if i not in [6,7,8,9,10]],
                'CanESM5_histaer_tas_daily_en':range(1,11)}
    nat_vars_en={'MIROC6_histnat_tas_daily_en':range(1,51),'HadGEM3_histnat_tas_daily_en':[i for i in range(1,21) if i not in [31,32,37]],
                'CanESM5_histnat_tas_daily_en':range(1,11)}
    hist_vars_en={'cesm1_tas_daily_en':range(2,18),'canesm2_tas_daily_en':range(1,17),'gfdlesm2m_tas_daily_en':range(1,17),
                 'gfdlcm3_tas_daily_en':range(1,17),'mk360_tas_daily_en':range(1,17),'ecearth_tas_daily_en':range(1,17)}
    hist_vars_en={'cesm1_tas_daily_en':range(2,22),'canesm2_tas_daily_en':range(1,21),'gfdlesm2m_tas_daily_en':range(1,21),
                 'gfdlcm3_tas_daily_en':range(1,21),'mk360_tas_daily_en':range(1,21)}
    ensembles={**ghg_vars_en, **aer_vars_en, **nat_vars_en, **hist_vars_en}
    ###
    ghg_vars_years={'MIROC6_histGHG_tas_daily_en':range(1850,2100),'HadGEM3_histGHG_tas_daily_en':range(1850,2020),'CanESM5_histGHG_tas_daily_en':range(1950,2020)}
    aer_vars_years={'MIROC6_histaer_tas_daily_en':range(1850,2100),'HadGEM3_histaer_tas_daily_en':range(1850,2020),'CanESM5_histaer_tas_daily_en':range(1950,2020)}
    nat_vars_years={'MIROC6_histnat_tas_daily_en':range(1850,2100),'HadGEM3_histnat_tas_daily_en':range(1850,2020), 'CanESM5_histnat_tas_daily_en':range(1950,2020)}
    hist_var_years={'cesm1_tas_daily_en':range(1950,2100),'canesm2_tas_daily_en':range(1950,2100),'gfdlesm2m_tas_daily_en':range(1950,2100),
                 'gfdlcm3_tas_daily_en':range(1950,2100),'mk360_tas_daily_en':range(1950,2100),'ecearth_tas_daily_en':range(1950,2100)}
    model_years={**ghg_vars_years, **aer_vars_years, **nat_vars_years, **hist_var_years}
    ###
