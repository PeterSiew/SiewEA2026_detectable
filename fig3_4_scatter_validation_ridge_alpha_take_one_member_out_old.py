import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import multiprocessing
from importlib import reload
import scipy; from scipy import stats

import sys;sys.path.insert(0, '/Users/home/siewpe/codes/')
import scipy
import tools


if __name__ == "__main__":

    alphas=[20] # We always set the alpha=20
    alphas=[1] # We always set the alpha=20

    ### Read the save files, calulcate the RMSE
    Y_test_trues_save={alpha:None for alpha in alphas}
    Y_test_predicts_save={alpha:None for alpha in alphas}
    Y_lens_raw_save={alpha:None for alpha in alphas}
    records_save={alpha:None for alpha in alphas}
    for alpha in alphas:
        ## Load and save
        Y_test_trues=np.load('./alphas/Y_test_trues_%s.npy'%alpha,allow_pickle=True).item() ## Y_test_trues are the same for all ensemlbes for a single model
        Y_test_predicts=np.load('./alphas/Y_test_predicts_%s.npy'%alpha, allow_pickle=True).item()
        Y_lens_raw=np.load('./alphas/Y_lens_raw_%s.npy'%alpha)
        train_records=np.load('./alphas/train_records_%s.npy'%alpha)
        Y_test_trues_save[alpha]=Y_test_trues
        Y_test_predicts_save[alpha]=Y_test_predicts
        Y_lens_raw_save[alpha]=Y_lens_raw
        records_save[alpha]=train_records

    ### Validation plots (Figures 3 and 4)
    alpha=20
    period=30
    model_vars=['cesm1_tas_daily_en','canesm2_tas_daily_en','gfdlesm2m_tas_daily_en','gfdlcm3_tas_daily_en','mk360_tas_daily_en','ecearth_tas_daily_en'] # all models
    ensembles={'cesm1_tas_daily_en':range(1,17),'canesm2_tas_daily_en':range(1,17),'gfdlesm2m_tas_daily_en':range(1,17), 'gfdlcm3_tas_daily_en':range(1,17),
                'mk360_tas_daily_en':range(1,17),'ecearth_tas_daily_en':range(1,17)}; ensemble_no=16
    model_years={'cesm1_tas_daily_en':range(1950,2100),'canesm2_tas_daily_en':range(1950,2100),'gfdlesm2m_tas_daily_en':range(1950,2100),
                 'gfdlcm3_tas_daily_en':range(1950,2100),'mk360_tas_daily_en':range(1950,2100),'ecearth_tas_daily_en':range(1950,2100)}
    ## Picking everying with alpha=20
    Y_test_trues=Y_test_trues_save[alpha]
    Y_test_predicts=Y_test_predicts_save[alpha]
    Y_lens_raw=Y_lens_raw_save[alpha]
    train_records=records_save[alpha]
    Y_org_slopes={var:{} for var in model_vars}
    Y_true_slopes={var:{} for var in model_vars}
    Y_predict_slopes={var:{} for var in model_vars}
    ipdb.set_trace()
    for i, var in enumerate(model_vars):
        for en in ensembles[var]:
            en=str(en)
            var_bool=train_records[:,0]==var
            en_bool=train_records[:,1]==en
            Y_org=Y_lens_raw[var_bool&en_bool]
            Y_true=Y_test_trues[var+en]
            Y_predict=Y_test_predicts[var+en]
            ###
            ## The org raw timeseresi (like obs raw)
            Y_org_slope=[]
            Y_true_slope=[]
            Y_predict_slope=[]
            years_label=[]
            for j, idx in enumerate(Y_org):
                ts_sel=Y_org[j:j+period] # Y_org has a shape of 150 (1950-2100)
                if len(ts_sel)!=period:
                    continue
                ## The orginal timeseies
                slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                Y_org_slope.append(slope*10) # Change per decade
                ## The Y_forced true
                ts_sel=Y_true[j:j+period]
                slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                Y_true_slope.append(slope*10) # Change per decade
                ## The Y-forced predict
                ts_sel=Y_predict[j:j+period]
                slope, __, __, __, __ = stats.linregress(range(len(ts_sel)),ts_sel)
                Y_predict_slope.append(slope*10) # Change per decade
                years_cover=str(model_years[var][j]+1)+'-\n'+str(model_years[var][j+period-1]+1)
                years_label.append(years_cover)
            ## Save them into dict
            Y_org_slopes[var][en]=Y_org_slope
            Y_true_slopes[var][en]=Y_true_slope
            Y_predict_slopes[var][en]=Y_predict_slope

    ### Start plotting figure 2
    colors={'cesm1_tas_daily_en':'red', 'canesm2_tas_daily_en':'orange', 'gfdlesm2m_tas_daily_en':'gold',
            'gfdlcm3_tas_daily_en':'pink', 'mk360_tas_daily_en':'magenta', 'ecearth_tas_daily_en':'blueviolet'}
    vars_labels={'cesm1_tas_daily_en':'NCAR-CESM1', 'canesm2_tas_daily_en':'CCCma-CanESM2','gfdlcm3_tas_daily_en':'GFDL-CM3', 
            'gfdlesm2m_tas_daily_en':'GFDL-ESM2M', 'mk360_tas_daily_en':'CSIRO-MK360', 'ecearth_tas_daily_en':'SMHI/KNMI-EC-Earth'}
    ABCDE=['A','B','C','D','E','F','G']
    plt.close()
    fig, axs = plt.subplots(len(model_vars),1,figsize=(4,8))
    ## For the timeseries (first to second last row)
    for i, var in enumerate(model_vars):
        en='1'
        x=range(len(Y_org_slopes[var][en]))
        axs[i].plot(x,Y_org_slopes[var][en],color='k',label='Original',lw=0.5,linestyle='-')
        axs[i].plot(x,Y_true_slopes[var][en],color='forestgreen',label='Actual forced',lw=1)
        axs[i].plot(x,Y_predict_slopes[var][en],color='brown',label='Estimated forced',lw=2)
        ## Plot RMSE
        rmse=tools.rmse_nan(Y_predict_slopes[var][en], Y_true_slopes[var][en])
        corr=tools.correlation_nan(Y_predict_slopes[var][en], Y_true_slopes[var][en])
        rmse_label="RMSE=%s, %s=%s"%(str(round(rmse,3)),r"$\rho$",str(round(corr,2)))
        axs[i].annotate(rmse_label,xy=(0.6,0.8), xycoords='axes fraction',size=10)
        #axs[i].annotate("RMSE (Total, true forced): %s"%str(round(rmse,3)),xy=(0.3,0.9), xycoords='axes fraction',size=9)
        #axs[i].annotate("RMSE: %s %s %s"%(str(round(rmse_org_trues[var][en],3)),r"$ \Longrightarrow$",str(round(rmse,3))),xy=(0.5,0.8), xycoords='axes fraction',size=10)
        ## Set title
        axs[i].annotate("%s, member  %s"%(vars_labels[var],en),xy=(-0.12,1.07), xycoords='axes fraction')
        axs[i].axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        axs[i].set_ylabel('30-year trend\n(K/decade)',size=9)
    axs[0].legend(bbox_to_anchor=(-0.23,1.2), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=10)
    if False:## For the bar plot in the last row
        xadj=np.linspace(-0.1,0.1,len(model_vars))
        bar_width=0.04
        x_pos=np.array([1,1.3])
        xticklabels=['Between total and\ntrue forced', 'Between predicted and\ntrue forced']
        for i, var in enumerate(model_vars):
            bars = [np.mean([rmse_org_trues[var][str(en)] for en in ensembles[var]]),np.mean([rmse_predict_trues[var][str(en)] for en in ensembles[var]])]
            axs[-1].bar(x_pos+xadj[i], bars, bar_width, color=colors[var],label=vars_labels[var])
            axs[-1].set_xticks(x_pos)
            axs[-1].set_xticklabels(xticklabels)
            axs[-1].set_ylabel('RMSE\n(K/decade)',size=9)
        axs[-1].legend(bbox_to_anchor=(-0.21,-1.1), ncol=3, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=10)
    ## Set axxis
    for i, ax in enumerate(axs):
        for j in ['right', 'top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
        ax.tick_params(axis='x', direction="out", length=3, colors='black')
        ax.tick_params(axis='y', direction="out", length=3, colors='black')
        ax.annotate(r"$\bf_{(%s)}$"%ABCDE[i],xy=(-0.23,1.15), xycoords='axes fraction', fontsize=14)
    #for i, ax in enumerate(axs[0:-1]):
    for i, ax in enumerate(axs):
        ax.set_ylim(-0.23,0.15)
        ax.set_yticks([-0.2,-0.1,0,0.1])
        ax.set_xlim(x[0],x[-1])
        if i==len(axs[0:-1]): # 2nd last row
            ax.set_xticks(x[::20])
            ax.set_xticklabels(years_label[::20])
        else:
            ax.set_xticks(x[::20])
            ax.set_xticklabels([])
    ## Save fig
    fig_name = 'fig3_validation_timeseries_rmse'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.45) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.001)


    ### Plot the scatter plot of 30-year trends for validation (Figure 4)
    #trues,predicts,orgs=[],[],[]
    trues={model:[] for model in model_vars}
    predicts={model:[] for model in model_vars}
    orgs={model:[] for model in model_vars}
    for var in model_vars:
        for en in ensembles[var]:
            trues[var].append(Y_true_slopes[var][str(en)][0:])
            predicts[var].append(Y_predict_slopes[var][str(en)][0:])
            orgs[var].append(Y_org_slopes[var][str(en)][0:])
    ## Start plotting
    plt.close()
    #fig, (ax1, ax2) = plt.subplots(2,1,figsize=(4,8))
    fig, ax2 = plt.subplots(1,1,figsize=(3.5,3.5))
    true_true=[]; predict_predict=[]; org_org=[]
    for var in model_vars:
        true_plot=np.array(trues[var]).flatten()
        predict_plot=np.array(predicts[var]).flatten()
        org_plot=np.array(orgs[var]).flatten()
        rmse_org_true=tools.rmse_nan(true_plot,org_plot)
        rmse_predict_true=tools.rmse_nan(true_plot,predict_plot)
        corr_predict_true=tools.correlation_nan(true_plot,predict_plot)
        #ax1.scatter(true_plot, org_plot, s=0.3,color=colors[var])
        ax2.scatter(true_plot,predict_plot,s=0.3,color=colors[var])
        #rmse_label=" (RMSE: %s %s %s)"%(str(round(rmse_org_true,3)),r"$ \Longrightarrow$",str(round(rmse_predict_true,3)))
        rmse_label=" (RMSE=%s, %s=%s)"%(str(round(rmse_predict_true,3)),r"$\rho$",str(round(corr_predict_true,2)))
        ax2.scatter([-100],[-100], s=4,color=colors[var],label=vars_labels[var]+rmse_label)
        true_true.append(true_plot); predict_predict.append(predict_plot); org_org.append(org_plot)
    ## Adding the total RMSE
    true_true=np.array(true_true).flatten(); predict_predict=np.array(predict_predict).flatten(); org_org=np.array(org_org).flatten()
    #rmse_org_true=tools.rmse_nan(org_org,true_true)
    rmse_predict_true=tools.rmse_nan(predict_predict,true_true)
    corr_predict_true=tools.correlation_nan(predict_predict,true_true)
    abs_diff=predict_predict-true_true
    print('One SD of the absolute diff: ',np.std(abs_diff))
    rmse_label=" (RMSE=%s, %s=%s)"%(str(round(rmse_predict_true,3)),r"$\rho$",str(round(corr_predict_true,2)))
    ax2.scatter([-100],[-100],s=5,color='k',label='All models' + rmse_label)
    for ax in [ax2]:
        ax.plot([-100,100],[-100,100],linestyle='--',color='gray',linewidth=1.5,zorder=1)
        ax.set_xlim(-0.18,0.06)
        ax.set_ylim(-0.25,0.11)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        for i in ['right', 'top']:
            ax.spines[i].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
        ax.tick_params(axis='x', direction="out", length=3, colors='black')
        ax.tick_params(axis='y', direction="out", length=3, colors='black')
    #ax1.set_xlabel('Actual forced trends derived from\nlarge-ensemble simulation (K/decade)')
    #ax1.set_ylabel('Original trends without\nforced estimate (K/decade)')
    ax2.set_xlabel('Actual forced trends (K/decade)')
    ax2.set_ylabel('Estimated forced trends (K/decade)')
    ax2.legend(bbox_to_anchor=(-0.25,1), ncol=1, loc='lower left', frameon=False, columnspacing=0.5, handletextpad=0.2, labelspacing=0.1, fontsize=10)
    if False: ## Setting ABCDE
        ABCDE=['A','B','C']
        for i, ax in enumerate([ax2]):
            ax.annotate(r"$\bf_{(%s)}$"%ABCDE[i],xy=(-0.23,1.05), xycoords='axes fraction', fontsize=14)
    ## Save fig
    fig_name = 'fig4_scatter_validation'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.3) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.001)


