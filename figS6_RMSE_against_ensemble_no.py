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

ensemble_numbers=[5,10,15]
ensemble_numbers=[i for i in range(1,17)]+[999] #999 means using all ensembles

if False:
    for en_num in ensemble_numbers:
        ensembles={'cesm1_tas_daily_en':range(1,en_num+1),'canesm2_tas_daily_en':range(1,en_num+1),'gfdlesm2m_tas_daily_en':range(1,en_num+1), 'gfdlcm3_tas_daily_en':range(1,en_num+1),
                    'mk360_tas_daily_en':range(1,en_num+1),'ecearth_tas_daily_en':range(1,en_num+1)}; ensemble_no=en_num
        if en_num==999: ### the full range
            ensembles={'cesm1_tas_daily_en':range(1,41),'canesm2_tas_daily_en':range(1,51),'gfdlesm2m_tas_daily_en':range(1,31), 'gfdlcm3_tas_daily_en':range(1,21),
                        'mk360_tas_daily_en':range(1,31),'ecearth_tas_daily_en':range(1,17)}; ensemble_no=en_num
        ridge_alpha=20000
        fig1_obs=False
        fig2A_fingerprint=False
        save_Y_test_trues_predicts=True
        exec(open("./fig1_fig2A_obs_ts_model_fingerprint.py").read())


if True:  ### Ploting fig S1 for RMSE changes with alpha values
    ### Read the save files, calulcate the RMSE
    rmses=[] # This is the RMSE for the whole timeseies, not for trends)
    alpha=20000
    period=30
    model_years={'cesm1_tas_daily_en':range(1950,2100),'canesm2_tas_daily_en':range(1950,2100),'gfdlesm2m_tas_daily_en':range(1950,2100),
                 'gfdlcm3_tas_daily_en':range(1950,2100),'mk360_tas_daily_en':range(1950,2100),'ecearth_tas_daily_en':range(1950,2100)}
    Y_true_slopes_FINAL={}
    Y_predict_slopes_FINAL={}
    Y_org_slopes_FINAL={}
    for en_num in ensemble_numbers:
        print(en_num)
        ## Load the data
        Y_test_trues=np.load('./alphas/Y_test_trues_alpha%s_ens%s.npy'%(alpha,en_num),allow_pickle=True).item()
        Y_test_predicts=np.load('./alphas/Y_test_predicts_alpha%s_ens%s.npy'%(alpha,en_num), allow_pickle=True).item()
        Y_lens_raw=np.load('./alphas/Y_lens_raw_alpha%s_ens%s.npy'%(alpha,en_num))
        train_records=np.load('./alphas/train_records_alpha%s_ens%s.npy'%(alpha,en_num))
        ##
        model_vars=list(Y_test_trues.keys())
        Y_test_trues=np.concatenate([Y_test_trues[var] for var in model_vars]) # if en_num=999; they have different shape
        Y_test_predicts=np.concatenate([Y_test_predicts[var] for var in model_vars])
        ###
        Y_org_slopes={var:{} for var in model_vars}
        Y_true_slopes={var:{} for var in model_vars}
        Y_predict_slopes={var:{} for var in model_vars}
        ensembles={'cesm1_tas_daily_en':range(1,en_num+1),'canesm2_tas_daily_en':range(1,en_num+1),'gfdlesm2m_tas_daily_en':range(1,en_num+1), 'gfdlcm3_tas_daily_en':range(1,en_num+1),
                    'mk360_tas_daily_en':range(1,en_num+1),'ecearth_tas_daily_en':range(1,en_num+1)}; ensemble_no=en_num
        if en_num==999: ### the full range
            ensembles={'cesm1_tas_daily_en':range(1,41),'canesm2_tas_daily_en':range(1,51),'gfdlesm2m_tas_daily_en':range(1,31), 'gfdlcm3_tas_daily_en':range(1,21),
                        'mk360_tas_daily_en':range(1,31),'ecearth_tas_daily_en':range(1,17)}; ensemble_no=en_num
        for i, var in enumerate(model_vars):
            for en in ensembles[var]:
                en=str(en)
                var_bool=train_records[:,0]==var
                en_bool=train_records[:,1]==en
                Y_org=Y_lens_raw[var_bool&en_bool]
                Y_true=Y_test_trues[var_bool&en_bool]
                Y_predict=Y_test_predicts[var_bool&en_bool]
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
        Y_org_slopes_FINAL[en_num]=Y_org_slopes
        Y_true_slopes_FINAL[en_num]=Y_true_slopes
        Y_predict_slopes_FINAL[en_num]=Y_predict_slopes

    ### Get RMSES (red) and correlation (blue)
    ensemble_numbers=[i for i in range(1,17)]+[999] #999 means using all ensembles
    ensemble_numbers=[i for i in range(1,17)]
    #rmses={en_num:{} for en_num in ensemble_numbers}
    rmses={var:[] for var in model_vars}
    corrs={var:[] for var in model_vars}
    rmses_all=[]
    corrs_all=[]
    for en_num in ensemble_numbers:
        ensembles={'cesm1_tas_daily_en':range(1,en_num+1),'canesm2_tas_daily_en':range(1,en_num+1),'gfdlesm2m_tas_daily_en':range(1,en_num+1), 'gfdlcm3_tas_daily_en':range(1,en_num+1),
                'mk360_tas_daily_en':range(1,en_num+1),'ecearth_tas_daily_en':range(1,en_num+1)}; ensemble_no=en_num
        if en_num==999: ### the full range
            ensembles={'cesm1_tas_daily_en':range(1,41),'canesm2_tas_daily_en':range(1,51),'gfdlesm2m_tas_daily_en':range(1,31), 'gfdlcm3_tas_daily_en':range(1,21),
                        'mk360_tas_daily_en':range(1,31),'ecearth_tas_daily_en':range(1,17)}; ensemble_no=en_num
        Xss,Yss=[],[]
        for i, var in enumerate(model_vars):
            Xs,Ys=[],[]
            for en in ensembles[var]:
                #X=Y_true_slopes_FINAL[en_num][var][str(en)]
                #X=Y_true_slopes_FINAL[16][var][str(en)] # true ensemble average by considring 16 members
                X=Y_true_slopes_FINAL[999][var][str(en)] # true ensemble average by considering all members (very similar results compared to 16)
                Xs.append(X)
                Y=Y_predict_slopes_FINAL[en_num][var][str(en)]
                Ys.append(Y)
            Xs_flatten=np.array(Xs).flatten()
            Ys_flatten=np.array(Ys).flatten()
            Xss.append(Xs_flatten)
            Yss.append(Ys_flatten)
            corr=tools.correlation_nan(Xs_flatten,Ys_flatten)
            corrs[var].append(corr)
            rmse=tools.rmse_nan(Xs_flatten,Ys_flatten)
            rmses[var].append(rmse)
        Xss_flatten=np.concatenate(Xss)
        Yss_flatten=np.concatenate(Yss)
        corr=tools.correlation_nan(Xss_flatten,Yss_flatten)
        corrs_all.append(corr)
        rmse=tools.rmse_nan(Xss_flatten,Yss_flatten)
        rmses_all.append(rmse)
    ###
    ### Start plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(4,1))
    x=range(len(ensemble_numbers))
    for var in model_vars:
        #ax1.plot(x, rmses[var])
        pass
    ax1.plot(x, rmses_all,color='orange',label='RMSE')
    ax2=ax1.twinx()
    ax2.plot(x,corrs_all, color='royalblue',label='Correlation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ensemble_numbers,rotation=45)
    ax1.set_xlabel('Numbers of ensemble member per simulation in RR training')
    ax1.set_ylabel('Root mean\nsqaure errors\n(RMSE)')
    ax2.set_ylabel('Correlation\n(r)')
    ax2.set_yticks([0.6,0.65,0.7])
    ax1.legend(bbox_to_anchor=(0.5,0.3), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=8)
    ax2.legend(bbox_to_anchor=(0.5,0.5), ncol=2, loc='lower left', frameon=False, columnspacing=1, handletextpad=0.5,fontsize=8)
    ###
    for i, ax in enumerate([ax1,ax2]):
        for j in ['top']:
            ax.spines[j].set_visible(False)
            ax.tick_params(axis='x', which='both',length=2)
            ax.tick_params(axis='y', which='both',length=2)
        ax.tick_params(axis='x', direction="out", length=3, colors='black')
        ax.tick_params(axis='y', direction="out", length=3, colors='black')
    ## Save figure
    fig_name = 'figSX_supplementary_ensemble_number'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.001)

