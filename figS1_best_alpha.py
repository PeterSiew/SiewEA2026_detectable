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


alphas=[1,2] # For testing
alphas=[20000]
alphas=[10,100,1000,2000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000]
alphas=[1,3,5,8,10,100,1000,2000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000] # As requested by reviewer

if False: ### Save the Y_predict and Y_true files
    ### Need to change the True/False statements in fig1_script: Take-one-simulation-out for training and validation 
    ###
    jobs=[]
    for alpha in alphas:
        ridge_alpha=alpha
        fig1_obs=False
        fig2A_fingerprint=False
        save_Y_test_trues_predicts=True
        exec(open("./fig1_fig2A_obs_ts_model_fingerprint.py").read())


if True: ### Getting the 30-year trends
    ### Read the save files, calulcate the RMSE
    period=30
    en_num=16
    ensembles={'cesm1_tas_daily_en':range(1,en_num+1),'canesm2_tas_daily_en':range(1,en_num+1),'gfdlesm2m_tas_daily_en':range(1,en_num+1),
              'gfdlcm3_tas_daily_en':range(1,en_num+1),'mk360_tas_daily_en':range(1,en_num+1),'ecearth_tas_daily_en':range(1,en_num+1)}; ensemble_no=en_num
    model_years={'cesm1_tas_daily_en':range(1950,2100),'canesm2_tas_daily_en':range(1950,2100),'gfdlesm2m_tas_daily_en':range(1950,2100),
                 'gfdlcm3_tas_daily_en':range(1950,2100),'mk360_tas_daily_en':range(1950,2100),'ecearth_tas_daily_en':range(1950,2100)}
    Y_true_slopes_FINAL={}
    Y_predict_slopes_FINAL={}
    Y_org_slopes_FINAL={}
    rmses_raw={}
    for alpha in alphas:
        print(alpha)
        ## Load the data
        Y_test_trues=np.load('./alphas/Y_test_trues_alpha%s_ens%s.npy'%(alpha,en_num),allow_pickle=True).item()
        Y_test_predicts=np.load('./alphas/Y_test_predicts_alpha%s_ens%s.npy'%(alpha,en_num), allow_pickle=True).item()
        Y_lens_raw=np.load('./alphas/Y_lens_raw_alpha%s_ens%s.npy'%(alpha,en_num))
        train_records=np.load('./alphas/train_records_alpha%s_ens%s.npy'%(alpha,en_num))
        ##
        model_vars=list(Y_test_trues.keys())
        Y_test_trues=np.array([Y_test_trues[var] for var in model_vars]).reshape(-1)
        Y_test_predicts=np.array([Y_test_predicts[var] for var in model_vars]).reshape(-1)
        #ipdb.set_trace()
        ### Get RMSE for the raw_timeseries (not the 30-year trnneds)
        rmse=tools.rmse_nan(Y_test_trues,Y_test_predicts)
        rmses_raw[alpha]=rmse
        #rmse=tools.correlation_nan(Y_trues,Y_predicts)
        #idx=np.argsort(rmses); print(np.array(alphas)[idx]) # Print the RMSE wit hthe smallest alphas
        ### Calculate 30-year trends for each var and each ensemble
        Y_org_slopes={var:{} for var in model_vars}
        Y_true_slopes={var:{} for var in model_vars}
        Y_predict_slopes={var:{} for var in model_vars}
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
        Y_org_slopes_FINAL[alpha]=Y_org_slopes
        Y_true_slopes_FINAL[alpha]=Y_true_slopes
        Y_predict_slopes_FINAL[alpha]=Y_predict_slopes
    ### Get RMSES for 30-year trends
    rmses_trends={}
    for alpha in alphas:
        Xs,Ys=[],[]
        for i, var in enumerate(model_vars):
            for en in ensembles[var]:
                X=Y_true_slopes_FINAL[alpha][var][str(en)]
                Xs.append(X)
                Y=Y_predict_slopes_FINAL[alpha][var][str(en)]
                Ys.append(Y)
        Xs_flatten=np.array(Xs).flatten()
        Ys_flatten=np.array(Ys).flatten()
        rmse=tools.rmse_nan(Xs_flatten,Ys_flatten)
        #rmse=tools.correlation_nan(Xs_flatten,Ys_flatten)
        rmses_trends[alpha]=rmse

    ### Start plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(5,1))
    bar_width=0.4
    x=np.arange(len(alphas))
    if True: # raw RMSE changes
        ax1.bar(x, [rmses_raw[alpha] for alpha in alphas], bar_width, color='k')
        yticks=[0,0.1,0.2]
    else: # rmse for 30-year trends
        ax1.bar(x, [rmses_trends[alpha] for alpha in alphas], bar_width, color='k')
        yticks=[0,0.025,0.05]
    ax1.set_yticks(yticks)
    #ax1.bar(x-0.4, rmse_test, bar_width, color='r')
    ax1.set_xticks(x)
    ax1.set_xticklabels(alphas,rotation=45)
    ax1.set_xlabel(r'Lambda ($lambda$)')
    ax1.set_ylabel('RMSE (K)')
    ## Save figure
    fig_name = 'figS1_supplementary'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.001)


