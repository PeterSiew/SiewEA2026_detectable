import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from PIL import Image
import datetime as dt
import subprocess
from multiprocessing import Process
import ipdb
import scipy

def figure5A():
    fig_add='5A'
    print("Starting Figure 5A")
    #vars=['cesm1','canesm2','mk360','ecearth']; repeat_no=2 # for testing
    vars=['cesm1','canesm2','mk360','ecearth']; repeat_no=16 # full member (EC-Earth is quite slow)
    ######
    # GFDL-CM3 also has PSL data - why not using?
    ####
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
    years=[i for i in range(1979,2024)] # Current decades (For Figure 5)
    early_years=[i for i in range(1979,2001)]
    late_years=[i for i in range(2001,2024)]
    if False: # Future decades (For Figure S7, S8)
        years=[i for i in range(2030,2070)] 
        early_years=[i for i in range(2030,2050)]
        late_years=[i for i in range(2051,2070)]
    shading_level_grid=np.linspace(-0.6,0.6,13)
    exec(open("./fig56_tas_slp_cold_hot_extremes_and_circulation_indices.py").read())

def figure5B():
    fig_add='5B'
    print("Starting Figure 5B")
    ensembles=range(1,3) # testing
    ensembles=range(1,31) # WACCAM
    psl_path, tas_path = {}, {}
    for en in ensembles:
        psl_path[en]='/mnt/data/data_a/liang_greenice/waccm6/slp_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
        tas_path[en]='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp1_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
    psl_var='PSL'
    tas_var='TREFHT'
    years=[i for i in range(1979,2014)] # For WACCAM
    early_years=[i for i in range(1979,1996)]
    late_years=[i for i in range(1996,2014)]
    shading_level_grid=np.linspace(-0.6,0.6,13)
    exec(open("./fig56_tas_slp_cold_hot_extremes_and_circulation_indices.py").read())

def figure5C():
    fig_add='5C'
    print("Starting Figure 5C")
    ensembles=range(1,3) # testing
    ensembles=range(1,31) # WACCAM
    psl_path, tas_path = {}, {}
    for en in ensembles:
        psl_path[en]='/mnt/data/data_a/liang_greenice/waccm6/slp_daily/exp2_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
        tas_path[en]='/mnt/data/data_a/liang_greenice/waccm6/tas_daily/exp2_ens%s_waccm_whoi_ncar.cam.h1.*-01-01-00000.nc'%str(en).zfill(2)
    psl_var='PSL'
    tas_var='TREFHT'
    years=[i for i in range(1979,2014)] # For WACCAM
    early_years=[i for i in range(1979,1996)]
    late_years=[i for i in range(1996,2014)]
    shading_level_grid=np.linspace(-0.6,0.6,13)
    exec(open("./fig56_tas_slp_cold_hot_extremes_and_circulation_indices.py").read())

def figure5D():
    fig_add='5D'
    print("Starting Figure 5D")
    ensembles=['']
    psl_path, tas_path = {}, {}
    for en in ensembles:
        tas_path[en]='/mnt/data/data_a/ERA5/T2M_daily/T2M_daily-1940Jan_2024Mar_1x1.nc'
        psl_path[en]='/mnt/data/data_a/ERA5/MSLP_daily/MSLP_daily-1940Jan_2024Mar_1x1.nc'
    psl_var='msl'
    tas_var='t2m'
    years=[i for i in range(1979,2024)]
    early_years=[i for i in range(1979,2001)]
    late_years=[i for i in range(2001,2024)]
    shading_level_grid=np.linspace(-0.9,0.9,13)
    shading_level_grid=np.linspace(-0.6,0.6,13) # As requested by the reviewer
    exec(open("./fig56_tas_slp_cold_hot_extremes_and_circulation_indices.py").read())

## To wait for all results to be here 
p1 = Process(target=figure5A); p1.start()
p2 = Process(target=figure5B); p2.start()
p3 = Process(target=figure5C); p3.start()
p4 = Process(target=figure5D); p4.start()

p1.join(); print("P1 finish")
p2.join(); print("P2 finish")
p3.join(); print("P3 finish")
p4.join(); print("P4 finish")

if False: # old order
    fig_adds=['5A','5B','5C','5D']
    ylabels=[r'$\bf{(A)}$'+'\nLarge-\nensemble\nsimulations',r'$\bf{(B)}$'+'\nWACCM6\nobserved\nsea ice',r'$\bf{(C)}$'+'\nWACCM6\nfixed\nsea ice',r'$\bf{(D)}$'+'\nERA5']
else:
    fig_adds=['5B','5C','5A','5D']
    ylabels=[r'$\bf{(A)}$'+'\nWACCM6\nobserved\nsea ice',r'$\bf{(B)}$'+'\nWACCM6\nclimatological\nsea ice',r'$\bf{(C)}$'+'\nLarge-\nensemble\nsimulations', r'$\bf{(D)}$'+'\nERA5']
### Figure 5 (the maps)
plt.close()
fig, axs = plt.subplots(len(fig_adds),1, figsize=(15,3*len(fig_adds)))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig5_maps_%s.png"%(today_date,fig_add) for fig_add in fig_adds]; fig_name = 'fig5_combine_maps'; hspace=-0.65
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        #axs[i].set_zorder(zorders[i])
        axs[i].annotate(ylabels[i], xy=(-0.13, 0.95), xycoords='axes fraction',fontsize=15, 
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=hspace)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

### Figure 6 (the timeseries)
plt.close()
fig, axs = plt.subplots(len(fig_adds),1, figsize=(15,3*len(fig_adds)))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig6_timeseries_%s.png"%(today_date,fig_add) for fig_add in fig_adds]; fig_name = 'fig6_combine_timeseries'; hspace=-0.5
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        #axs[i].set_zorder(zorders[i])
        axs[i].annotate(ylabels[i], xy=(-0.13, 0.95), xycoords='axes fraction',fontsize=15, 
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=hspace)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

### Figure 7 (the histogram) - actually this is a figure for the response letter or supplementary figure
plt.close()
fig, axs = plt.subplots(len(fig_adds),1, figsize=(15,3*len(fig_adds)))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig7_tas_histogram_early_late_%s.png"%(today_date,fig_add) for fig_add in fig_adds]; fig_name = 'fig7_combine_histogram'; hspace=0.1
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        #axs[i].set_zorder(zorders[i])
        axs[i].annotate(ylabels[i], xy=(-0.19, 0.95), xycoords='axes fraction',fontsize=15, 
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=hspace)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)

