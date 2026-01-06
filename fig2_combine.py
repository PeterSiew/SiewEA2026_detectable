import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from PIL import Image
import datetime as dt
import subprocess
from multiprocessing import Process
import ipdb
import scipy

### Combine Figure 2A (model fingerprint) and Figure 2B (the obs trends)
today_date=dt.date.today()
filenames=["/Users/home/siewpe/codes/graphs/%s_fig2A_model_fingerprints.png"%today_date,"/Users/home/siewpe/codes/graphs/%s_fig2B_tempsd_or_AA_trend.png"%today_date]

plt.close()
fig, axs = plt.subplots(len(filenames),1, figsize=(7,6))
axs=axs.flatten()
ABCD=[r'$\bf_{(A)}$',r'$\bf_{(B)}$']
for i in range(len(filenames)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
        #axs[i].set_zorder(zorders[i])
        axs[i].annotate(ABCD[i], xy=(-0.038, 1.05), xycoords='axes fraction',fontsize=12, 
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='white',alpha=1, pad=0.001), zorder=100)
fig_name = 'fig2_combine_maps'; hspace=-0.853
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=hspace)
plt.savefig('/Users/home/siewpe/codes/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)
