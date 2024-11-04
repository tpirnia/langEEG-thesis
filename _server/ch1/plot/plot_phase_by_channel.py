# need to update this to use the content from ch1_classify_by_channel_plot_average_all_subjects.ipynb
import os, sys
import numpy as np
import itertools, warnings
sys.path.append('/project_data/volume0/tpirnia/data/EEGWS/code')
from stats.fdr import fdr_correction
from data_handling.preprocessing import get_bad_channels

import matplotlib.pyplot as plt
import mne

from mpl_toolkits import axes_grid1
from matplotlib import colors
import matplotlib as mpl


import seaborn as sns

params = {'lines.markersize':6,
          'axes.labelsize': 17,
          'axes.titlesize':20,
          'axes.labelpad':0,
          'xtick.labelsize' :15,
          'ytick.labelsize' :15,
          'xtick.major.pad':7, 
          'ytick.major.pad':6, 
          'legend.title_fontsize':15,
          'figure.figsize':(8,10),
          'figure.subplot.hspace': .03,
          'figure.subplot.wspace': .2,
          'image.origin': 'lower',
          'figure.titlesize': 20,
          'figure.autolayout': False,
          'scatter.marker':'o',
          'savefig.dpi': 450,
          'savefig.bbox': 'tight',
          'savefig.transparent': True,
         }

def plot_acc_brain(arr, info, ch_badmask, subject, params, title='', fname='', show=True):
    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                 linewidth=0, markersize=15)
    with mpl.rc_context(params):
        #plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-poster')
        fig, ax = plt.subplots(figsize=(8,8))
        im, cm = mne.viz.plot_topomap(arr*100, pos=info, axes=ax, show=False, show_names=True, names=info.ch_names,mask=ch_badmask,mask_params=mask_params,vmax=100,vmin=50,res=300,ch_type='EEG',image_interp='none')#,outlines='skirt')
        fig.suptitle(" Subject {}: {}".format(subject,title))

        ax_x_start = 0.99
        ax_x_width = 0.04
        ax_y_start = 0.15
        ax_y_height = 0.75
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('Accuracy',pad=15)
        if fname:
            plt.savefig(fname,
                        dpi=450, 
                        transparent=True,
                        bbox_inches = "tight")
        if show:
            plt.show()
        plt.close()

sfreq = float(1000)
decim=2

montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
info = mne.create_info(montage.ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage(montage)

w,s = 1000,500

project_path='/project_data/volume0/tpirnia/data/EEGWS/'
data_path='{}results_alias/ch1-by_channel/data/w{}'.format(project_path, w)

subject_list = ("EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' '))

for subject in subject_list:
    # Files saved:
    obs_fname ='{}/{}_by_channel_test.npy'.format(data_path, subject)
    fdr_ind_fname = '{}/fdr/{}_by_channel-ind-by.npy'.format(data_path, subject)

    fig_fname = project_path + '/results_alias/ch1-by_channel/figures/w1000/{}_by_channel'.format(subject)

    bad_ch = get_bad_channels(project_path, subject) 
    bad_names = list('E'+ str(bc+1) for bc in bad_ch)

    info['bads'] = bad_names

    chmask = np.isin(info.ch_names,info['bads'],invert=True)
    ch_badmask = np.isin(info.ch_names,info['bads'],invert=False)
    mask_ind = np.argwhere(chmask).squeeze()


    #load obs
    obs_ = np.load(obs_fname)
    obs_m = obs_.mean(axis=1).squeeze()

    #load fdr-corrected indicator
    fdr_ind = np.load(fdr_ind_fname)

    #array of channel numbes
    chs = np.arange(obs_.shape[0])

    #print(obs_.shape,fdr_ind.shape)

    notsig = np.isin(chs, np.argwhere(fdr_ind.squeeze()),invert=True)

    obs_corr = obs_m.copy()
    obs_corr[notsig] = 0

    plot_acc_brain(obs_m, info, ch_badmask,subject,params, title='observed',fname=(fig_fname+'.png'),show=False)
    plot_acc_brain(obs_corr, info, ch_badmask,subject,params,title='fdr-corrected',fname=(fig_fname+'-fdr_corr.png'),show=False)