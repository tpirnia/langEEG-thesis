import sys, os
import numpy as np
import matplotlib.pyplot as plt
import mne
project_path = "/project_data/volume0/tpirnia/thesis_repository/"
subject_list = ("EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' '))

w,s=1000,500

sys.path.append(project_path + 'repo')
from data_handling.preprocessing import get_bad_channels


def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)


out_path =  os.path.join(project_path,'results', f'phase_accuracy_by_channel_w{w}')
# out_path = '{}data/across_subjects/results/ch1-by_channel/data/'.format(project_path)

for subject in subject_list:
    sfreq = float(1000)
    decim=2
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    info = mne.create_info(montage.ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    
    bad_ch = get_bad_channels(project_path, subject) 
    bad_names = list('E'+ str(bc) for bc in bad_ch)
    info['bads'] = bad_names
    
    arr_ = np.load(out_path + '/obs/{}_by_channel_test.npy'.format(subject))
    arr_m = np.nanmean(arr_,axis=1).squeeze()
    arr_m[np.where(arr_m[:]<.55)]=0

    chmask = np.isin(info.ch_names,info['bads'],invert=True)
    
    fig, ax = plt.subplots(figsize=(8,8))
    im, cm = mne.viz.plot_topomap(arr_m, pos=info, axes=ax, show=False, show_names=False, names=info.ch_names, mask=chmask, vmax=1, vmin=.55, contours=6, res=100)
    ax_x_start = 0.99
    ax_x_width = 0.04
    ax_y_start = 0.0
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('accuracy',fontsize=10) # title on top of colorbar
    ax.set_title(label=" Subject {}: Full trial classification".format(subject),size=15)
    mne.viz.tight_layout()
    fill_path(out_path+'/figures/w{}/'.format(w))
    plt.savefig(fname=out_path+'/figures/w{}/{}_by_channel_test.png'.format(w, subject))
    plt.close()