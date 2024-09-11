DATA_SUFFS = ['','data_avg.3reps_context']
SUFFS_X = ['','_phase']

EEG_ch = 129

mode='accurate'
reset_bads=True
method=dict(eeg='MNE')

project_path = '/project_data/volume0/tpirnia/data/EEGWS/'


SUBJECTS = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' ')

import sys, os
import numpy as np
import pandas as pd
import mne
import copy
import itertools

sys.path.append(project_path + 'maybe_repo')
from data_handling.preprocessing import get_bad_channels

def make_mne_info(sfreq=float(1000)):
    decim=2
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    info = mne.create_info(montage.ch_names, 
                           sfreq=sfreq,
                           ch_types='eeg',verbose=False)
    info.set_montage(montage,verbose=False)
    return info


for subject in SUBJECTS:
    for data_suffix, suffix_X in zip(DATA_SUFFS, SUFFS_X):
        if data_suffix:
            out_path = project_path + 'data/{}/{}/'.format(subject,
                                                           data_suffix)
        else:
            out_path = project_path + 'data/{}/'.format(subject)
            
        X = np.load(out_path +  'X' + suffix_X + '.npy')
        bads_ind = get_bad_channels(project_path, subject)
        bads_name = list(map('E{}'.format,bads_ind + 1))

        if not bads_ind.size > 0:
            np.save(out_path + 'X' + suffix_X +'_interp.npy',
                    X)
        else:
            original_ind = np.delete(np.arange(EEG_ch), bads_ind)

            # make empty eeg array
            n_trials, _, n_times = X.shape
            raw = np.zeros((n_trials, EEG_ch, n_times))
            raw[:,original_ind] = copy.deepcopy(X)


            info = make_mne_info()
            _epochs = mne.EpochsArray(raw, info,verbose=False)
            _epochs.info['bads'] = bads_name  # mark bad channels

            #interpolated via MNE
            evoked_interp_mne = _epochs.copy().interpolate_bads(mode=mode,
                reset_bads=reset_bads, method=method, verbose=False)

            np.save(out_path + 'X' + suffix_X + '_interp.npy',
                    evoked_interp_mne.get_data())
            del(raw, _epochs, evoked_interp_mne)
        del(X)
    print(subject)