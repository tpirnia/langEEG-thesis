import sys, os, argparse
import numpy as np
import pandas as pd
import copy

project_path = '/project_data/volume0/tpirnia/data/EEGWS/'

sys.path.append(project_path + 'maybe_repo')
from data_handling.preprocessing import get_bad_channels

subject_list = ("EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' '))

out_path = '{}data/across_subjects/results/ch1-by_channel/data/'.format(project_path)

t_start = 1
w,s = 1000, 500

for subject in subject_list:
    bad_ch = get_bad_channels(project_path, subject) 
    df_fname = project_path + 'results_alias/ch1-by_channel/by_subject/results_c_avg_w{}_s{}_block5-5_ridge2_{}.csv'.format(w,s, subject)

    df_ = pd.read_csv(df_fname)

    df_.drop_duplicates(inplace=True)
    df_.reset_index(inplace=True,drop=True)
    df_filt = df_[(df_['w_train']>=t_start)]
    df_filt.drop_duplicates(subset=['fold','channel','w_train'],inplace=True, ignore_index=True)

    df_filt.drop(columns=['window','step','classifier','w_train','t_0'],inplace=True)

    df_filt.head()

    df_filt.sort_values(by=['channel','fold'],inplace=True, ignore_index=True)

    folds = df_filt.fold.unique()
    n_folds = len(folds)

    n_channels = 128 + 1
    chs = np.arange(n_channels)
    # chs_inds = np.isin(chs, bad_ch,invert=True) 
    chs_inds = chs[np.isin(chs, bad_ch,invert=True)]


    arr_out = np.empty(shape=(n_channels,n_folds, 1))
    arr_tr_out = np.empty(shape=(n_channels,n_folds, 1))
    #arr_out[:],arr_tr_out[:]=np.nan, np.nan

    for ch_i, ch in enumerate(chs_inds):
        arr_out[ch,:,:] = df_filt.loc[(df_filt['channel']==ch_i),'t_500':].values
        arr_tr_out[ch,:,:] = (df_filt.loc[(df_filt['channel']==ch_i),'train_accuracy'].values)[:,np.newaxis]

    out_fname = out_path + 'w{}/{}_by_channel_test.npy'.format(w, subject)
    out_tr_fname = out_path + 'w{}/{}_by_channel_train.npy'.format(w, subject)

    np.save(out_fname, arr_out)
    np.save(out_tr_fname, arr_tr_out)

    del(df_filt, arr_out, arr_tr_out)