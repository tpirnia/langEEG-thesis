import sys, os, argparse
import numpy as np
import pandas as pd
import copy

project_path = '/project_data/volume0/tpirnia/thesis_repository'

sys.path.append(project_path + 'maybe_repo')
from data_handling.preprocessing import get_bad_channels

w, s = 1000, 500

subject_list = ("EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' '))

out_path = os.path.join(project_path, 'results/phase_accuracy_by_channel_w1000')
# out_path = '{}data/across_subjects/results/ch1-by_channel/data/'.format(project_path)

t_start = 1


for subject in subject_list:

    bad_ch = get_bad_channels(project_path, subject) 

    df_fname = os.path.join(project_path, 
                            f'results/permutations/phase_accuracy_by_channel_w{}/{}_ridge_accuracy.csv'.format(w, subject)
                           )
                            
    df_ = pd.read_csv(df_fname)
    df_.drop_duplicates(inplace=True)
    df_.reset_index(inplace=True,drop=True)
    df_.rename(columns={'perm':'permutation'},inplace=True)
    df_filt = df_[(df_['w_train']>=t_start)]
    df_filt.drop_duplicates(subset=['permutation',
                                    'fold','channel','w_train'],
                            inplace=True, 
                            ignore_index=True)

    df_filt.drop(columns=['window',
                          'step',
                          'classifier',
                          'w_train',
                          't_0'],inplace=True)

    df_filt.head()

    df_filt.sort_values(by=['permutation','channel','fold'],inplace=True, ignore_index=True)
    del(df_)
    folds = df_filt.fold.unique()
    n_folds = len(folds)

    n_channels = 128 + 1
    chs = np.arange(n_channels)
    # chs_inds = np.isin(chs, bad_ch,invert=True) 
    chs_inds = chs[np.isin(chs, bad_ch,invert=True)]

    perms = df_filt['permutation'].unique()
    n_perms = len(perms)

    # df_filt.head()

    ch_perms = {}
    for ch_i, ch in enumerate(chs_inds):
        ch_perms[ch_i] = df_filt[df_filt['channel']==ch_i]['permutation'].unique()
        #print(ch_i, len(ch_perms[ch_i]))

    arr_out = np.empty(shape=(n_perms, n_channels,n_folds, 1))
    arr_tr_out = np.empty(shape=(n_perms, n_channels,n_folds, 1))

    for p_i, p in enumerate(np.arange(n_perms)): 
        for f in folds:
            arr_out[p_i, chs_inds,f,:] = df_filt.loc[((df_filt['permutation']==p)&(df_filt['fold']==f)),'t_500':].values
            arr_tr_out[p_i, chs_inds,f,:] = (df_filt.loc[((df_filt['permutation']==p)&(df_filt['fold']==f)),'train_accuracy'].values)[:,np.newaxis]

    out_fname = out_path + 'w{}/null/{}_by_channel_test.npy'.format(w, subject)
    out_tr_fname = out_path + 'w{}/null/{}_by_channel_train.npy'.format(w, subject)

    np.save(out_fname, arr_out)
    np.save(out_tr_fname, arr_tr_out)
    del(arr_out, arr_tr_out)