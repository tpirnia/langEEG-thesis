from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock # added 04/25/22

from sklearn.model_selection import GroupKFold
from functools import partial
from joblib import Parallel, delayed
import itertools

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def sts_to_csv(sts, arg, w_j,ch, header=True):
    fname = '{}data/{}/results/w{}_s{}/by_channel/{}_ch{}_{}.csv'.format(arg.project_path,
                                                                     arg.subject,
                                                                     arg.window,arg.step, 
                                                                     arg.out_name, str(ch).zfill(3),
                                                                     str(w_j).zfill(3))
    group_fname = '{}data/across_subjects/results/ch1-by_channel/by_subject/{}_{}.csv'.format(arg.project_path,
                                                                                          arg.out_name,
                                                                                          arg.subject)
    test_df = pd.concat(sts,ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')

    if arg.perm:
        test_df['permutation'] = arg.ith_perm
        reordered_col=(list(test_df.loc[:,:'w_train'].columns.values)[:-1] \
                       + ['permutation']  \
                       + list(test_df.loc[:,'w_train':'permutation'].columns.values))[:-1]
        test_df = test_df.loc[:, reordered_col]
        fname = fname.split('.')[0] + '-perm.csv'
        group_fname = group_fname.split('.')[0]  + '-perm.csv'
    
    fill_path(fname)
    if os.path.exists(fname): 
        header=False

    test_df.to_csv(fname, mode='a', index=False, header=header)
    

    by_cols = list(test_df.loc[:,:'fold'].columns.values)#.columns.values[:5])

    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    acc_agg = acc_agg.drop(columns=["test_trial_num"])
    tmp_col_names = ','.join(list(acc_agg.columns))

    acc_agg = list(acc_agg.values.squeeze())
    #acc_agg = ','.join(map(np.str, acc_agg))

    fill_path(group_fname)
    with FileLock(group_fname+".lock"): # added 04/25/22
        if not os.path.exists(group_fname):
            with open(group_fname, "a+") as f:  _ = f.write(tmp_col_names+'\n')
        with open(group_fname, "a+") as f:  
            #     _ = f.write(acc_agg+'\n')
            for acc_ in acc_agg:
                acc_ = ','.join(map(np.str, acc_))
                _ = f.write(acc_+'\n')

if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    project_path = sys.argv[1]
    arg_fname = project_path + "/data/args/" + sys.argv[2]
    subject = sys.argv[3]
    
    first_window = 500
    # ========================
    #  IMPORT FUNCTIONS 
    sys.path.append(project_path + '/maybe_repo')
    from data_handling.phase_condition import load_args, get_phaseEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.nested_window_bychannel import inner_CV_ch
    # ========================
    #  LOAD DATA:
    # ==========
    arg = load_args(arg_fname, subject)
    w, step = arg.window, arg.step
    X, y, groups, arg = get_phaseEEG(arg, subject)
    if np.logical_and(np.logical_or((arg.classifier == 'ridge'),(arg.classifier == 'svm')), (np.array_equal(np.unique(y),([0,1])))):
        y[y==0]= -1
    elif np.logical_and((arg.classifier == 'log'),(np.array_equal(np.unique(y),([-1, 0,1])))):
        y[y==-1]= 2
    n_trials, n_channels, n_timepoints = X.shape
    t_ws = np.arange(0, n_timepoints - w, step)
    n_windows = len(t_ws)
    n_instances = n_trials * n_windows
    arg.t_end = n_timepoints

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    cv_outer = GroupKFold(n_splits=arg.outer_splits)
    #trte = list(cv_outer.split(X, y, groups))

    # ========================
    #  SEGMENT WINDOWS
    # ======
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)

    # ========================
    # NESTED - CLASSIFICATION
    # ======
    num_cores = arg.outer_splits
    CHS = np.arange(n_channels)
    try:
        select_window = np.int(sys.argv[4]) + 500
        WINS = list(np.where(t_ws==select_window)[0])
    except:
        WINS = np.arange(np.where(t_ws==first_window)[0], n_windows)
    
    rngen = np.random.RandomState()
    for perm in range(arg.n_perms):
        arg.ith_perm = perm # 2. set permutation number
        ind_perm = _shuffle(y, None, rngen) # 1. shuffle labels
        trte = list(cv_outer.split(X, y[ind_perm,0], groups[:,0]))
        for w_j,ch in itertools.product(WINS, CHS):
            rclf_ = partial(inner_CV_ch,
                            w_j=w_j,
                            X=X[:,:,ch,:],
                            y=y[ind_perm],
                            groups=groups,
                            trte=trte,
                            ch=ch,
                            arg=arg,
                            **vars(arg))
            sts = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
            sts_to_csv(sts,arg,w_j,ch)

        # last_perm = _get_last_perm(subject, project_path)
        # rngen = np.random.RandomState()
        # for perm in range(arg.n_perms):
        #     arg.ith_perm = perm
        #     ind_perm_j = _shuffle(y[:,0], groups[:,0], rngen)
        #     #ind_perm_train = _shuffle(y_train[:,0], groups_train[:,0], rngen)
        #     ind_perm_train = _shuffle(y_train, groups_train, rngen)
        #     if arg.ith_perm > last_perm:
        #         clf_ = partial(classify, 
        #                        X_train=X_train,
        #                        y_train=y_train[ind_perm_train],
        #                        groups_train=groups_train[ind_perm_train],
        #                        X_test=X_j,
        #                        y_test=y_j[ind_perm_j],
        #                        ind_subj=ind_j[ind_perm_j],
        #                        arg=arg,
        #                        **vars(arg))
        #         _ = Parallel(n_jobs=num_cores)(delayed(clf_)(w_j) for w_j in windows[50:]) #change to 0 for full -500 to 1000ms