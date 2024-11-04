from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock # added 04/25/22

from sklearn.model_selection import GroupKFold
from functools import partial
from joblib import Parallel, delayed

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def sts_to_csv(sts, arg, w_j):
    test_df = pd.concat(sts,ignore_index=True)
    test_df['train_accuracy'] = test_df['train_accuracy'].astype(np.float)
    test_df.iloc[:,8:] = test_df.iloc[:,8:].astype(np.float)
    test_df.iloc[:,2:6] = test_df.iloc[:,2:6].astype(np.int)
    test_df.iloc[:,7] = test_df.iloc[:,7].astype(np.int)


    fname = (arg.project_path 
             + 'data/' 
             + arg.subject 
             + '/results/w'+ np.str(arg.window)+'_s'+np.str(arg.step)+'/' 
             + arg.out_name 
             + '_'
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    test_df.to_csv(fname)
    by_cols = list(test_df.columns.values[:5])
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    acc_agg = acc_agg.drop(columns=["fold", "test_trial_num"])
    tmp_col_names = ','.join(list(acc_agg.columns))


    acc_agg = list(acc_agg.values.squeeze())
    acc_agg = ','.join(map(np.str, acc_agg))
    group_fname = arg.project_path + 'data/across_subjects/results/' + arg.out_name + '.csv'
    with FileLock(group_fname+".lock"): # added 04/25/22
        if not os.path.exists(group_fname):
            with open(group_fname, "a+") as f:  _ = f.write(tmp_col_names+'\n')
        with open(group_fname, "a+") as f:  _ = f.write(acc_agg+'\n')

        
if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    project_path = sys.argv[1]
    arg_fname = project_path + "/data/args/" + sys.argv[2]
    subject = sys.argv[3]
    select_window = np.int(sys.argv[4]) + 500
    # ========================
    #  IMPORT FUNCTIONS 
    sys.path.append(project_path + '/maybe_repo')
    from data_handling.phase_condition import load_args, get_phaseEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.nested_kfold_cv import inner_CV_permute as classify
    # ========================
    #  LOAD DATA:
    # ==========
    arg = load_args(arg_fname, subject)
    w, step = arg.window, arg.step
    X, y, groups, arg = get_phaseEEG(arg, subject)
    n_trials, n_channels, n_timepoints = X.shape
    t_ws = np.arange(0, n_timepoints - w, step)
    windows = list(np.where(t_ws==select_window)[0]) 
    arg.t_end = n_timepoints

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    cv_outer = GroupKFold(n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups)) 

    # ========================
    #  SEGMENT WINDOWS
    # ======
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)

    # ========================
    # NESTED - CLASSIFICATION
    # ======
    num_cores = 25
    if arg.perm:
        for w_j in windows:
            rclf_ = partial(classify,
                            w_j=w_j,
                            X=X,
                            y=y,
                            groups=groups,
                            trte=trte,
                            arg=arg,
                            **vars(arg))
            _ = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
            #sts_to_csv(sts, arg, w_j)
    else:
        print("No classification run: perm=0 in argument file. ")