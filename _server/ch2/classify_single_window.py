from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock # added 04/25/22

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed

def update_labels(classifier, y):
    if np.logical_and(np.logical_or((classifier == 'ridge'),
                                    (classifier == 'svm')), 
                      (np.array_equal(np.unique(y),([0,1])))):
        y[y==0]= -1
    elif np.logical_and(
        (classifier == 'log'),
        (np.array_equal(np.unique(y),([-1, 0,1])))):
        y[y==-1]= 2
    return y
  
CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}

if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    project_path = sys.argv[1]
    arg_fname = project_path + "/data/args/" + sys.argv[2]
    subject = sys.argv[3]
    # ========================
    #  IMPORT FUNCTIONS 
    sys.path.append(project_path + '/maybe_repo')
    from data_handling.context_condition import load_args, get_contextEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.nested_window_ch2 import inner_CV_by_rep as inner_CV
    from utils.utils import sts_to_csv 
    # ========================
    #  LOAD DATA:
    # ==========
    arg = load_args(arg_fname, subject)
    
    if len(sys.argv)>5:
        data_suffix = sys.argv[5] #***
    elif "data_suffix" in arg:
        data_suffix = arg.data_suffix
    else: 
        data_suffix=None
    
    w, step = arg.window, arg.step
    
    X, y, groups, arg = get_contextEEG(arg, subject, suffix=data_suffix)
    y = update_labels(arg.classifier, y)
    
    n_trials, _, arg.t_end = X.shape
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    cv_outer = CVs[arg.splitter](n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))

    # ========================
    #  SEGMENT WINDOWS
    # ======
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)

    # ========================
    # NESTED - CLASSIFICATION
    # ======
    num_cores = 3
    select_window = np.int(sys.argv[4]) + 500
    windows = list(np.where(t_ws==select_window)[0])
    for w_j in windows:
        rclf_ = partial(inner_CV,
                        w_j=w_j,
                        X=X,
                        y=y,
                        groups=groups,
                        trte=trte,
                        arg=arg,
                        **vars(arg))
        sts = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
        sts_to_csv(sts, arg, w_j)  #***

        