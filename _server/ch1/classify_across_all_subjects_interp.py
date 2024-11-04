import sys, os, argparse
import numpy as np
import pandas as pd

from functools import partial
from joblib import Parallel, delayed
from filelock import FileLock

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    project_path = sys.argv[1] #'/project_data/volume0/tpirnia/data/EEGWS/'
    arg_fname = project_path + "/data/args/" + sys.argv[2] #'args_ch1-phase_c_avg.5reps_multisubject_ridge_w50_s10_1-5.txt'
    subject = sys.argv[3]
    
    subject_list = ("EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(' '))
    
    if subject in subject_list:
        subject_list.remove(subject)
    
    # ========================
    #  IMPORT FUNCTIONS
    # ==========
    sys.path.append(project_path + '/maybe_repo')
    from data_handling.phase_condition import load_args, get_phaseEEG, load_multi_subject
    from data_handling.preprocessing import get_bad_channels
    from data_handling.segment import segment
        
    # ========================
    #  LOAD DATA:
    # ==========
    arg = load_args(arg_fname)
    w, step = arg.window, arg.step
    #bad_ch = get_bad_channels(arg.project_path, subject) #subject= test subject

    # trained on all subjects but j
    #j: left out subject to be tested on
    X_j, y_j, groups_j, arg = get_phaseEEG(arg, 
                                           subject,
                                          )
    #X_train = np.delete(X_train, bad_ch, axis=1)

    # load all train subjects: 
    X_train, y_train, groups_train, ind_subj = load_multi_subject(arg,
                                                               subject_list, #removed delete channels
                                                              )
    arg.t_end, arg.subject = X_train.shape[-1], subject
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)

    # ========================
    #  IMPORT CLASSIFICATION FUNCTION
    # ==========
    if arg.perm:
        from classification.kfold_cv import classify_permute_test as classify
    else:
        from classification.kfold_cv import classify

    # ========================
    #  SEGMENT WINDOWS
    # ======
    X_j, y_j, groups__j = segment(X_j, y_j, groups_j, w, step)
    X_train, y_train, groups_train = segment(X_train, y_train, groups_train, w, step)

    
    # ========================
    # CLASSIFICATION (train/test windows in parallel)
    # ======
    num_cores = 3
    if len(sys.argv)>4:
        select_window = np.int(sys.argv[4]) + 500
        windows = list(np.where(t_ws==select_window)[0])
        print(windows)
    else:
        windows = np.arange(n_windows)

    clf_ = partial(classify, 
                   X_train=X_train,
                   y_train=y_train,
                   groups_train=groups_train,
                   X_test=X_j,
                   y_test=y_j,
                   ind_subj=ind_subj,
                   arg=arg,
                   **vars(arg))
    _ = Parallel(n_jobs=num_cores)(delayed(clf_)(w_j) for w_j in windows) 
