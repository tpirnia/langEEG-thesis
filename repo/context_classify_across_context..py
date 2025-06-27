# classify_single_window_across_conditions.py
# python ython repo/context_classify_across_context..py -c svm -s EP03 -w 1000 --test semantic --single_window
from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock # added 04/25/22

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed

PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository/"

cc_dict = {
    'phonological':{
        'label': 'US-UP',
        'train':['US',2], # label for phonological in y is 2
        'test': ['UP',1] # label for phonological in y is 1
    },
    'semantic':{
        'label': 'UP-US',
        'train':['UP',1], # label for phonological in y is 2
        'test': ['US',2] # label for phonological in y is 1
    }
}

context_dict = {
    'US':'semantic',
    'UP':'phonological',
    'phonological': 'UP',
    'semantic':'US'
}

CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}

def parse_arguments():
    """Parse command line arguments.
    --test: "phonological" or "semantic"
    args.test and cross_condition refer to the same thing.
    """
    description = ('File Locations:\n'
              '  Input config: {}/repo/args/context_XX-XX_w<window>.txt\n'
              '  Output results: {}/results/context_accuracy_XX-XX_w<window>.csv\n\n'.format(PROJECT_PATH, PROJECT_PATH))

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description=description)
    parser.add_argument('-c', '--classifier', default='svm', 
                   help='Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)')
    parser.add_argument('-s', '--subject', required=True, 
                       help='Subject ID (e.g., EP06)')
    parser.add_argument('-w', '--window', type=int, default=1000, 
                       help='Window size in ms (default=1000)')
    parser.add_argument('-ts', '--t_start', type=int, default=0, help='Time (ms) to start classification (default=0)')
    parser.add_argument('--single_window', action='store_true', default=False,
                       help='Run classification on only the first window at stimulus onset (500ms)')
    parser.add_argument('--train', required=False, 
                       help="Context condition to train on Options: 'semantic' or 'phonological'.")
    parser.add_argument('--test', required=True, 
                       help="Context condition to test on. 'semantic' or 'phonological'. This will assume the one not set to test should be the training set.")
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations (default=1000)')
    parser.add_argument('--run_perms', action='store_true', default=False,
                        help='Run permutation testing instead')    
    return parser.parse_args()

def assign_args(arg_fname, args,load_args):
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH +'/'
    arg.fstem = arg.out_name
    for key, value in vars(args).items():
        setattr(arg, key, value)
    return arg
    
def sts_to_csv(sts, arg, w_j, suffix=""):
    test_df = pd.concat(sts,ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    out_name = arg.out_name  + suffix
    by_cols = ['subject', 'classifier', 'within_condition','cross_condition', 'window', 'step', 'w_train', 'test_trial_val']
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean(numeric_only=True)
    acc_agg = acc_agg.drop(columns=["fold", "test_trial_num"])
    tmp_col_names = ','.join(list(acc_agg.columns))
    # acc_agg = list(acc_agg.values.squeeze())  # type: ignore
    # acc_agg = ','.join(map(str, acc_agg))
    group_fname = arg.project_path + '/results/' + out_name + '.csv'
    with FileLock(group_fname+".lock"): 
        if not os.path.exists(group_fname):
            with open(group_fname, "a+") as f:  _ = f.write(tmp_col_names+'\n')
        with open(group_fname, "a+") as f:  
            for acc_ in acc_agg.values.squeeze():
                acc_ = ','.join(map(str, acc_))
                _ = f.write(acc_+'\n')
    return group_fname

        
def setup_paths(window_size, test_condition):
    """Set up project paths and filenames."""
    arg_fname = '{}/repo/args/context_{}_w{}.txt'.format(PROJECT_PATH, cc_dict[test_condition]['label'], window_size)
    results_fname = '{}/results/context_accuracy_{}_w{}'.format(PROJECT_PATH, cc_dict[test_condition]['label'], window_size)
    return arg_fname, results_fname

def load_dependencies():
    """Import necessary project-specific functions."""
    sys.path.append(PROJECT_PATH + 'repo')
    from data_handling.context_condition import load_args, get_contextEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.nested_window_across_condition import inner_CV_by_rep as inner_CV
    from classification.cross_condition import conflictConditions
    return load_args, get_contextEEG, segment_ts, order_by_trials, inner_CV, conflictConditions
        
if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    # arg_fname = project_path + "repo/args/" + sys.argv[1]
    # subject = sys.argv[2]
    # cross_condition = str(sys.argv[3]) 
    
    args = parse_arguments()
    print(args.__dict__)
    arg_fname, results_fname = setup_paths(args.window, args.test)
    
    load_args, get_contextEEG, segment_ts, order_by_trials, inner_CV, conflictConditions = load_dependencies()
    # ========================
    #  IMPORT FUNCTIONS 
    # sys.path.append(project_path + 'repo')
    # from data_handling.context_condition import load_args, get_contextEEG
    # from data_handling.segment import segment_ts, order_by_trials
    # from classification.nested_window_across_condition import inner_CV_by_rep as inner_CV
    # from classification.cross_condition import conflictConditions
    # from utils.utils import sts_to_csv 
    # ========================
    #  LOAD DATA:
    # ==========
    arg = assign_args(arg_fname, args,load_args)
    arg.project_path = PROJECT_PATH
    cross_condition=arg.test
    if (cross_condition=='phonological'):
        cross_condition = 1
        arg.out_name = arg.out_name + "_US-P"
    elif (cross_condition=='semantic'): 
        cross_condition = 2
        arg.out_name = arg.out_name + "_UP-S"
    
    if "data_suffix" in arg:
        data_suffix = arg.data_suffix
    else: 
        data_suffix=None
    
    w, step = arg.window, arg.step
    
    X, y, groups, arg = get_contextEEG(arg, arg.subject, suffix=data_suffix)
    
    n_trials, _, arg.t_end = X.shape
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    cv_outer = CVs[arg.splitter](n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))
    
    conditions = conflictConditions(cross_condition, y) ##### NEW: 03/29/23
    print(conditions)
    # ========================
    #  SEGMENT WINDOWS
    # ======
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)

    # ========================
    # NESTED - CLASSIFICATION
    # ======
    num_cores = 3
    select_window = 0 + 500 # change to arg.t_start
    windows = list(np.where(t_ws==select_window)[0])
    for w_j in windows:
        rclf_ = partial(inner_CV,
                        w_j=w_j,
                        X=X,
                        y=y,
                        conditions=conditions, ## for cross-classification
                        groups=groups,
                        trte=trte,
                        arg=arg,
                        **vars(arg))
        sts = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
        accuracy_df, predict_df = zip(*sts)
        group_fname = sts_to_csv(accuracy_df, arg, w_j, suffix='_acc')  #***
        # sts_to_csv(predict_df, arg, w_j, suffix='_pred')  #***

        