# permutation batched across all subjects.
from __future__ import print_function
import sys, time, argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed
import pandas as pd
from filelock import FileLock # added 04/25/22
import os
        
PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository"

within_dict = {
    'phonological':{
        'label': 'UP-CI',
        'train':['UP',1], # label for phonological in y is 1
        'test': ['phase',3] # label for phase in y is 3
    },
    'semantic':{
        'label': 'US-CI',
        'train':['US',2], # label for semantic in y is 2
        'test': ['phase',3] # label for phase in y is 3
    }
}


def sts_out(sts, arg, w_j):
    test_df = pd.concat(sts, ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    out_name = arg.out_name
    cols_binary = [
        "subject",
        "classifier",
        "within_condition",
        "cross_condition",
        "window",
        "step",
        "w_train"]
    acc_agg_bin = test_df.groupby(by=cols_binary, as_index=False).mean(numeric_only=True).drop(columns=["fold","test_trial_num","test_trial_val"])
    group_fname = arg.project_path + f'/results/accuracy/{arg.fstem}.csv'
    os.makedirs(os.path.dirname(group_fname), exist_ok=True)
    save_dataframe(group_fname, acc_agg_bin) #was called binary before
    return group_fname

def save_dataframe(filename, dataframe):
    lock = FileLock(filename + ".lock")
    add_column_names = not os.path.exists(filename)

    with lock.acquire():
        with open(filename, "a+") as f:
            if add_column_names:
                f.write(','.join(dataframe.columns.to_list()) + '\n')

            # Convert the dataframe to a string and write it to the file in a single call
            dataframe_str = dataframe.to_csv(header=False, index=False, lineterminator='\n')
            f.write(dataframe_str)
            return dataframe_str
def remove_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError as e:
            print(f"Error removing lock file: {e}")

def parse_arguments(args_list=None):
    """Parse arguments to set the subject and training set. Test labels are always phase-straight vs reverse.
    
    Parameters:
    -----------
    args_list : list, optional
        List of argument strings. If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(description='Cross-classification across subjects')
    parser.add_argument('-s', '--subject', required=True, help='Subject ID (e.g., EP06)')
    parser.add_argument('--train', required=True, help='Within condition (semantic or phonological)')
    parser.add_argument('-c', '--classifier', default='svm', 
                   help='Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)')
    parser.add_argument('--arg_file', default='context_across_phase',
                       help='Configuration file name: (default: args/context_cross_phase_w1000.txt)')
    parser.add_argument('-w', '--window', type=int, default=1000, 
                       help='Window size in ms (default=1000)')
    parser.add_argument('--single_window', action='store_true', default=False,
                       help='Run classification on only the first window at stimulus onset (500ms)')
    parser.add_argument('--step', type=int, default=500, 
                       help='Step size size in ms (default=500)')
    parser.add_argument('--t_start', type=int, default=0, 
                       help='start window. Default is cue presentation (at 0ms)')
    parser.add_argument('--run_perms', action='store_true', default=False,
                   help='Run permutation testing instead')
    parser.add_argument('--test', default='phase',
                       help='Cross condition--it should always stay as phase here.')
    return parser.parse_args(args_list)

CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}
# subject_list = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(" ")

def assign_args(arg_fname, args,load_args):
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH +'/'
    arg.fstem = f"{arg.out_name}_{within_dict[args.train]['label']}_w{args.window}"
    print(f'Saved to:{arg.fstem}.csv')
    for key, value in vars(args).items():
        setattr(arg, key, value)
    return arg, arg.window, arg.step

def get_permutations(fname):
    """ get indices for 3 folds of the ith permutation """
    import pickle
    with open(fname, 'rb') as f:
        permutations = pickle.load(f) 
    return permutations # results in a list of three (tr,te) one per fold.
    
if __name__ == '__main__':

  
    # ========================
    #  IMPORT FUNCTIONS 
    sys.path.append(PROJECT_PATH + '/repo')
    from data_handling.context_condition import load_args, get_contextEEG
    from data_handling.segment import segment_array, segment_1Darray
    from classification.nested_window_across_condition import inner_CV_phase_context as inner_CV #***
    from classification.cross_condition import crossConflict
        

    # ======         ======
    #  SET INPUT ARGUMENTS 
    # ======         ======
    args = parse_arguments()
    # args = parse_arguments(['-s','EP03','--train','semantic'])
    arg, w, step = assign_args((f'{PROJECT_PATH}/repo/args/{args.arg_file}_w{args.window}'), args,load_args)
    cross_condition = args.test
    within_condition = args.train

      
    print(f'Starting cross-classification for {arg.subject}')
    # ========================
    #  LOAD DATA:
    # ==========

    if "data_suffix" in arg:
        data_suffix = arg.data_suffix
        y2_fname =os.path.join(arg.project_path,
                               'data',
                               arg.subject,
                               arg.data_suffix) # this is where all data for context experiments live (including phase data. 
    else: 
        data_suffix=None
        y2_fname = arg.project_path + 'data/' + arg.subject + '/' 

    X, y, groups, arg = get_contextEEG(arg, arg.subject, suffix=data_suffix)

    y2 =  np.load(y2_fname + '/y' + arg.suffix_y2 + '.npy') #loading separate cross condition test labels

    n_trials, _, arg.t_end = X.shape
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    cv_outer = CVs[arg.splitter](n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))

    conditions = crossConflict(within_condition, cross_condition) ##### NEW: 05/23/23
    arg.out_name = arg.out_name + "_" + conditions.suff

    X = segment_array(X, w, step)
    y, y2, groups = (segment_1Darray(a, n_windows) for a in [y, y2, groups])

    # ========================
    # NESTED - CLASSIFICATION
    # ======
    num_cores = len(trte)
    select_window = args.t_start + 500  # Changed from int(sys.argv[6])
    windows = list(np.where(t_ws==select_window)[0])

    if arg.run_perms:
        permutations = get_permutations()  # Note: needs filename parameter
        for arg.ith_perm in np.arange(last_perm, arg.n_perms):
            for w_j in windows:
                rclf_ = partial(inner_CV,
                                w_j=w_j,
                                X=X,
                                y=y,
                                y2=y2,
                                conditions=conditions, 
                                groups=groups,
                                trte=trte,
                                arg=arg,
                                n_jobs=1,
                                rand_gen=permutations[arg.ith_perm],
                                **vars(arg))
                sts = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
                accuracy_df, predict_df = zip(*sts)
                group_fname = sts_out(accuracy_df, arg, w_j, suffix='batch_perm')  #***
        remove_lock(group_fname)
    else:
        for w_j in windows:
            rclf_ = partial(inner_CV,
                            w_j=w_j,
                            X=X,
                            y=y,
                            y2=y2,
                            conditions=conditions, 
                            groups=groups,
                            trte=trte,
                            arg=arg,
                            n_jobs=1,
                            **vars(arg))
            sts = Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 
            accuracy_df, predict_df = zip(*sts)
            group_fname = sts_out(accuracy_df, arg, w_j)  #***
        remove_lock(group_fname)