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

from classification.permutations import save_permutations, get_permutation_filepath, load_permutations, generate_all_permutations, get_last_perm, get_ith_permutation

PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository"

def fill_path(fname):
    """Create necessary file directories, if missing."""
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))


def to_csv(sts, arg, w_j, ch, nth_perm=None):
    """Save classification accuracy per subject per channel * training window.
    Input is a list of results at each CV fold, 0-1 accuracy of each trial per channel
    Output is a table of test accuracy indexed by: 
        subject, classifier type, window size ("window"), step, training window ("w_train"), channel, fold.
    """
    results_fname = os.path.join(arg.project_path, 
                                 'results',
                                 arg.fstem, 
                                 '{}_{}_accuracy.csv'.format(arg.subject, arg.classifier)
                                )
    fill_path(results_fname)
    acc_df = pd.concat(sts, ignore_index=True) # [0 or 1] accuracy per [channel x fold x trial].
    acc_df = acc_df.apply(pd.to_numeric, errors='ignore') 
    if nth_perm is not None:
        acc_df.insert(acc_df.columns.get_loc('step') + 1, 'perm', nth_perm)
    group_by = list(acc_df.loc[:,:'fold'].columns)  # groupby: (subj, classifier, window, step, w_train, ch, fold)
    acc_agg = acc_df.groupby(by=group_by, as_index=False).mean(numeric_only=True) # mean across trials
    acc_agg = acc_agg.drop(columns=["test_trial_num"]) # drop trial number column
    
    header_str = ','.join(list(acc_agg.columns)) 
    with FileLock(results_fname+".lock"): # added 04/25/22
        if not os.path.exists(results_fname):
            with open(results_fname, "a+") as f:  _ = f.write(header_str+'\n')
        with open(results_fname, "a+") as f:  
            for acc_ in acc_agg.values.squeeze():
                acc_ = ','.join(map(str, acc_))
                _ = f.write(acc_+'\n')
    return results_fname

def remove_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f'Removed lockfile {filename}')
        except OSError as e:
            print(f"Error removing lock file: {e}")
            
def parse_arguments():
    """Parse command line arguments."""
    description = ('File Locations:\n'
              '  Expected arg/config file: {}/repo/args/phase_w<window>.txt\n'
              '  Output results: {}/results/phase_accuracy_sliding_w<window>/\n'
              '  Default window size: 1000 ms\n\n'.format(PROJECT_PATH, PROJECT_PATH))
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description=description)
    parser.add_argument('-c', '--classifier', default='ridge', 
                   help='Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)')
    parser.add_argument('-s', '--subject', required=True, 
                       help='Subject ID (e.g., EP06)')
    parser.add_argument('-w', '--window', type=int, default=1000, 
                       help='Window size in ms (default=1000)')
    parser.add_argument('-ts', '--t_start', type=int, default=0, help='Time (ms) to start classification (default=0)')
    return parser.parse_args()


def setup_paths(window_size):
    """Set up project paths and filenames.
    args: phase_accuracy_by_channel_w{window_size}.txt 
    results: phase_accuracy_by_channel_w{window_size}/{subject}/"""
    arg_fname = os.path.join(PROJECT_PATH,
                             'repo/args',
                             f'phase_by_channel_w{window_size}.txt')
    results_fname = os.path.join(PROJECT_PATH,
                             'results',
                             f'phase_accuracy_by_channel_w{window_size}')
    return arg_fname, results_fname

def load_dependencies():
    """Import necessary experiment-specific functions."""
    sys.path.append(PROJECT_PATH + '/repo')
    from data_handling.phase_condition import load_args, get_phaseEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.models import adjust_y
    from classification.nested_window_by_channel import permute_CV_ch as inner_CV_ch
    return load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV_ch 

def prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y):
    """Load and prepare data for classification."""
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH +'/'
    
    for key, value in vars(args).items():
        setattr(arg, key, value)
        
    arg.fstem=f'phase_accuracy_by_channel_w{arg.window}'
    X, y, groups, arg = get_phaseEEG(arg, args.subject)
    y = adjust_y(y, arg.classifier)
    return X, y, groups, arg, arg.window, arg.step

# def adjust_y(y, classifier_type):
#     """Implement appropriate labels for each classifier."""
#     # Previously:
#     classifiers_needing_neg_one = ['ridge', 'svm', 'svm_linear']
#     if classifier_type in classifiers_needing_neg_one and np.array_equal(np.unique(y), [0,1]):
#         y[y==0] = -1
#     elif classifier_type == 'log' and np.array_equal(np.unique(y), [-1, 0, 1]):
#         y[y==-1] = 2
#     return y

def setup_cross_validation(X, y, groups, outer_splits):
    cv_outer = GroupKFold(n_splits=outer_splits)
    trte = list(cv_outer.split(X, y, groups))
    return trte #, len(t_ws), n_trials

def select_train_windows(n_timepoints, window_size, step, t_start):
    t_0 = 500 # stimulus onset at 500ms in.
    
    # get time (ms) at window, from start of trial (-500 before stim onset -> 1000ms after).
    window_times = np.arange(0 , n_timepoints - window_size, step) - t_0
    
    # get index per window at or after selected starting time (ms).
    window_inds = np.arange(len(window_times))[window_times>=(t_start)] 
    return window_inds

def segment_by_time_channel(X, y, groups, w, step, t_start):
    n_trials, n_channels, n_timepoints = X.shape
    window_inds = select_train_windows(n_timepoints, w, step, t_start)
    
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)
    
    return X, y, groups, np.arange(n_channels), window_inds, n_timepoints


def get_or_generate_permutations(y, groups, n_perms, project_path, subject):
    """Get existing permutations or generate new ones if they don't exist."""
    # Make permutation filepath
    perm_fname = get_permutation_filepath(
        project_path, 
        subject, 
        f'phase_by_channel_n{n_perms}')
    
    # Check if permutations exist
    if os.path.exists(perm_fname):
        perm_dict = load_permutations(perm_fname)
    else:  
        # Generate permutations if they don't exist
        perm_dict = generate_all_permutations(y=y, groups=groups, n_perms=n_perms)
        # Save generated permutations
        save_permutations(perm_dict, perm_fname)
    return perm_dict

def save_perm_progress(file_path, subject, perms_completed, n_perms):
    os.makedirs(file_path, exist_ok=True)
    csv_file = os.path.join(file_path, "permutation_progress.csv")

    # Load existing record or create new
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if subject in df['subject'].values:
            df.loc[df['subject'] == subject, ['permutation_progress', 'permutation_total']] = [perms_completed, n_perms]
        else:
            df = pd.concat([df, 
                            pd.DataFrame({'subject': [subject], 
                                              'permutation_progress': [perms_completed], 
                                              'permutation_total': [n_perms]})], 
                           ignore_index=True)
    else:
        df = pd.DataFrame({'subject': [subject], 
                           'permutation_progress': [perms_completed], 
                           'permutation_total': [n_perms]})

    df.to_csv(csv_file, index=False)

def run_classification(X, y, groups, trte, w_j, ch,arg, inner_CV_ch):
    num_cores = arg.outer_splits
    rclf_ = partial(inner_CV_ch,
                    w_j=w_j,
                    X=X[:,:,ch,:],
                    y=y,
                    groups=groups,
                    trte=trte,
                    ch=ch,
                    arg=arg,
                    **vars(arg))
    return Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte)))



if __name__ == '__main__':
    # ======         ======
    #  SET INPUT ARGUMENTS
    # ======         ======
    args = parse_arguments()

    # SET UP PATHS
    arg_fname, results_fname = setup_paths(args.window)

    # ========================
    #  LOAD DEPENDENCIES:
    load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV_ch = load_dependencies()
    
    # ========================
    # PREPARE DATA:
    X, y, groups, arg, w, step = prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y)

    # ========================
    # NESTED - SPLIT OUTER CV FOLDS:
    # =======
    # trte = setup_cross_validation(X, y, groups, arg.outer_splits)
    cv_outer = GroupKFold(n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))

    # ========================
    #  SEGMENT WINDOWS + Nested windows
    # ======
    X, y, groups, CHS, WINS, arg.t_end = segment_by_time_channel(X, y, groups, w, step, arg.t_start)
    
    
    # ========================
    # NESTED - CLASSIFICATION
    # ======
    perm_dict = get_or_generate_permutations(y, groups, arg.n_perms, PROJECT_PATH, arg.subject)    

    last_perm = get_last_perm(results_fname, arg.subject)
    for perm_i in range(last_perm + 1, arg.n_perms):
        perm_idx = get_ith_permutation(perm_dict, perm_i)
        for w_j, ch in itertools.product(WINS, CHS):
            sts = run_classification(X, y[perm_idx], groups[perm_idx], trte, w_j, ch, arg, inner_CV_ch)
            group_fname = to_csv(sts, arg, w_j, ch, perm_i)
        if perm_i % 10 == 0:
            save_perm_progress(results_fname, arg.subject, perm_i, arg.n_perms) 
    save_perm_progress(arg.project_path, args.subject, perm_i, arg.n_perms)
    if group_fname:
        remove_lock(group_fname)