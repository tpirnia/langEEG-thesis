from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock

from sklearn.model_selection import GroupKFold
from functools import partial
from joblib import Parallel, delayed
# from utils.utils import fill_path, remove_lock


# Define project path
PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository/"

def fill_path(fname):
    """Create necessary file directories, if missing."""
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def to_csv(sts, arg, w_j, filename, nth_perm=None):
    """Save classification accuracy per subject."""
    test_df = pd.concat(sts, ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    if nth_perm is not None:
        test_df.insert(test_df.columns.get_loc('step') + 1, 'perm', nth_perm)
        filename = os.path.join(arg.project_path, 
                                'results',
                                'permutations',
                                arg.fstem, 
                                '{}_{}_accuracy.csv'.format(arg.subject, arg.classifier))
        fill_path(filename)
    
    by_cols = list(test_df.columns.values[:6 if nth_perm is not None else 5])
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean(numeric_only=True)
    acc_agg.drop(columns=["fold", "test_trial_num", "t_0"], inplace=True)
    
    col_header = ','.join(acc_agg.columns)
    acc_agg_str = ','.join(map(str, acc_agg.values.squeeze()))

    with FileLock(filename + ".lock"):
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write(col_header + '\n')
            f.write(acc_agg_str + '\n')
    return filename

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
        print(f"loading permutations from {perm_fname}")
    else:  
        # Generate permutations if they don't exist
        perm_dict = generate_all_permutations(y=y, groups=groups, n_perms=n_perms)
        # Save generated permutations
        save_permutations(perm_dict, perm_fname)
    return perm_dict

def remove_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError as e:
            print(f"Error removing lock file: {e}")
            
# def adjust_y(y, classifier_type):
#     """Implement appropriate labels for each classifier."""
#     classifiers_needing_neg_one = ['ridge', 'svm', 'svm_linear']
#     if classifier_type in classifiers_needing_neg_one and np.array_equal(np.unique(y), [0,1]):
#         y[y==0] = -1
#     elif classifier_type == 'log' and np.array_equal(np.unique(y), [-1, 0, 1]):
#         y[y==-1] = 2
#     return y

def parse_arguments():
    """Parse command line arguments."""
    description = ('File Locations:\n'
              '  Input config: {}/repo/args/phase_w<window>.txt\n'
              '  Output results: {}/results/phase_accuracy_sliding_w<window>.csv\n\n'.format(PROJECT_PATH, PROJECT_PATH))
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description=description)
    parser.add_argument('-c', '--classifier', default='ridge', 
                   help='Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)')
    parser.add_argument('-s', '--subject', required=True, 
                       help='Subject ID (e.g., EP06)')
    parser.add_argument('--n_perms', type=int, default=1000,
                   help='Number of permutations (default=1000)')
    parser.add_argument('--run_perms', action='store_true', default=False,
                   help='Run permutation testing instead')
    parser.add_argument('-w', '--window', type=int, default=1000, 
                       help='Window size in ms (default=1000)')
    parser.add_argument('--single_window', action='store_true', default=False,
                       help='Run classification on only the first window at stimulus onset (500ms)')
    
    return parser.parse_args()

def setup_paths(window_size):
    """Set up project paths and filenames."""
    arg_fname = '{}/repo/args/phase_w{}.txt'.format(PROJECT_PATH, window_size)
    if window_size == 1000:
        results_fname = '{}/results/phase_accuracy_w{}.csv'.format(PROJECT_PATH, window_size)
    else: 
        results_fname = '{}/results/phase_accuracy_sliding_w{}.csv'.format(PROJECT_PATH, window_size)
    return arg_fname, results_fname

def load_dependencies():
    """Import necessary project-specific functions."""
    sys.path.append(PROJECT_PATH + '/repo')
    from data_handling.phase_condition import load_args, get_phaseEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.models import adjust_y
    from classification.nested_window import inner_CV,classify_permute
    return load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV, classify_permute

def prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y):
    """Load and prepare data for classification."""
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH + '/'
    
    for key, value in vars(args).items():
        setattr(arg, key, value)
        
    if args.single_window and arg.window != 50:
        arg.fstem = f'phase_accuracy_w{arg.window}'
    else:  
        arg.fstem = f'phase_accuracy_sliding_w{arg.window}'
        
    X, y, groups, arg = get_phaseEEG(arg, arg.subject, 'phase')
    y = adjust_y(y, arg.classifier)
    
    return X, y, groups, arg, arg.window, arg.step

def setup_cross_validation(X, y, groups, w, step, arg, single_window=False):
    """Set up cross-validation and window parameters."""
    n_trials, n_channels, arg.t_end = X.shape
    window_inds = select_train_windows(arg.t_end, w, step)
    
    if single_window or 'sliding' not in arg.fstem:
        # Use only the first window at stimulus onset (t=500ms)
        window_inds = [window_inds[0]]
    
    cv_outer = GroupKFold(n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups)) 
    return trte, window_inds, n_trials

def select_train_windows(n_timepoints, window_size, step, t_start=0):
    t_0 = 500 # stimulus onset at 500ms in.
    # get time (ms) at window, from start of trial (-500 before stim onset -> 1000ms after).
    window_times = np.arange(0, n_timepoints - window_size, step) - t_0
    # get index per window at or after selected starting time (ms).
    window_inds = np.arange(len(window_times))[window_times>=(t_start)] 
    return window_inds

def run_classification(X, y, groups, trte, w_j, arg, inner_CV):
    """Run parallel classification."""
    num_cores = 20
    rclf_ = partial(inner_CV,
                   w_j=w_j,
                   X=X,
                   y=y,
                   groups=groups,
                   trte=trte,
                   arg=arg,
                   **vars(arg))
    
    return Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) 
                                    for fld in np.arange(len(trte)))

def run_classify_permute(X, y, groups, trte, w_j, arg, classify_func):
    """Run parallel classification for permutations."""
    num_cores = 20
    rclf_ = partial(classify_func,  # Use the passed function
                   w_j=w_j,
                   X=X,
                   y=y,
                   groups=groups,
                   trte=trte,
                   arg=arg,
                   **vars(arg))
    
    return Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) 
                                    for fld in np.arange(len(trte)))

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Setup paths
    arg_fname, results_fname = setup_paths(args.window)
    
    # Load dependencies
    load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV, classify_permute = load_dependencies()
    
    # Prepare data
    X, y, groups, arg, w, step = prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y)
    
    # Setup cross validation
    trte, WINS, n_trials = setup_cross_validation(X, y, groups, w, step, arg, args.single_window)
    
    # Segment data
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)
    
    filename = None
    
    if args.run_perms:
        from classification.permutations import (save_permutations, get_permutation_filepath, 
                                                 load_permutations, generate_all_permutations, 
                                                 get_last_perm, get_ith_permutation,save_perm_progress)

        perm_dict = get_or_generate_permutations(y, groups, arg.n_perms, arg.project_path, args.subject)
        progress_path = os.path.join(arg.project_path,'results','permutations',arg.fstem)
        last_perm = get_last_perm(progress_path, args.subject)
        print(f"Previously completed {last_perm} permutations.")
        for perm_i in range(last_perm + 1, arg.n_perms):
            perm_idx = get_ith_permutation(perm_dict, perm_i)
            arg.perm_i = perm_i
            # Run classification for all windows with permuted data
            for w_j in WINS:
                sts = run_classify_permute(X, y[perm_idx], groups[perm_idx], 
                                         trte, w_j, arg, classify_permute)
                filename = to_csv(sts, arg, w_j, "", nth_perm=perm_i)
            if perm_i % 10 == 0:
                save_perm_progress(progress_path, args.subject, perm_i, args.n_perms)
        # if arg.n_perms > last_perm + 1:  
        save_perm_progress(progress_path, args.subject, perm_i, arg.n_perms)
        
    else: 
        # Run classification
        for w_j in WINS:
            sts = run_classification(X, y, groups, trte, w_j, arg, inner_CV)
            # Save results
            filename = to_csv(sts, arg, w_j, results_fname)
    if filename:  
        remove_lock(filename)