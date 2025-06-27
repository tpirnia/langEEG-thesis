from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock

from sklearn.model_selection import GroupKFold
from functools import partial
from joblib import Parallel, delayed

# Define project path
PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository"

def fill_path(fname):
    """Create necessary file directories, if missing."""
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def to_csv(sts, arg, filename):
    """Save classification accuracy per subject."""
    test_df = pd.concat(sts, ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')

    by_cols = list(test_df.columns.values[:5])
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    acc_agg.drop(columns=["fold", "test_trial_num", "t_0"], inplace=True)

    col_header = ','.join(acc_agg.columns)
    acc_agg_str = ','.join(map(str, acc_agg.values.squeeze()))

    with FileLock(filename + ".lock"):
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write(col_header + '\n')
            f.write(acc_agg_str + '\n')

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
              '  Output results: {}/results/phase_accuracy_w<window>.csv\n\n'.format(PROJECT_PATH, PROJECT_PATH))

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                   description=description)
    parser.add_argument('-c', '--classifier', default='ridge', 
                   help='Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)')
    parser.add_argument('-s', '--subject', required=True, 
                       help='Subject ID (e.g., EP06)')
    parser.add_argument('-w', '--window', type=int, default=1000, 
                       help='Window size in ms (default=1000)')
    return parser.parse_args()

def setup_paths(window_size):
    """Set up project paths and filenames."""
    arg_fname = '{}/repo/args/phase_w{}.txt'.format(PROJECT_PATH, window_size)
    results_fname = '{}/results/phase_accuracy_w{}.csv'.format(PROJECT_PATH, window_size)
    return arg_fname, results_fname

def load_dependencies():
    """Import necessary project-specific functions."""
    sys.path.append(PROJECT_PATH + '/repo')
    from data_handling.phase_condition import load_args, get_phaseEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.models import adjust_y
    from classification.nested_window import inner_CV
    return load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV

def prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y):
    """Load and prepare data for classification."""
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH + '/'
    arg.classifier = args.classifier
    arg.window = args.window
    w, step = arg.window, arg.step
    
    arg.fstem = f'phase_accuracy_w{w}'
    X, y, groups, arg = get_phaseEEG(arg, args.subject, 'phase')
    y = adjust_y(y, arg.classifier)
    return X, y, groups, arg, w, step

def setup_cross_validation(X, y, groups, w, step, arg):
    """Set up cross-validation and window parameters."""
    n_trials, n_channels, arg.t_end = X.shape
    t_ws = np.arange(0, arg.t_end - w, step)
    
    cv_outer = GroupKFold(n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))
    
    w0 = 500  # cue starts at 500
    w_j = (w0 - t_ws[0]) // step
    
    return trte, w_j, n_trials

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

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Setup paths
    arg_fname, results_fname = setup_paths(args.window)
    
    # Load dependencies
    load_args, get_phaseEEG, segment_ts, order_by_trials, adjust_y, inner_CV = load_dependencies()
    
    # Prepare data
    X, y, groups, arg, w, step = prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y)
    
    # Setup cross validation
    trte, w_j, n_trials = setup_cross_validation(X, y, groups, w, step, arg)
    
    # Segment data
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)
    
    # Run classification
    sts = run_classification(X, y, groups, trte, w_j, arg, inner_CV)
    
    # Save results
    to_csv(sts, arg, results_fname)
    remove_lock(results_fname)