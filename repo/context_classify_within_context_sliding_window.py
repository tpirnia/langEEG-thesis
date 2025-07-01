from __future__ import print_function
import sys, os 
import numpy as np
import pandas as pd
import time, random, copy, argparse
from filelock import FileLock # added 04/25/22

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed


PROJECT_PATH='/project_data/volume0/tpirnia/thesis_repository/'
CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}

def parse_arguments():
    """Parse command line arguments."""
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
    parser.add_argument('--context', required=True, 
                       help="Context condition for train and test sets. \n('US' for unrelated vs. semantically-related or 'UP' for unrelated vs. phonologically-related.)")
    return parser.parse_args()

def setup_paths(window_size, context):
    """Set up project paths and filenames."""
    arg_fname = os.path.join(
        PROJECT_PATH,
        'repo',
        'args',
        f'context_{context}-{context}_w{window_size}.txt'
    )
    results_fname = os.path.join(
        PROJECT_PATH, 
        'results', 
        f'context_accuracy_{context}-{context}_w{window_size}.csv'
    )
    fstem = f'context_accuracy_{context}-{context}_w{window_size}'
    return arg_fname, results_fname, fstem

def load_dependencies():
    """Import necessary project-specific functions."""
    sys.path.append(PROJECT_PATH + 'repo')
    from data_handling.context_condition import load_args, get_contextEEG
    from data_handling.segment import segment_ts, order_by_trials
    from classification.nested_window_ch2 import inner_CV_by_rep as inner_CV
    from classification.models import adjust_y
    return load_args, get_contextEEG, segment_ts, order_by_trials, adjust_y, inner_CV

def prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y):
    """Load and prepare data for classification."""
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH + '/'
    arg.classifier = args.classifier
    arg.window = args.window
    w, step = arg.window, arg.step

    X, y, groups, arg = get_contextEEG(arg, args.subject)
    y = adjust_y(y, arg.classifier)
    return X, y, groups, arg, w, step

def setup_cross_validation(X, y, groups, w, step, arg):
    """Set up cross-validation and window parameters."""
    n_trials, n_channels, arg.t_end = X.shape
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)
    
    cv_outer = CVs[arg.splitter](n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))
    
    w0 = 500  # cue starts at 500
    w_j = (w0 - t_ws[0]) // step
    
    return trte, n_windows, n_trials

def run_classification(X, y, groups, trte, w_j, arg, inner_CV):
    """Run parallel classification."""
    num_cores = 3
    rclf_ = partial(inner_CV,
                    w_j=w_j,
                    X=X,
                    y=y,
                    groups=groups,
                    trte=trte,
                    arg=arg,
                    **vars(arg))
    return Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 

# def adjust_y(y, classifier):
#     if np.logical_and(np.logical_or((classifier == 'ridge'),
#                                     (classifier == 'svm')), 
#                       (np.array_equal(np.unique(y),([0,1])))):
#         y[y==0]= -1
#     elif np.logical_and(
#         (classifier == 'log'),
#         (np.array_equal(np.unique(y),([-1, 0,1])))):
#         y[y==-1]= 2
#     return y

def to_df(sts, arg):
    train_set = arg.suffix_X[-2:].upper()
    test_set = arg.suffix_y[-2:].upper()

    by_cols = ['subject', 'classifier', 'train_set','test_set','window', 'step', 'w_train']

    df = pd.concat(sts,ignore_index=True)
    df['train_set'], df['test_set'] = train_set, test_set
    df = (df.apply(pd.to_numeric, errors='ignore')
               .groupby(by=by_cols, as_index=False).mean(numeric_only=True)
               .drop(columns=['test_trial_num', 'fold']))
    return df

def to_csv(df, filename):
    df_header = ','.join(df.columns)
    df_str = ','.join(map(str, df.values.squeeze()))

    with FileLock(filename + ".lock"):
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write(df_header + '\n')
            f.write(df_str + '\n')

def remove_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError as e:
            print(f"Error removing lock file: {e}")
            
if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments() # classifier, subject, window (size), context
    
    # Setup paths
    arg_fname, results_fname, fstem = setup_paths(args.window, args.context)
    
    # Load dependencies
    load_args, get_contextEEG, segment_ts, order_by_trials, adjust_y, inner_CV = load_dependencies()
    
    # Prepare data
    X, y, groups, arg, w, step = prepare_data(args, arg_fname, get_contextEEG, load_args, adjust_y)
    arg.fstem = fstem #file and directory stem to use throughout...
    t_ws = np.arange(0, X.shape[-1] - w, step)
    
    # Setup cross validation
    trte, n_windows, n_trials = setup_cross_validation(X, y, groups, w, step, arg)
    
    # Segment data
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)
    
    # Run classification
    for w_j in np.arange(n_windows)[t_ws>=500]:
        sts = run_classification(X, y, groups, trte, w_j, arg, inner_CV)
        # Save results
        df = to_df(sts, arg)
        to_csv(df, results_fname)

    remove_lock(results_fname)