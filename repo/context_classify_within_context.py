from __future__ import print_function
import sys, os, argparse
import numpy as np
import pandas as pd
from filelock import FileLock 

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed


PROJECT_PATH='/project_data/volume0/tpirnia/thesis_repository/'
CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}

def parse_arguments(args_list=None):
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
    parser.add_argument('--single_window', action='store_true', default=False,
                        help='Run classification on only the first window at stimulus onset (500ms)')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations (default=1000)')
    parser.add_argument('--run_perms', action='store_true', default=False,
                        help='Run permutation testing instead')
    parser.add_argument('--perm_id', type=int, default=None,
                        help='Specific permutation ID to run (for job arrays)')
    parser.add_argument('--perm_start', type=int, default=None,
                        help='Starting permutation ID for range processing')
    parser.add_argument('--perm_end', type=int, default=None, 
                        help='Ending permutation ID for range processing')
    return parser.parse_args(args_list)

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
        'accuracy',
        f'context_accuracy_{context}-{context}_w{window_size}.csv'
    )
    return arg_fname, results_fname

def load_dependencies():
    """Import necessary project-specific functions."""
    sys.path.append(os.path.join(PROJECT_PATH ,'repo'))
    from data_handling.context_condition import load_args, get_contextEEG
    from data_handling.segment import segment_ts, order_by_trials
    # from classification.nested_window_ch2 import inner_CV_by_rep as inner_CV
    from classification.models import adjust_y
    from classification.nested_window import classify_permute,inner_CV
    return load_args, get_contextEEG, segment_ts, order_by_trials, adjust_y, inner_CV,classify_permute

def prepare_data(args, arg_fname, get_phaseEEG, load_args, adjust_y):
    """Load and prepare data for classification."""
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH #+ '/'
    # arg.classifier = args.classifier
    # arg.window = args.window
    # w, step = arg.window, arg.step
    for key, value in vars(args).items():
        setattr(arg, key, value)
    
    arg.fstem = f'context_accuracy_{arg.context}-{arg.context}_w{arg.window}'
    X, y, groups, arg = get_contextEEG(arg, arg.subject)
    y = adjust_y(y, arg.classifier)
    if 'suffix' not in arg: arg.suffix=None
    return X, y, groups, arg, arg.window, arg.step

def select_train_windows(n_timepoints, window_size, step, t_start=0):
    t_0 = 500 # stimulus onset at 500ms in.
    window_times = np.arange(0, n_timepoints - window_size, step) - t_0     # get time (ms) at window, from start of trial (-500 before stim onset -> 1000ms after).
    window_inds = np.arange(len(window_times))[window_times>=(t_start)]     # get index per window at or after selected starting time (ms).
    return window_inds

def setup_cross_validation(X, y, groups, w, step, arg):
    """Set up cross-validation and window parameters."""
    n_trials, _, arg.t_end = X.shape
    window_inds = select_train_windows(arg.t_end, w, step)
    if arg.single_window: window_inds = [window_inds[0]]
    # t_ws = np.arange(0, arg.t_end - w, step)
    # n_windows = len(t_ws)
    
    cv_outer = CVs[arg.splitter](n_splits=arg.outer_splits)
    trte = list(cv_outer.split(X, y, groups))
    
    
    # w0 = 500  # cue starts at 500
    # w_j = (w0 - t_ws[0]) // step
    # return trte, n_windows, n_trials
    return trte, window_inds, n_trials


def run_classification(X, y, groups, trte, w_j, arg, inner_CV):
    """Run parallel classification."""
    # num_cores = 3
    
    # num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 4)) - 1
    num_cores = max(1, min(num_cores, len(trte)))
    rclf_ = partial(inner_CV,
                    w_j=w_j,
                    X=X,
                    y=y,
                    groups=groups,
                    trte=trte,
                    arg=arg,
                    **vars(arg))
    return Parallel(n_jobs=num_cores)(delayed(rclf_)(fld) for fld in np.arange(len(trte))) 

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

def to_csv(df, arg, filename=None, nth_perm=None):
    if nth_perm is not None:
        df.insert(df.columns.get_loc('step') + 1, 'perm', nth_perm)
        filename = os.path.join(arg.project_path, 
                                'results',
                                'permutations',
                                arg.fstem, 
                                '{}_{}_accuracy.csv'.format(arg.subject, arg.classifier))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_header = ','.join(df.columns)
    df_str = ','.join(map(str, df.values.squeeze()))
    if filename is not None:
        with FileLock(filename + ".lock"):
            file_exists = os.path.exists(filename)
            with open(filename, "a") as f:
                if not file_exists:
                    f.write(df_header + '\n')
                f.write(df_str + '\n')
        return filename

def remove_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except OSError as e:
            print(f"Error removing lock file: {e}")

def perm_setup(arg, y, groups):
    import os  # You'll need this import too
    from classification.permutations import (save_permutations, get_permutation_filepath, 
                                         load_permutations, generate_all_permutations, 
                                         get_last_perm, get_ith_permutation, save_perm_progress)
    
    def get_or_generate_permutations():
        """Get existing permutations or generate new ones if they don't exist."""
        # Make permutation filepath
        perm_fname = get_permutation_filepath(
            arg.project_path, 
            arg.subject, 
            f'context_n{arg.n_perms}')
        
        # Check if permutations exist
        if os.path.exists(perm_fname):
            perm_dict = load_permutations(perm_fname)
            print(f"loading permutations from {perm_fname}")
        else:  
            # Generate permutations if they don't exist
            perm_dict = generate_all_permutations(y=y, groups=groups, n_perms=arg.n_perms)
            # Save generated permutations
            save_permutations(perm_dict, perm_fname)
        return perm_dict
    
    perm_dict = get_or_generate_permutations()
    
    progress_path = os.path.join(arg.project_path, 'results', 'permutations', arg.fstem)
    
    last_perm = get_last_perm(progress_path, arg.subject)  # Fixed: was args.subject
    
    if last_perm + 1 > arg.n_perms:
        print(f'{last_perm} of {arg.n_perms} permutations already complete!')
        
    return perm_dict, progress_path, last_perm, get_ith_permutation, save_perm_progress

def perms_to_run(arg, last_perm):
    if arg.perm_start is not None and arg.perm_end is not None:
        arg.perm_start = arg.perm_start - 1  # Convert to 0-based indexing
        arg.perm_end = arg.perm_end + 1  
    elif arg.perm_id is not None:
        arg.perm_start = arg.perm_id
        arg.perm_end = arg.perm_id+1
    else: 
        arg.perm_start = last_perm + 1
        arg.perm_end = arg.n_perms

    return range(arg.perm_start, arg.perm_end)


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    # args = parse_arguments['-c','svm','-s', 'EP03','-w','1000','--context','US','--single_window','--run_perms'])

    # setup paths
    arg_fname, results_fname = setup_paths(args.window, args.context)

    # load dependencies
    load_args, get_contextEEG, segment_ts, order_by_trials, adjust_y, inner_CV, classify_permute = load_dependencies()

    # prepare data
    X, y, groups, arg, w, step = prepare_data(args, arg_fname, get_contextEEG, load_args, adjust_y)
    
    # setup cross validation
    trte, WINS, n_trials = setup_cross_validation(X, y, groups, w, step, arg)

    # segment data
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)

    if arg.run_perms:
        # load perm dependences, last permutation completed, & permutation indices.
        perm_dict, progress_path, last_perm, get_ith_permutation, save_perm_progress = perm_setup(arg, y, groups)

        perm_range = perms_to_run(arg, last_perm)
        # Process a range of permutations (for chunked jobs)
        print(f"Processing permutations {arg.perm_start} to {arg.perm_end}")

        for perm_i in perm_range:

            if 0 <= perm_i < arg.n_perms:
                print(f"Running permutation {perm_i}")
                perm_idx = get_ith_permutation(perm_dict, perm_i)
                
                for w_j in WINS:
                    sts = run_classification(X, y[perm_idx], groups[perm_idx], trte, w_j, arg, classify_permute)
                    df_filename = to_csv(to_df(sts, arg), arg, nth_perm=perm_i)

                if perm_i % 10 == 0: save_perm_progress(progress_path, arg.subject, perm_i, arg.n_perms)
        if 'perm_i' in locals():  # save if permutations ran
            save_perm_progress(progress_path, arg.subject, perm_i, arg.n_perms)
        print(f"Completed permutations {arg.perm_start} to {arg.perm_end}")
    else:
        # run classification (non-permutation)
        for w_j in WINS:
            sts = run_classification(X, y, groups, trte, w_j, arg, inner_CV)
            # Save results
            df_filename = to_csv(to_df(sts, arg), arg, filename=results_fname)

    # remove lock if df_filename was actually created
    if 'df_filename' in locals() and df_filename is not None:
        remove_lock(df_filename)