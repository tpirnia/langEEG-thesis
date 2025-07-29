import sys, os, argparse
import numpy as np
import pandas as pd

from functools import partial
from joblib import Parallel, delayed
from filelock import FileLock

# the project path also needs to be changed in the arg file.
PROJECT_PATH = "/project_data/volume0/tpirnia/thesis_repository/"


def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname), exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    description = (
        "File Locations:\n"
        "  Input config: {}/repo/args/phase_w<window>.txt\n"
        "  Output results: {}/results/phase_accuracy_sliding_w<window>.csv\n\n".format(
            PROJECT_PATH, PROJECT_PATH
        )
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=description
    )
    parser.add_argument(
        "-c",
        "--classifier",
        default="ridge",
        help="Classifier type (ridge, svm, svm_linear, log, decisiontree; default=ridge)",
    )
    parser.add_argument(
        "-s", "--subject", required=True, help="Subject ID (e.g., EP06)"
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=1000,
        help="Window size in ms (default=1000)",
    )
    parser.add_argument(
        "--single_window",
        action="store_true",
        default=False,
        help="Run classification on only the first window at stimulus onset (500ms)",
    )
    parser.add_argument(
        "--n_perms",
        type=int,
        default=1000,
        help="Number of permutations (default=1000)",
    )
    parser.add_argument(
        "--run_perms",
        action="store_true",
        default=False,
        help="Run permutation testing instead",
    )
    return parser.parse_args()


def setup_paths(window_size):
    """Set up project paths and filenames."""
    arg_fname = "{}/repo/args/phase_loo_w{}.txt".format(PROJECT_PATH, window_size)
    return arg_fname


def prepare_data(arg_fname, args):
    arg = load_args(arg_fname, args.subject)
    arg.project_path = PROJECT_PATH

    for key, value in vars(args).items():
        setattr(arg, key, value)

    arg.fstem = f"phase_accuracy_loo_w{arg.window}"

    # X, y, groups, arg = get_phaseEEG(arg, arg.subject, 'phase')
    # y = adjust_y(y, arg.classifier)
    # return X, y, groups, arg, arg.window, arg.step
    return arg, arg.window, arg.step


def get_or_generate_permutations(
    y_train, groups_train, y_test, groups_test, arg, seed=1234
):
    """Get existing permutations or generate new ones if they don't exist."""
    # Make permutation filepath
    perm_fname = get_permutation_filepath(
        arg.project_path, arg.subject, f"phase_loo_n{arg.n_perms}"
    )

    # Check if permutations exist
    if os.path.exists(perm_fname):
        perm_dict = load_permutations(perm_fname)
        print(f"loading permutations from {perm_fname}")
    else:
        # Generate permutations if they don't exist
        perm_dict = generate_all_permutations(
            y_train=y_train,
            groups_train=groups_train,
            y_test=y_test,
            groups_test=y_test,
            n_perms=arg.n_perms,
            seed=seed,
        )
        # Save generated permutations
        save_permutations(perm_dict, perm_fname)
    return perm_dict


if __name__ == "__main__":
    # ======         ======
    #  SET INPUT ARGUMENTS
    # ======         ======
    args = parse_arguments()
    arg_fname = setup_paths(args.window)

    # arg_fname = PROJECT_PATH + sys.argv[1] #'repo/args/phase_multisubject_w50_s10.txt'
    # subject = sys.argv[2] #EP06

    subject_list = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(
        " "
    )

    if args.subject in subject_list:
        subject_list.remove(args.subject)

    # ========================
    #  IMPORT FUNCTIONS
    # ==========
    sys.path.append(PROJECT_PATH + "repo")
    from data_handling.phase_condition import (
        load_args,
        get_phaseEEG,
        load_multi_subject,
    )
    from data_handling.preprocessing import get_bad_channels
    from data_handling.segment import segment

    # from classification.models import adjust_y

    # ========================
    #  LOAD DATA:
    # ==========

    arg, w, step = prepare_data(arg_fname, args)
    # trained on all subjects but j
    # j: left out subject to be tested on
    X_j, y_j, groups_j, arg = get_phaseEEG(arg, arg.subject)

    # load all train subjects:
    X_train, y_train, groups_train, ind_subj = load_multi_subject(
        arg,
        subject_list,
    )

    arg.t_end = X_train.shape[-1]
    t_ws = np.arange(0, arg.t_end - w, step)
    n_windows = len(t_ws)

    # ========================
    #  SEGMENT WINDOWS
    # ======
    X_j, y_j, groups_j = segment(X_j, y_j, groups_j, w, step)
    X_train, y_train, groups_train = segment(X_train, y_train, groups_train, w, step)
    # ========================
    # CLASSIFICATION (train/test windows in parallel)
    # ======
    num_cores = 5
    windows = np.arange(n_windows)[t_ws >= 500]
    # print(arg.__dict__)
    if arg.run_perms:
        from classification.permutations import (
            save_permutations,
            get_permutation_filepath,
            load_permutations,
            generate_all_permutations,
            get_last_perm,
            get_ith_permutation,
            save_perm_progress,
        )
        from classification.kfold_cv import classify_permute_test as classify

        progress_path = os.path.join(
            arg.project_path, "results", "permutations", arg.fstem
        )

        # Generate permutations for both train and test data
        perm_dict = get_or_generate_permutations(
            y_train=y_train,
            groups_train=groups_train,
            y_test=y_j,
            groups_test=groups_j,
            arg=arg,
            seed=1234,
        )
        # perm_dict = get_or_generate_permutations(y_train, groups_train, y_test, groups_test, arg, seed=1234)
        last_perm = get_last_perm(progress_path, arg.subject)

        for perm_i in range(last_perm + 1, arg.n_perms):
            train_idx, test_idx = get_ith_permutation(perm_dict, perm_i)
            arg.ith_perm = perm_i
            clf_ = partial(
                classify,
                X_train=X_train,
                y_train=y_train[train_idx],
                groups_train=groups_train[train_idx],
                X_test=X_j,
                y_test=y_j[test_idx],
                ind_subj=ind_subj,
                arg=arg,
                **vars(arg),
            )

            _ = Parallel(n_jobs=num_cores)(delayed(clf_)(w_j) for w_j in windows)

            if perm_i % 10 == 0:
                save_perm_progress(progress_path, args.subject, perm_i, args.n_perms)
        if arg.n_perms >= last_perm + 1:
            save_perm_progress(progress_path, args.subject, perm_i, arg.n_perms)

    else:
        from classification.kfold_cv import classify

        clf_ = partial(
            classify,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_j,
            y_test=y_j,
            ind_subj=ind_subj,  # not used for group analysis
            arg=arg,
            **vars(arg),
        )
        _ = Parallel(n_jobs=num_cores)(delayed(clf_)(w_j) for w_j in windows)
