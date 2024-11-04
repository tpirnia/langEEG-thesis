# train on context condition, test on phase
# permutation is batched across all subjects.
from __future__ import print_function
import sys, time
import numpy as np

from sklearn.model_selection import StratifiedKFold, GroupKFold
from functools import partial
from joblib import Parallel, delayed

def get_permutations( fname='/project_data/volume0/tpirnia/data/EEGWS/data/args/ch2/permutations_s101_500.pkl'):
    """ get indices for 3 folds of the ith permutation """
    import pickle
    with open(fname, 'rb') as f:
        permutations = pickle.load(f) 
    return permutations # results in a list of three (tr,te) one per fold.

CVs={"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}
subject_list = "EP03 EP05 EP06 EP07 EP09 EP10 EP11 EP12 EP13 EP14 EP15 EP16 EP17 EP18 EP19 EP20 EP21 EP22 EP23 EP25 EP26 EP27 EP28 EP29 EP30 EP31 EP32 EP33".split(" ")

if __name__ == '__main__':
    print('Starting cross-classification for each subject:')
    for subject in subject_list:
        print(f'\t{subject}')
        # ======         ======
        #  SET INPUT ARGUMENTS 
        # ======         ======
        project_path = sys.argv[1]
        arg_fname = project_path + "/data/args/" + sys.argv[2]
        _ = sys.argv[3]
        cross_condition = str(sys.argv[4])
        within_condition= str(sys.argv[5])
        try:
            last_perm = int(sys.argv[7])
            last_perm = last_perm + 1 if last_perm>0 else last_perm
        except IndexError:
            last_perm = 0
        # ========================
        #  IMPORT FUNCTIONS 
        sys.path.append(project_path + '/maybe_repo')
        from data_handling.context_condition import load_args, get_contextEEG
        from data_handling.segment import segment_array, segment_1Darray
        from classification.nested_window_across_condition import inner_CV_phase_context as inner_CV #***
        from classification.cross_condition import crossConflict
        from utils.utils import sts_save_crossconflict as sts_out  # type: ignore
        # ========================
        #  LOAD DATA:
        # ==========
        arg = load_args(arg_fname, subject)

        if "data_suffix" in arg:
            data_suffix = arg.data_suffix
            y2_fname = arg.project_path + 'data/' + subject + '/' + arg.data_suffix + '/'
        else: 
            data_suffix=None
            y2_fname = arg.project_path + 'data/' + subject + '/' 

        w, step = arg.window, arg.step

        X, y, groups, arg = get_contextEEG(arg, subject, suffix=data_suffix)
        y2 =  np.load(y2_fname + 'y' + arg.suffix_y2 + '.npy') #loading separate cross condition test labels
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
        select_window = int(sys.argv[6]) + 500
        windows = list(np.where(t_ws==select_window)[0])
        if arg.perm:
            # seed = int(time.time())
            # rngen = np.random.RandomState(seed)
            permutations = get_permutations()
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
                    sts_out(accuracy_df, arg, w_j, suffix='_acc-batch_perm')  #***
                    # sts_out(predict_df, arg, w_j, suffix='_pred-perm')  #***
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
                sts_out(accuracy_df, arg, w_j, suffix='_acc')  #***
            # sts_out(predict_df, arg, w_j, suffix='_pred')  #***


