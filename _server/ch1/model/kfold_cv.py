import sys, os, time
import numpy as np
import pandas as pd
from seglearn.transform import Segment
from sklearn.model_selection import GridSearchCV, GroupKFold
from classification.models import make_model, predict_results, prediction_accuracy
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d
from filelock import FileLock # added 04/25/22
import gc

def classify(w_j, X_train, y_train, groups_train, X_test, y_test, ind_subj,
                   inner_splits, classifier, subject, suffix, arg=None, **_):
    """    # input x, y of train  and x,y of test.
        # grab x_train for a specific time window.
        # k fold cv for parameter selection/training.
        # train model with x_train
        # use trained model to classify all windows for all subjects.
        # output of results should be: [train_subj, clf, w, s, w_j, fld,
        #             train_acc, test_subj, test_trial, test_acc{1,..,m}, permutation(np.nan)]"""
    
    X_train, y_train, groups_train = get_segment(w_j, 
                                                 X_train, 
                                                 y_train, 
                                                 groups_train,
                                                )
    cv = list(GroupKFold(n_splits=inner_splits).split(X_train, 
                                                      y_train, 
                                                      groups_train))
    mod_ = train(classifier, 
                 X_train, 
                 y_train, 
                 groups_train, 
                 cv)
    
    test_accuracy = test(mod_, 
                         X_test, 
                         y_test)
    
    test_df = compile_results(arg, 
                              w_j, 
                              mod_.best_score_, 
                              ind_subj, 
                              y_test, 
                              test_accuracy)
    save_df(test_df, arg, w_j)#return test_df
    
def classify_permute_test(w_j, X_train, y_train, groups_train, X_test, y_test, ind_subj,
                   inner_splits, classifier, subject, suffix, arg=None, **_):
    """ permute test labels and measure accuracy of models trained on non-permuted data
        # input x, y of train  and x,y of test.
        # grab x_train for a specific time window.
        # k fold cv for parameter selection/training.
        # train model with x_train
        # use trained model to classify all windows for all subjects.
        # output of results should be: [train_subj, clf, w, s, w_j, fld,
        #             train_acc, test_subj, test_trial, test_acc{1,..,m}, permutation]"""

    X_train, y_train, groups_train = get_segment(w_j, 
                                                 X_train, 
                                                 y_train, 
                                                 groups_train,
                                                )
    
    cv = list(GroupKFold(n_splits=inner_splits).split(X_train, 
                                                      y_train, 
                                                      groups_train))
    
    mod_ = train(classifier, 
                 X_train, 
                 y_train, 
                 groups_train, 
                 cv)
    TDF = []
    rngen = np.random.RandomState(1234)
    X_test = tw_to_3d(X_test)
    for perm in range(arg.n_perms):
        ind_perm = _shuffle(y_test,ind_subj,rngen)
        test_accuracy = test_permute(mod_, 
                                     X_test, 
                                     y_test[ind_perm])

        #test_df.append(compile_results_perm(arg, 
        test_df = compile_results_perm(arg, 
                                       w_j, 
                                       mod_.best_score_, 
                                       ind_subj[ind_perm], 
                                       y_test[ind_perm], 
                                       test_accuracy,
                                       y_inds=ind_perm,
                                       perm=perm)
        TDF.append(test_df)
        del(test_df)
        gc.collect()
    TDF = pd.concat(TDF,ignore_index=True)
    save_df(TDF, arg, w_j)
    #test_df = pd.concat(test_df,ignore_index=True)
    #save_df(test_df, arg, w_j) #return test_df

def get_segment(w_j, X, y, groups):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    X = np.squeeze(X[:, w_j,:,:])
    y = np.squeeze(y[:, w_j])
    groups = np.squeeze(groups[:, w_j])
    return X, y, groups

def train(classifier, X_train, y_train, groups_train, cv):
    modelCV = make_model(classifier, cv) 
    trained_modelCV = modelCV.fit(X_train, y_train)
    return trained_modelCV

def test(mod, X, y):
    n_test = len(y)
    X = tw_to_3d(X)
    y = target_to_1d(y)
    yhat = predict_results(mod, X, y)
    test_accuracy = prediction_accuracy(y, yhat)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    return test_accuracy

def test_permute(mod, X, y):
    n_test = len(y)
    y = target_to_1d(y)
    yhat = predict_results(mod, X, y)
    test_accuracy = prediction_accuracy(y, yhat)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    return test_accuracy
def _shuffle(a, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(a))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return indices

def _shuffle_by_index(a,indices):
    return np.array(list(a[ind] for ind in indices))

def compile_results(arg, w_j, train_accuracy, ind_subj, y_test, test_accuracy, y_inds=None, perm=np.nan):
    # y test: one value for each trial (not including segments)
    #n_test = len(ind_subj)
    n_test = len(y_test[:,0]) #added on 12/13/22
    if arg.perm: #added on 12/13/22
        perm = arg.ith_perm
    
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [arg.window] * n_test,
                                  [arg.step] * n_test,
                                  [w_j] * n_test, 
                                  [train_accuracy] * n_test, 
                                  ind_subj,
                                  np.arange(n_test),
                                  y_test[:,0],
                                  [perm] * n_test,
                                 )))
    columns = col_names(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=columns)
    return test_df

def compile_results_perm(arg, w_j, train_accuracy, ind_subj, y_test, test_accuracy, y_inds=None,perm=np.nan):
    # y test: one value for each trial (not including segments)
    n_test = len(ind_subj)
    if arg.n_perms:
        test_ind = y_inds
    else:
        test_ind = np.arange(n_test)
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [arg.window] * n_test,
                                  [arg.step] * n_test,
                                  [w_j] * n_test, 
                                  [train_accuracy] * n_test, 
                                  ind_subj,
                                  test_ind,
                                  y_test[:,0],
                                  [perm] * n_test,
                                 )))
    columns = col_names(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=columns)
    return test_df

def col_names(arg):
    """train_subj, clf, w, s, w_j, fld, train_acc, test_subj, 
            test_trial_num, test_trial_val, test_acc{1,..,m}, permutation_number """
    t_windows = np.arange(0, arg.t_end - arg.window, arg.step)
    t_starts = [i + np.str(j) for i, j in zip(['t_'] * len(t_windows), t_windows)]

    df_cols = np.concatenate((["subject",
                               "classifier",
                               "window",
                               "step",
                               "w_train",
                               "train_accuracy",
                               "test_subject",
                               "test_trial_num",
                               "test_trial_val",
                               "permutation"], 
                              t_starts,))
    return df_cols

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
def save_df(test_df, arg, w_j):
    test_df = test_df.apply(pd.to_numeric, errors='ignore')

    fname = (arg.project_path
             + 'data/' 
             + arg.subject 
             + '/results/w'+ np.str(arg.window)
             + '_s'+np.str(arg.step)
             +'/' 
             + arg.out_name 
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    pd_to_csv(fname, test_df) # previously: test_df.to_csv(fname)
    
    group_cols = ['subject', 'classifier', 'window', 'step', 'w_train','test_subject'] 
    drop_cols = ["test_trial_val", "test_trial_num"]
    
    acc_agg = test_df.groupby(by=group_cols, as_index=False).mean()
    acc_agg = acc_agg.drop(columns=drop_cols)
    group_fname = (arg.project_path 
                   + 'data/across_subjects/results/' 
                   + arg.out_name 
                   + '.csv')
    pd_to_csv(group_fname, acc_agg)

def pd_to_csv(fname, df,header=False):
    with FileLock(fname+".lock"):
        if not os.path.exists(fname): 
            header=True
        with open(fname, "a+") as f:  df.to_csv(f, header=header, index=False)

            
#def save_df(test_df, arg, w_j):
    ## test_df['train_accuracy'] = test_df['train_accuracy'].astype(np.float) #test_df.columns[5]
    ## test_df.iloc[:,9:] = test_df.iloc[:,9:].astype(np.float) #['t_0',...]
    ## test_df.iloc[:,2:5] = test_df.iloc[:,2:5].astype(np.int) #['window', 'step', 'w_train']
    ## test_df.iloc[:,7:9] = test_df.iloc[:,7:9].astype(np.int)   #['test_trial_num', 'test_trial_val']

    # test_df = test_df.apply(pd.to_numeric, errors='ignore')

    # fname = (arg.project_path 
    #  + 'data/' 
    #  + arg.subject 
    #  + '/results/w'+ np.str(arg.window)+'_s'+np.str(arg.step)+'/' 
    #  + arg.out_name 
    #  + str(w_j).zfill(3)
    #  + '.csv')
    # fill_path(fname)
    # pd_to_csv(fname, test_df) # previously: test_df.to_csv(fname)
    # by_cols = ['subject', 'classifier', 'window', 'step', 'w_train','test_subject'] 
    # acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    # acc_agg = acc_agg.drop(columns=["test_trial_val", "test_trial_num"])
    # group_fname = arg.project_path + 'data/across_subjects/results/' + arg.out_name + '.csv'
    # pd_to_csv(group_fname, acc_agg)
    #
    ## # previously:
    ## with FileLock(group_fname+".lock"): # added 04/25/22
    ##    header=False
    ##    if not os.path.exists(group_fname): header=True
    ##    with open(group_fname, "a+") as f:  acc_agg.to_csv(f, header=header, index=False)