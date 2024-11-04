import sys, os, time
import numpy as np
import pandas as pd
from seglearn.transform import Segment
from sklearn.model_selection import GridSearchCV, GroupKFold,StratifiedKFold
from classification.models import make_model, predict_results, prediction_accuracy
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d
from filelock import FileLock # added 04/25/22
# missing: make_model, predict_results, tw_to_
###############  
# don't need yet:
# from sklearn.pipeline import make_pipeline
# from mne.decoding import Scaler, cross_val_multiscore, Vectorizer
# from sklearn.metrics import accuracy_score
############### ############### ###############    
############### ############### ###############

def inner_CV_ch(fld, w_j, X, y, groups, trte, 
             inner_splits, 
             classifier, subject, suffix, ch=None, arg=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    t_inner = time.time()
    train_ind, test_ind = trte[fld]
    n_train, n_test = len(test_ind), len(test_ind)
    
    if X.ndim==3:
        X_train,X_test = np.squeeze(X[train_ind, w_j,:])[:,np.newaxis,:], X[test_ind,:,np.newaxis,:]
    elif X.ndim==4:
        X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind] #original line 30
        ch=None
    else: 
        print("X dimensions are neither 3 nor 4 (dims={})".format(X.ndim))
        pass
    
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]

    # configure the cross-validation procedure    
    cv = list(GroupKFold(n_splits=inner_splits).split(X_train, y_train, groups_train))
    modelCV = make_model(classifier, cv) 
    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # change to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    
    test_accuracy_0 = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy_0, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy, ch=ch)#, yhat_fold)
    return test_df


def compile_inner_acc(arg, fold, trte, w_j, train_accuracy, test_accuracy,ch=None):
    n_test = len(trte[fold][1])
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [arg.window] * n_test,
                                  [arg.step] * n_test,
                                  [w_j] * n_test, 
                                  [ch] * n_test,
                                  [fold] * n_test, 
                                  [train_accuracy] * n_test, 
                                  trte[fold][1], #y,
                                 )))
    test_col = col_inner_acc(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=test_col)
    return test_df


def col_inner_acc(arg):
    t_windows = np.arange(0, arg.t_end - arg.window, arg.step)
    t_starts = [i + np.str(j) for i, j in zip(['t_'] * len(t_windows), t_windows)]

    test_col = np.concatenate((["subject",
                                "classifier",
                                "window",
                                "step",
                                "w_train",
                                "channel",
                                "fold",
                                "train_accuracy",
                                "test_trial_num"],#,"test_trial_val"], 
                               t_starts))
    return test_col

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def save_df(test_df, arg, w_j):
    #test_df = pd.concat(sts,ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')


    fname = (arg.project_path 
             + 'data/' 
             + arg.subject 
             + '/results/w'+ np.str(arg.window)+'_s'+np.str(arg.step)+'/' 
             + arg.out_name 
             + '_'
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    pd_to_csv(fname, test_df) # previously: test_df.to_csv(fname)
    
    group_cols = ['subject', 'classifier', 'window', 'step', 'w_train'] #list(test_df.columns.values[:5])
    drop_cols = ["fold", "test_trial_num"]
    print(group_cols)
        
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