import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from classification.models import make_model, predict_results, prediction_accuracy, save_fold_params, make_model_with_params, load_fold_params
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d

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
        X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind] 
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
    save_fold_params(modelCV.best_params_, arg, w_j, fld, ch=ch)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # change to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    
    test_accuracy_0 = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy_0, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy, ch=ch)
    return test_df

def compile_inner_acc(arg, fold, trte, w_j, train_accuracy, test_accuracy, ch=None):
    n_test = len(trte[fold][1])
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [arg.window] * n_test,
                                  [arg.step] * n_test,
                                  [w_j] * n_test, 
                                  [ch] * n_test,
                                  [fold] * n_test, 
                                  [train_accuracy] * n_test, 
                                  trte[fold][1],
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
                                "test_trial_num"],
                               t_starts))
    return test_col



def permute_CV_ch(fld, w_j, X, y, groups, trte, 
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
        X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind] 
        ch=None
    else: 
        print("X dimensions are neither 3 nor 4 (dims={})".format(X.ndim))
        pass
    
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]

    # Get parameters from model
    best_params = load_fold_params(arg,w_j,fld,ch)
    modelCV = make_model_with_params(classifier, best_params) 
    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.score(X_train, y_train)
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # change to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    
    test_accuracy_0 = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy_0, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy, ch=ch)
    return test_df