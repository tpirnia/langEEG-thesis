import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from classification.models import make_model, predict_results, prediction_accuracy, load_fold_params, save_fold_params, make_model_with_params
from classification.permutations import get_permutation_indices_split
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d
from filelock import FileLock

cv_splitter={"groupkfold": GroupKFold, 
             "stratifiedkfold": StratifiedKFold}

def inner_CV(fld, w_j, X, y, groups, trte, 
             inner_splits, 
             classifier, subject, results_out=False, arg=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    n_train, n_test = len(test_ind), len(test_ind)
    
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del(X, y, groups)
    
    # configure the cross-validation procedure
    cv = list(StratifiedKFold(n_splits=inner_splits).split(X_train, 
                                                           y_train, 
                                                           groups_train))
    modelCV = make_model(classifier, cv) 
    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    save_fold_params(modelCV.best_params_, arg, w_j, fld)
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # changes to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy)#, yhat_fold)
    #save_df(test_df, arg, w_j)
    if results_out:
        return test_df, resultCV, y_test, yhat_fold
    else:
        return test_df

def inner_CV_by_rep(fld, w_j, X, y, groups, trte, 
             inner_splits, 
             classifier, subject, splitter, results_out=False, arg=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    n_train, n_test = len(test_ind), len(test_ind)
    
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del(X, y, groups)
    
    # configure the cross-validation procedure
    cv = list(cv_splitter[splitter](n_splits=inner_splits).split(X_train, 
                                                           y_train, 
                                                           groups_train))
    modelCV = make_model(classifier, cv) 
    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    # save model parameters from the fold.
    save_fold_params(modelCV.best_params_, arg, w_j, fld)
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # changes to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy)#, yhat_fold)
    #save_df(test_df, arg, w_j)
    if results_out:
        return test_df, resultCV, y_test, yhat_fold
    else:
        return test_df


def inner_CV_by_rep_permutations(fld, w_j, X, y, groups, trte, classifier,arg=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    _, n_test = len(train_ind), len(test_ind)
    
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del(X, y, groups)
    
    best_params = load_fold_params(arg,w_j,fld) #***
    
    for perm in range(arg.n_perms): # add get last perm later.#***
        indices_train, indices_test = get_permutation_indices_split(
            y_train, groups_train, y_test, groups_test, perm, base_seed=1234
        )
        modelCV = make_model_with_params(classifier, best_params) 
        # fit model, estimate test prediction + accuracy
        resultCV = modelCV.fit(X_train, y_train[indices_train])

        # train accuracy:
        train_accuracy = resultCV.score(X_train, y_train[indices_train])  # best_score_
    
        # test accuracy:
        X_test = tw_to_3d(X_test) # changes to trial*n_winds for classification
        yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test[indices_test]))
        test_accuracy = prediction_accuracy(target_to_1d(y_test[indices_test]), yhat_fold)
        test_accuracy = target_to_2d(test_accuracy, n_test)
        
        test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy)

        # add perm number to test_df and combine with existing df if it exists
        test_df['perm'] = perm
        save_permutations(test_df, arg, w_j)
        if perm % 10 == 0:
            save_progress(perm, arg, w_j, fld)


def save_progress(perm, arg, w_j, fld):
    perm_path = '{}/results/best_params/context_accuracy_{}-{}_w{}'.format(arg.project_path, arg.context, arg.context, arg.window)
    fname = perm_path + f"/{arg.subject}_{arg.classifier}_w{str(w_j).zfill(3)}_fld_perm_progress.csv"
    fill_path(fname)
    pd_to_csv(fname, pd.DataFrame({'fold': fld, 'perm': perm}))

def save_permutations(test_df, arg, w_j):
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    perm_path = '{}/results/best_params/context_accuracy_{}-{}_w{}'.format(arg.project_path, arg.context, arg.context, arg.window)
    fname = perm_path + f"/{arg.subject}_{arg.classifier}_w{str(w_j).zfill(3)}_fld_perm_accuracy.csv"
    fill_path(fname)
    pd_to_csv(fname, test_df) 
    
def across_classify(fld, w_j, X, y, groups, trte, conditions, 
             inner_splits, 
             classifier, subject, splitter, results_out=False, arg=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    n_train, n_test = len(test_ind), len(test_ind)
    
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del(X, y, groups)
    
    # configure the cross-validation procedure
    cv = list(cv_splitter[splitter](n_splits=inner_splits).split(X_train, 
                                                           y_train, 
                                                           groups_train))
    modelCV = make_model(classifier, cv) 
    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # changes to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    
    test_df = compile_inner_acc(arg, fld, trte, w_j, train_accuracy, test_accuracy)
    if results_out:
        return test_df, resultCV, y_test, yhat_fold
    else:
        return test_df

    
def compile_inner_acc(arg, fold, trte, w_j, train_accuracy, test_accuracy):
    n_test = len(trte[fold][1])
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [arg.window] * n_test,
                                  [arg.step] * n_test,
                                  [w_j] * n_test, 
                                  [fold] * n_test, 
                                  [train_accuracy] * n_test, 
                                  trte[fold][1], #y,
                                 )))
    test_col = col_inner_acc(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=test_col)
    return test_df


def col_inner_acc(arg):
    t_windows = np.arange(0, arg.t_end - arg.window, arg.step)
    t_starts = [i + str(j) for i, j in zip(['t_'] * len(t_windows), t_windows)]  #changed np.str to str

    test_col = np.concatenate((["subject",
                                "classifier",
                                "window",
                                "step",
                                "w_train",
                                "fold",
                                "train_accuracy",
                                "test_trial_num"],#,"test_trial_val"], 
                               t_starts))
    return test_col

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))

def save_df(test_df, arg, w_j):
    test_df = test_df.apply(pd.to_numeric, errors='ignore')


    fname = (arg.project_path 
             + 'data/' 
             + arg.subject 
             + '/results/w'+ str(arg.window)+'_s'+ str(arg.step)+'/'  #changed np.str to str
             + arg.out_name 
             + '_'
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    pd_to_csv(fname, test_df) # previously: test_df.to_csv(fname)
    
    group_cols = ['subject', 'classifier', 'window', 'step', 'w_train'] 
    drop_cols = ["fold", "test_trial_num"]
        
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