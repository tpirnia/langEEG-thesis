import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from classification.models import make_model, predict_results, prediction_accuracy
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d
from classification.cross_condition import relabel_svm
###############  
# don't need yet:
# from sklearn.pipeline import make_pipeline
# from mne.decoding import Scaler, cross_val_multiscore, Vectorizer
# from sklearn.metrics import accuracy_score
############### ############### ###############    
############### ############### ###############
cv_splitter={"groupkfold": GroupKFold, 
             "stratifiedkfold": StratifiedKFold,
             "gridsearchcv": GridSearchCV}

def inner_CV_by_rep(fold, w_j, X, y, conditions, groups, trte, 
             inner_splits, 
             classifier, subject, splitter, perm=False, results_out=False, arg=None, rand_gen=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    t_inner = time.time()
    
    ######## Below is new.
    trte, y_cross, y_clf = conditions.relabel_classes(y, trte)
    ###### ABOVE is new
    
    train_ind, test_ind = trte[fold]
    n_train, n_test = len(train_ind), len(test_ind)
    
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    if perm:
        rifperm_tr,rifperm_te = rand_gen.permutation(n_train), rand_gen.permutation(n_test)
    else:
        rifperm_tr, rifperm_te = np.arange(n_train), np.arange(n_test)
    y_train =  np.squeeze(y_clf[train_ind, w_j])[rifperm_tr]
    y_test = y_cross[test_ind,:][rifperm_te,:]
    y_true = y[test_ind,:][rifperm_te,:]
    #print('train_ind:{}\ntest_ind:{}'.format(len(train_ind),len(test_ind)))
    
    groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del(X, groups) #removed y from this line.
    ###############################################################
    
    
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
    
    yhat_fold = target_to_2d(yhat_fold, n_test)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    
    accuracy_df = compile_results(arg, fold, trte, w_j, y_true, conditions, train_accuracy, test_accuracy)
    predict_df = compile_results(arg, fold, trte, w_j, y_true, conditions, train_accuracy, yhat_fold)
    #save_df(test_df, arg, w_j)
    if results_out:
        return accuracy_df, predict_df, resultCV#y_test, yhat_fold
    else:
        return accuracy_df, predict_df

def inner_CV_phase_context(fold, w_j, X, y, y2, conditions, groups, trte, 
             inner_splits, 
             classifier, splitter,  n_jobs=1,  perm=False, arg=None, rand_gen=None, **_):
    """input: 
        - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
        - y: 2 dimensions: [n_trials, n_windows]
        - groups: 2 dimensions: [n_trials, n_windows]
    """
    
    train_ind, test_ind = conditions.filter_trte(fold, y, y2, trte) ### this is specific to across phase-context classification
    n_train, n_test = len(train_ind), len(test_ind)
        
    X_train, X_test = np.squeeze(X[train_ind, w_j,:,:]), X[test_ind]
    if not perm:
        rifperm_tr, rifperm_te = np.arange(n_train), np.arange(n_test)
    else:
        # rifperm_tr,rifperm_te = rand_gen.permutation(n_train), rand_gen.permutation(n_test)
        rifperm_tr,rifperm_te = rand_gen[fold]
        
    y_train_true =  np.squeeze(y[train_ind, w_j])[rifperm_tr]
    y_train = relabel_svm(y_train_true.copy()) # y_train: training, context labels unrelated='-1', phonological='+1'; (all have straight phase y2=-1)

    y_test = y2[test_ind,:][rifperm_te,:] #y_test is the testing, phase label: straight='-1', reverse='+1'
    y_test = relabel_svm(np.copy(y_test))
    y_true = y[test_ind,:][rifperm_te,:] #y_true is the context label of test data: semantic='2'


    groups_train = np.squeeze(groups[train_ind, w_j])
    #print('train_ind:{}\ntest_ind:{}'.format(len(train_ind),len(test_ind)))
    
    del(X, groups) #removed y from this line.
    ###############################################################
    
    
    # configure the cross-validation procedure
    cv = list(cv_splitter[splitter](n_splits=inner_splits).split(X_train, y_train, groups_train))

    modelCV = make_model(classifier, cv, n_jobs=n_jobs) 

    
    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)
    
    # train accuracy:
    train_accuracy = resultCV.best_score_
    
    # test accuracy:
    X_test = tw_to_3d(X_test) # changes to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test)) # dimensions = (trials * windows)
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold) # dimensions = (trials * windows)
    
    yhat_fold = target_to_2d(yhat_fold, n_test) # dimensions = (trials, windows)
    test_accuracy = target_to_2d(test_accuracy, n_test)  #dimensions = (trials, windows)
    
    accuracy_df = compile_results(arg, fold, trte, w_j, y_test, conditions, train_accuracy, test_accuracy, y_alt=y_true)
    predict_df = compile_results(arg, fold, trte, w_j, y_test, conditions, train_accuracy, yhat_fold, y_alt=y_true)
    return accuracy_df, predict_df


def compile_results(arg, fold, trte, w_j, y_true, conditions, train_accuracy, test_accuracy, y_alt=None, perm=np.nan):
    test_ind = trte[fold][1]
    try: 
        n_test = len(test_ind)
    except:
        n_test = test_accuracy.shape[0]
        
    if arg.perm: #added on 12/13/22
        perm = int(arg.ith_perm)
    
    try:
        y_alt = y_alt[:,0]
    except:
        y_alt = [np.nan] * n_test
    
    test_out = list(map(list, zip([arg.subject] * n_test, 
                                  [arg.classifier] * n_test,
                                  [conditions.within['y']]* n_test, #within condition
                                  [conditions.cross['y']]* n_test, #across condition
                                  [arg.window] * n_test, #window width
                                  [arg.step] * n_test, #step size
                                  [w_j] * n_test, #training window
                                  [fold] * n_test, #fold
                                  test_ind,  # trte[fold][1], #index of test trial
                                  y_true[:,0], #correct value of condition (not modified)
                                  y_alt,
                                  [perm] * n_test,
                                  [train_accuracy] * n_test, #train accuracy (over all)
                                 )))
    columns = col_names(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=columns)
    return test_df

def col_names(arg):
    """train_subj, clf, w, s, w_j, fold, train_acc, test_subj, 
            test_trial_num, test_trial_val, test_acc{1,..,m}, permutation_number """
    t_windows = np.arange(0, arg.t_end - arg.window, arg.step)
    t_starts = [i + str(j) for i, j in zip(['t_'] * len(t_windows), t_windows)]

    df_cols = np.concatenate((["subject",
                               "classifier",
                               "within_condition",
                               "cross_condition",
                               "window",
                               "step",
                               "w_train",
                               "fold",
                               "test_trial_num", #index of test trial
                               "test_trial_val",# should put truth here. (conflict #2)
                               "test_trial_alt_conflict_val", #value of conflict#1 in the test set
                               "permutation",
                               "train_accuracy"], #should be nan if not permutation
                              t_starts,))
    return df_cols
