import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,StratifiedKFold
from classification.models import (
    make_model,
    predict_results,
    prediction_accuracy,
    load_fold_params,
    save_fold_params,
    make_model_with_params,
)
from data_handling.segment import tw_to_3d, target_to_2d, target_to_1d

cv_splitter = {"groupkfold": GroupKFold, "stratifiedkfold": StratifiedKFold}


def inner_CV(
    fld,
    w_j,
    X,
    y,
    groups,
    trte,
    inner_splits,
    classifier,
    subject,
    suffix,
    arg=None,
    **_
):
    """input:
    - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
    - y: 2 dimensions: [n_trials, n_windows]
    - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    _, n_test = len(test_ind), len(test_ind)
    X_train, X_test = np.squeeze(X[train_ind, w_j, :, :]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    groups_train, _ = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del (X, y, groups)

    if splitter is None:
        splitter = "GroupKFold"

    # configure the cross-validation procedure
    cv = list(
        cv_splitter[splitter](n_splits=inner_splits).split(
            X_train, y_train, groups_train
        )
    )

    modelCV = make_model(classifier, cv)

    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)

    # train accuracy:
    train_accuracy = resultCV.best_score_
    # save model parameters from the fold.
    save_fold_params(modelCV.best_params_, arg, w_j, fld)

    # test accuracy:
    X_test = tw_to_3d(X_test)  # changes to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy, n_test)

    test_df = compile_inner_acc(
        arg, fld, trte, w_j, train_accuracy, test_accuracy
    )  # , yhat_fold)
    return test_df


def classify_permute(
    fld,
    w_j,
    X,
    y,
    groups,
    trte,
    inner_splits,
    classifier,
    subject,
    suffix,
    arg=None,
    **_
):
    """input:
    - X: 4 dimensions: [n_trials, n_windows, n_channels, n_timepoints]
    - y: 2 dimensions: [n_trials, n_windows]
    - groups: 2 dimensions: [n_trials, n_windows]
    """
    train_ind, test_ind = trte[fld]
    n_test = len(test_ind)

    X_train, X_test = np.squeeze(X[train_ind, w_j, :, :]), X[test_ind]
    y_train, y_test = np.squeeze(y[train_ind, w_j]), y[test_ind]
    # groups_train, groups_test = np.squeeze(groups[train_ind, w_j]), groups[test_ind]
    del (X, y, groups)
    # configure the cross-validation procedure

    # load model parameters
    best_params = load_fold_params(arg, w_j, fld)
    modelCV = make_model_with_params(classifier, best_params)

    # fit model, estimate test prediction + accuracy
    resultCV = modelCV.fit(X_train, y_train)

    # train accuracy:
    train_accuracy = resultCV.score(X_train, y_train)

    # test accuracy:
    X_test = tw_to_3d(X_test)  # change to trial*n_winds for classification
    yhat_fold = predict_results(resultCV, X_test, target_to_1d(y_test))
    test_accuracy = prediction_accuracy(target_to_1d(y_test), yhat_fold)
    test_accuracy = target_to_2d(test_accuracy, n_test)
    test_df = compile_inner_acc(
        arg, fld, trte, w_j, train_accuracy, test_accuracy
    )  # , yhat_fold)
    return test_df


def compile_inner_acc(arg, fold, trte, w_j, train_accuracy, test_accuracy):
    n_test = len(trte[fold][1])
    test_out = list(
        map(
            list,
            zip(
                [arg.subject] * n_test,
                [arg.classifier] * n_test,
                [arg.window] * n_test,
                [arg.step] * n_test,
                [w_j] * n_test,
                [fold] * n_test,
                [train_accuracy] * n_test,
                trte[fold][1],  # y,
            ),
        )
    )
    test_col = col_inner_acc(arg)
    test_df = pd.DataFrame(np.hstack((test_out, test_accuracy)), columns=test_col)
    return test_df


def col_inner_acc(arg):
    t_windows = np.arange(0, arg.t_end - arg.window, arg.step)
    t_starts = [
        i + str(j) for i, j in zip(["t_"] * len(t_windows), t_windows)
    ]  # changed np.str to str

    test_col = np.concatenate(
        (
            [
                "subject",
                "classifier",
                "window",
                "step",
                "w_train",
                "fold",
                "train_accuracy",
                "test_trial_num",
            ],  # ,"test_trial_val"],
            t_starts,
        )
    )
    return test_col
