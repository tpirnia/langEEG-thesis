import numpy as np

import pickle,os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, Vectorizer

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer



def make_model(classifier, cv_inner, n_jobs=1):
        pipe = get_pipeline(classifier)

        mod = GridSearchCV(pipe, 
                              Parameters[classifier], 
                              scoring=Scoring[classifier], 
                              cv=cv_inner, 
                              refit='accuracy', 
                              n_jobs=n_jobs,
                          )
        return mod

def get_pipeline(classifier):
    clf = make_pipeline(Scaler(scalings='mean'),
                        Vectorizer(),
                        Classifiers[classifier])
    return clf


def _build_param_path(arg):
    """Helper function to build parameter path."""
    path_parts = [arg.project_path, 'results', 'best_params', arg.fstem]
    if 'w50' in arg.fstem or 'channel' in arg.fstem:
        path_parts.append(arg.subject)
    return os.path.join(*path_parts)

def _build_filename(arg, w_j, fld, ch=None):
    """Helper function to build parameter filename."""
    ch_part = f"_ch{ch}" if ch is not None else ""
    return f"{arg.subject}_{arg.classifier}_w{w_j}{ch_part}_fold{fld}_params.pkl"

def save_fold_params(best_params, arg, w_j, fld, ch=None):                   
    """Save best parameters for a specific outer fold."""     
    param_path = _build_param_path(arg)
    os.makedirs(param_path, exist_ok=True)
    
    filename = _build_filename(arg, w_j, fld, ch)
    filepath = os.path.join(param_path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(best_params, f)

def load_fold_params(arg, w_j, fld, ch=None):
    """Load best parameters for a specific outer fold."""        
    param_path = _build_param_path(arg)
    filename = _build_filename(arg, w_j, fld, ch)
    filepath = os.path.join(param_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Best parameters not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def make_model_with_params(classifier, best_params=None):
    """Create model with fixed parameters [no grid search!]."""
    pipe = get_pipeline(classifier)
    if best_params is not None:
        pipe.set_params(**best_params)
    return pipe

def predict_results(estimator, X, y):
    # yhat = estimator.best_estimator_.predict(X)
    # return yhat #accuracy_score(y, yhat)
    predictor = getattr(estimator, 'best_estimator_', estimator) # to handle both when not a gridsearch
    return predictor.predict(X)

def prediction_accuracy(y, yhat):
    acc = np.zeros(y.shape)
    acc[yhat==y]=1
    return acc

def adjust_y(y, classifier):
    """Implement appropriate labels for each classifier."""
    classifiers_needing_neg_one = ['ridge', 'svm', 'svm_linear']
    if classifier in classifiers_needing_neg_one and np.array_equal(np.unique(y), [0,1]):
        y[y==0] = -1
    elif classifier == 'log' and np.array_equal(np.unique(y), [-1, 0, 1]):
        y[y==-1] = 2
    # decision tree's can handle either so we leave it unchanged.
    return y

Parameters = {'decisiontree': {'decisiontreeclassifier__criterion':['gini','entropy'],
                               # 'decisiontreeclassifier__min_samples_split': np.arange(1, 10).tolist(),
                               # 'decisiontreeclassifier__min_samples_leaf': np.arange(1, 5).tolist(),
                              },
              "ridge": {'ridgeclassifier__alpha': [10., 1.,0.1,0.01,0.001, 0.0001,0.00001,0.000001],
                       },
              "svm": {'svc__gamma': np.logspace(-9, 3, 13), 
                      'svc__C': np.logspace(-2, 10, 13),
                     },
              "svm_linear": {'svc__gamma': [10., 1.,0.1,0.01,0.001, 0.0001,0.00001,0.000001],#np.logspace(-9, 3, 13), 
                      'svc__C':[10., 1.,0.1,0.01,0.001, 0.0001,0.00001,0.000001], #np.logspace(-2, 10, 13),
                     },
              "log":{"logisticregression__C": np.logspace(-1, 1, 3)},
             }

Classifiers = {"decisiontree" : DecisionTreeClassifier(random_state=42),
               "ridge" : RidgeClassifier(),
               "svm" : SVC(),
               "svm_linear" : SVC(kernel='linear'),
               "log": LogisticRegression(max_iter=300),
              }

Scoring = {"decisiontree": "accuracy",# {"AUC": "roc_auc", "accuracy": make_scorer(accuracy_score),},
           "ridge": "accuracy",
           "svm": "accuracy",
           "svm_linear": "accuracy",
           "log": "accuracy",
          }

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)


