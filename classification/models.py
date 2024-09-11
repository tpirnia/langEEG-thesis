import numpy as np


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


def predict_results(estimator, X, y):
    yhat = estimator.best_estimator_.predict(X)
    return yhat #accuracy_score(y, yhat)

def prediction_accuracy(y, yhat):
    acc = np.zeros(y.shape)
    acc[yhat==y]=1
    return acc

Parameters = {'decisiontree': {'decisiontreeclassifier__criterion':['gini','entropy'],
                               'decisiontreeclassifier__min_samples_split': np.arange(1, 10).tolist(),
                               'decisiontreeclassifier__min_samples_leaf': np.arange(1, 5).tolist(),
                              },
              "ridge": {'ridgeclassifier__alpha':[10., 1.,0.1,0.01,0.001, 0.0001,
                                                  0.00001,0.000001],
                       },
              "svm": {'svc__gamma':np.logspace(-9, 3, 13), 
                      'svc__C':np.logspace(-2, 10, 13),
                     },
              "svm_linear": {'svc__gamma':np.logspace(-9, 3, 13), 
                      'svc__C':np.logspace(-2, 10, 13),
                     },
              "log":{"logisticregression__C": np.logspace(-1, 1, 3)},
             }

Classifiers = {"decisiontree" : DecisionTreeClassifier(random_state=42),
               "ridge" : RidgeClassifier(),
               "svm" : SVC(),
               "svm_linear" : SVC(kernel='linear'),
               "log": LogisticRegression(max_iter=300),
              }

Scoring = {"decisiontree": {"AUC": "roc_auc",
                            "accuracy": make_scorer(accuracy_score),
                           },
           "ridge": "accuracy",
           "svm": "accuracy",
           "svm_linear": "accuracy",
           "log": "accuracy",
          }

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)