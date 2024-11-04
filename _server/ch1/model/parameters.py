import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer

Parameters = {'decisiontree': {'decisiontreeclassifier__criterion':['gini','entropy'],
                               'decisiontreeclassifier__min_samples_split': np.arange(2, 10).tolist(),
                               'decisiontreeclassifier__min_samples_leaf': np.arange(1, 5).tolist(),
                              },
              "ridge": {'ridgeclassifier__alpha':[10., 1.,0.1,0.01,0.001, 0.0001],
                       },
              "svm": {'svc__gamma':np.logspace(-9, 3, 13), 
                      'svc__C':np.logspace(-2, 10, 13),
                     },
              "log":{'C':[1, 0.1, 0.01,0.001]},
             }

Classifiers = {"decisiontree" : DecisionTreeClassifier(random_state=42),
               "ridge" : RidgeClassifier(),
               "svm" : SVC(),
               "log": LogisticRegression(),
              }



Scoring = {"decisiontree": make_scorer(accuracy_score),
           "ridge": "accuracy",
           "svm": "accuracy",
           "log": "accuracy",
          }

# Scoring = {"decisiontree": {"AUC": "roc_auc",
#                             "accuracy": make_scorer(accuracy_score),
#                            },
#            "ridge": "accuracy",
#            "svm": "accuracy",
#           }
# Subject_CV = {"subject":[],
#               "test_acc":[],
#               "train_acc":[],
#               "best_param":[],
#               "classifier":[],
#               "trial_len":[],
#               "suffix":[],
#               "time":[],
#              }