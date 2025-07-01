import sys, os, copy
import numpy as np
from functools import reduce

class conflictConditions(): #change later to crossContext
    def __init__(self,  cross_condition, y):
        within_condition = set(y).difference([0,cross_condition]).pop()

        self.within = {'y':  within_condition,
                       'y_clf': 1,
                      }

        self.cross = {'y': cross_condition,
                      'y_clf': 2, 
                     }

        self.lc = {'y': 0,
                   'y_clf': -1, #if I do log, it needs to be changed to 0
                  }

    def __iter__(self):
        for key in self.__dict__.keys():
            yield (key, self.__dict__[key])

    def __str__(self):
        clf_labels = "high-conflict:\n\twithin:\ty = {}\ty_clf = {}\n\tcross:\ty = {}\ty_clf = {}\n\nlow-conflict:\ty = {}\ty_clf = {}".format(self.within['y'],
                                                                                                                         self.within['y_clf'],
                                                                                                                         self.cross['y'],
                                                                                                                         self.cross['y_clf'],
                                                                                                                         self.lc['y'],
                                                                                                                         self.lc['y_clf'])
        return clf_labels
    
    def _add_loc(self, y):
        for key in self.__dict__.keys():#items():
            if 'y' in self.__dict__[key]:
                ky = self.__dict__[key]['y']
                loc = np.where(y==ky)[0]
                self.__dict__[key].update({'loc':loc})
                self.__dict__[key].update({'count':len(loc)})

    def _get_ylabels(self, y):
        y_tmp = copy.deepcopy(y)
        y_tmp[self.within['loc']] = self.within['y_clf']
        y_tmp[self.cross['loc']] = self.cross['y_clf']
        y_tmp[self.lc['loc']] = self.lc['y_clf']
        print('get_ylabels({}) -> {}.'.format(np.unique(y),np.unique(y_tmp)))
        return y_tmp

    def _relabel(self, y_clf):
        within_y = self.within['y_clf']
        y_tmp = copy.deepcopy(y_clf)
        y_tmp[self.cross['loc']] = within_y
        y_tmp[self.within['loc']] = within_y
        print('cross_relabel({}) -> {}'.format(np.unique(y_clf),np.unique(y_tmp)))
        return y_tmp

    def cross_split(self, y, trte):
        """
        Run after ._add_loc(y)
        """
        tr_tmp,te = zip(*trte)
        tr = list(tr_f[np.isin(tr_f, self.cross['loc'], invert=True)] for tr_f in tr_tmp)
        tr_cross= list(tr_f[np.isin(tr_f, self.cross['loc'])] for tr_f in tr_tmp)

        trte = list(zip(tr,te))
        self.cross['excluded_tr'] = tr_cross
        return trte
    
    def relabel_classes(self, y, trte):
        self._add_loc(y)
        y_class = self._get_ylabels(y)
        y_cross = self._relabel(y_class)
        splits = self.cross_split(y, trte)
        
        return splits, y_cross, y_class

    def permute_relabel_classes(self, y, trte):
        self._add_loc(y)
        y_class = self._get_ylabels(y)
        y_cross = self._relabel(y_class)
        splits = self.cross_split(y, trte)
        return splits, y_cross, y_class
    
def filter_train_test_2y(fold, y, y2, trte_index):
    y2 = y2[:, 0] if y2.ndim > 1 else y2
    y = y[:, 0] if y.ndim > 1 else y

    train_filt = reduce(np.intersect1d, (trte_index[fold][0], np.where(y != 2)[0], np.where(y2 == -1)[0]))
    test_filt = np.intersect1d(trte_index[fold][1], np.where(y == 2)[0])

    return train_filt, test_filt

def relabel_svm(y):
    if (y==0).any():
        y[y==0] = -1
    if (y>1).any():
        y[y>1] = 1
    return y


conditions_set = {
    "phonological": [1, 0, 2],
    "semantic": [2, 0, 1],
    "phase": [3, -1],
}

suffix = {
    0: "U",
    1: "UP", #was UR
    2: "US",
    3: "NP",
    -1: "N",
}


class crossConflict:
    def __init__(self, within_condition, across_condition):
        WITHIN = conditions_set[within_condition]
        CROSS = conditions_set[across_condition]
        self.within = {
            "y": WITHIN[0],
            "y_clf": 1,
            "lc": WITHIN[1],
            "alt_conflict": CROSS[-1],
            "dropped": WITHIN[2] if len(WITHIN) == 3 else np.nan,
            "labels": WITHIN,
        }
        self.cross = {
            "y": CROSS[0],
            "y_clf": 1,
            "lc": CROSS[1],
            "alt_conflict": WITHIN[-1],
            "dropped": CROSS[2] if len(CROSS) == 3 else np.nan,
            "labels": CROSS,
        }
        self.suff = suffix[WITHIN[0]] + "-" + suffix[CROSS[0]][-1]
    
    def filter_trte(self, fold, y, y2, trte_index):
        y2 = y2[:, 0] if y2.ndim > 1 else y2
        y = y[:, 0] if y.ndim > 1 else y

        y_test =  self.within["dropped"] # was 2
        y2_train = self.within["alt_conflict"] #was -1

        train_filt = reduce(np.intersect1d, (trte_index[fold][0], np.where(y != y_test)[0], np.where(y2 == y2_train)[0]))
        test_filt = np.intersect1d(trte_index[fold][1], np.where(y == y_test)[0])

        return train_filt, test_filt
