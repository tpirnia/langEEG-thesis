import sys, os, copy
import numpy as np
from functools import reduce

import numpy as np

def _ensure_1d(arr):
    """Ensure array is 1D by taking first column if 2D."""
    return arr[:, 0] if arr.ndim == 2 else arr


class conflictConditions(): #change later to crossContext
    def __init__(self, cross_condition, y):
        within_condition = set(y).difference([0, cross_condition]).pop()
        
        self.within = {'y': within_condition, 'y_clf': 1}
        self.cross = {'y': cross_condition, 'y_clf': 2}
        self.lc = {'y': 0, 'y_clf': -1}

    def __iter__(self):
        for key in self.__dict__.keys():
            yield (key, self.__dict__[key])

    def __str__(self):
        clf_labels = f"high-conflict:\n\twithin:\ty = {self.within['y']}\ty_clf = {self.within['y_clf']}\n\tcross:\ty = {self.cross['y']}\ty_clf = {self.cross['y_clf']}\n\nlow-conflict:\ty = { self.lc['y']}\ty_clf = {self.lc['y_clf']}"
        return clf_labels
    
    def _add_loc(self, y):
        for key in self.__dict__.keys():
            if 'y' in self.__dict__[key]:
                ky = self.__dict__[key]['y']
                loc = np.where(y==ky)[0]
                self.__dict__[key].update({'loc':loc})
                self.__dict__[key].update({'count':len(loc)})

    def get_fold_labels(self, y, groups, fold_idx, perm_indices=None, test_folds=[2, 1, 0]):
        """Get train/test labels for a specific fold, optionally permuted.
        
        Returns raw labels before relabeling - use with relabel_classes() workflow.
        """
        y_1d = _ensure_1d(y)
        groups_1d = _ensure_1d(groups)
        test_fold = test_folds[fold_idx]
        
        # Create filtered datasets (same logic as your original)
        within_mask = y_1d != self.cross['y']
        cross_mask = y_1d != self.within['y']
        
        y_within = y[within_mask]
        y_cross = y[cross_mask]
        groups_within = groups_1d[within_mask]
        groups_cross = groups_1d[cross_mask]
        
        # Apply permutation if provided
        if perm_indices is not None:
            y_within = y_within[perm_indices]
            y_cross = y_cross[perm_indices]
        
        # Get train/test splits
        train_labels = y_within[groups_within != test_fold]
        test_labels = y_cross[groups_cross == test_fold]
        
        return train_labels, test_labels

    def get_fold_indices(self, y, groups, fold_idx, test_folds=[2, 1, 0]):
        """Get original indices for train/test trials in a fold.
        
        Use this to index into X for getting train/test data.
        """
        y_1d = _ensure_1d(y)
        groups_1d = _ensure_1d(groups)
        test_fold = test_folds[fold_idx]
        
        # Get original indices
        within_indices = np.where(y_1d != self.cross['y'])[0]
        cross_indices = np.where(y_1d != self.within['y'])[0]
        
        # Filter by fold
        within_groups = groups_1d[within_indices]
        cross_groups = groups_1d[cross_indices]
        
        train_idx = within_indices[within_groups != test_fold]
        test_idx = cross_indices[cross_groups == test_fold]
        
        return train_idx, test_idx

    def relabel_for_classifier(self, y_, clf_type='default'):
        y_1d = _ensure_1d(y_)
        if len(set(y_1d)) == 3:
            print('Relabeling both high conflict states to the same label value.')
        y_relabeled = y_.copy()

        is_lc = (y_1d == min(y_1d))

        if clf_type == 'svm':
            y_relabeled[is_lc], y_relabeled[~is_lc] = -1, 1
        elif clf_type == 'log':
            y_relabeled[is_lc], y_relabeled[~is_lc] = 0, 1
        else:  # default
            y_relabeled[is_lc] = self.lc['y_clf']
            y_relabeled[~is_lc] = self.within['y_clf']

        return y_relabeled
    
    def get_fold_labels_relabeled(self, y, groups, fold_idx, test_folds=[2, 1, 0], perm_indices=None,clf_type='default'):
        """
        Get train/test labels for a fold, optionally permuted and relabeled for classifier.
        """
        train_labels, test_labels = self.get_fold_labels(y, groups, fold_idx, perm_indices, test_folds=test_folds)
        train_labels = self.relabel_for_classifier(train_labels, clf_type=clf_type)
        test_labels = self.relabel_for_classifier(test_labels, clf_type=clf_type)
        return train_labels, test_labels


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
