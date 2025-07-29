import numpy as np

def _ensure_1d(arr):
    """Ensure array is 1D by taking first column if 2D."""
    return arr[:, 0] if arr.ndim == 2 else arr

class conflictConditions:
    def __init__(self, cross_condition, y):
        within_condition = set(y).difference([0, cross_condition]).pop()
        
        self.within = {'y': within_condition, 'y_clf': 1}
        self.cross = {'y': cross_condition, 'y_clf': 2}
        self.lc = {'y': 0, 'y_clf': -1}
    
    def get_labels(self, y, groups, fold_idx, perm_indices=None, test_folds=[2, 1, 0]):
        """Get train/test labels for a fold, optionally permuted.
        
        Args:
            y: labels array (1D or 2D)
            groups: groups array (1D or 2D) 
            fold_idx: which fold (0, 1, 2)
            perm_indices: permutation indices (None for true labels)
            test_folds: test fold order
            
        Returns:
            train_labels, test_labels
        """
        y_1d = _ensure_1d(y)
        groups_1d = _ensure_1d(groups)
        test_fold = test_folds[fold_idx]
        
        # Get filtered datasets
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
    
    def get_true_labels(self, y, groups, fold_idx, test_folds=[2, 1, 0]):
        """Get true (unpermuted) train/test labels."""
        return self.get_labels(y, groups, fold_idx, perm_indices=None, test_folds=test_folds)
    
    def get_permuted_labels(self, y, groups, fold_idx, perm_indices, test_folds=[2, 1, 0]):
        """Get permuted train/test labels."""
        return self.get_labels(y, groups, fold_idx, perm_indices, test_folds=test_folds)

# Usage example:
if __name__ == "__main__":
    # Example usage
    y = np.repeat([0, 1, 2], 60)
    groups = np.tile([1, 2, 3], 60)
    
    # Initialize
    cc = conflictConditions(cross_condition=2, y=y)
    
    # Get true labels for fold 1
    train_true, test_true = cc.get_true_labels(y, groups, fold_idx=1)
    
    # Get permuted labels for fold 1
    perm_indices = np.random.permutation(120)  # Example permutation
    train_perm, test_perm = cc.get_permuted_labels(y, groups, fold_idx=1, perm_indices=perm_indices)
    
    print(f"True train shape: {train_true.shape}")
    print(f"Permuted train shape: {train_perm.shape}")
    print(f"True test shape: {test_true.shape}")
    print(f"Permuted test shape: {test_perm.shape}")
