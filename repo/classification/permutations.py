import numpy as np
import pandas as pd
import pickle as pkl
import os

def _shuffle_split(y_train, groups_train, y_test, groups_test, random_state):
    """Returns shuffled indices for both train and test sets"""

    def _get_indices_for_set(y, groups, random_state):
        """get shuffled indices for one set"""
        idx = np.arange(len(y))
        if groups is None:
            return random_state.permutation(idx)
        # Use first column for grouping if 2D, otherwise use as-is
        group_col = groups if groups.ndim == 1 else groups[:, 0]
        for group in np.unique(group_col):
            group_mask = group_col == group
            idx[group_mask] = random_state.permutation(idx[group_mask])
        return idx
    
    # Get shuffled indices for train set
    idx_train = _get_indices_for_set(y_train, groups_train, random_state)
    # Get shuffled indices for test set  
    idx_test = _get_indices_for_set(y_test, groups_test, random_state)
    return idx_train, idx_test

def get_permutation_indices_split(y_train, groups_train, y_test, groups_test, perm_id, seed=1234):
    """Generate reproducible permutation indices for split train and test data."""
    perm_seed = seed + perm_id
    random_state = np.random.RandomState(perm_seed)
    return _shuffle_split(y_train, groups_train, y_test, groups_test, random_state)


def _shuffle(a, groups, random_state):
    """Return shuffled indices, preserving groups."""
    idx = np.arange(len(a))
    if groups is None:
        return random_state.permutation(idx)
    # Use first column for grouping if 2D, otherwise use as-is
    group_col = groups if groups.ndim == 1 else groups[:, 0]
    for group in np.unique(group_col):
        group_mask = group_col == group
        idx[group_mask] = random_state.permutation(idx[group_mask])
    return idx

def get_permutation_indices(y, groups, perm_id, seed=1234):
    """Generate unique permutations for given perm_id."""
    # Each permutation gets a seed.
    perm_seed = seed + perm_id
    random_state = np.random.RandomState(perm_seed)
    return _shuffle(y, groups, random_state)


def generate_all_permutations(y=None, groups=None,
                              y_train=None, groups_train=None, 
                              y_test=None, groups_test=None, 
                              n_perms=1000, seed=1234):
    """Generate all permutation indices for entire set of labels (y,groups) or labels split into train and test sets (y_train, groups_train, y_test, groups_test)."""
    
    # handling whether to do one set of indices or split them. 
    is_split = y_train is not None and y_test is not None
    
    if is_split and (y_train is None or y_test is None):
        raise ValueError("Both y_train and y_test must be provided.")
    if not is_split and y is None:
        raise ValueError("y must be provided.")
    
    permutations = {}
    for perm_id in range(n_perms):
        if is_split:
            train_idx, test_idx = get_permutation_indices_split(
                y_train, groups_train, y_test, groups_test, perm_id, seed)
            permutations[perm_id] = {'train': train_idx, 'test': test_idx}
        else:
            idx = get_permutation_indices(y, groups, perm_id, seed)
            permutations[perm_id] = {'indices': idx}
    return {
        'permutations': permutations,
        'is_split': is_split,
        'n_perms': n_perms,
        'base_seed': seed
    }

def save_permutations(perm_set, filepath):
    """Save permutation data to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pkl.dump(perm_set, f)

def load_permutations(filepath):
    """Load permutation data from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pkl.load(f)

def get_ith_permutation(perm_set, perm_id):
    """Indices for the ith permutation."""
    perm = perm_set['permutations'][perm_id]
    
    if perm_set['is_split']:
        return perm['train'], perm['test']
    else:
        return perm['indices']

def get_permutation_filepath(project_path, subject, file_stem):
    """Standardized perm naming."""
    return os.path.join(project_path, 'data', subject, "permutations",file_stem+'.pkl')


def save_perm_progress(file_path, subject, perms_completed, n_perms):
    os.makedirs(file_path, exist_ok=True)
    csv_file = os.path.join(file_path, "permutation_progress.csv")

    # Load existing record or create new
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if subject in df['subject'].values:
            df.loc[df['subject'] == subject, ['permutation_progress', 'permutation_total']] = [perms_completed, n_perms]
        else:
            df = pd.concat([df, 
                            pd.DataFrame({'subject': [subject], 
                                              'permutation_progress': [perms_completed], 
                                              'permutation_total': [n_perms]})], 
                           ignore_index=True)
    else:
        df = pd.DataFrame({'subject': [subject], 
                           'permutation_progress': [perms_completed], 
                           'permutation_total': [n_perms]})

    df.to_csv(csv_file, index=False)


    
def get_last_perm(file_path, subject):
    """Get the last completed permutation for a subject."""
    csv_file = os.path.join(file_path, "permutation_progress.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if subject in df['subject'].values:
            return df.loc[df['subject'] == subject, 'permutation_progress'].iloc[0]
    else: print(f"No permutation_progress.csv file found at {os.path.dirname(csv_file)}")
    return 0  # Return 0 if no progress found


def save_perm_progress(file_path, subject, perms_completed, n_perms):
    os.makedirs(file_path, exist_ok=True)
    csv_file = os.path.join(file_path, "permutation_progress.csv")

    # Load existing record or create new
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if subject in df['subject'].values:
            df.loc[df['subject'] == subject, ['permutation_progress', 'permutation_total']] = [perms_completed, n_perms]
        else:
            df = pd.concat([df, 
                            pd.DataFrame({'subject': [subject], 
                                              'permutation_progress': [perms_completed], 
                                              'permutation_total': [n_perms]})], 
                           ignore_index=True)
    else:
        df = pd.DataFrame({'subject': [subject], 
                           'permutation_progress': [perms_completed], 
                           'permutation_total': [n_perms]})

    df.to_csv(csv_file, index=False)