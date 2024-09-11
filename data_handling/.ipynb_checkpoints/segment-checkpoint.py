import numpy as np
from seglearn.transform import Segment


# class sliding_window():

def tw_to_3d(X):
    """ convert 4D segmented X to 3D segment:
        input: 
            X : [n_trials, n_windows, len_dim2, len_dim3] 
        returns: 
            X_3d : [n_trials *  n_windows, len_dim2, len_dim3] 
    """
    if X.ndim==3:
        return X
    else:
        X_3d = np.reshape(X, (-1, X.shape[-2],  X.shape[-1]), order='C')
        return X_3d


def tw_to_4d(X, n_trials):
    """ convert 3D segmented X to 4D segment:
        input: 
            X: [n_trials * n_windows, len_dim2, len_dim3] 
        returns:
            X_4d: [n_trials, n_windows, len_dim2, len_dim3]
        # come back to editing the following dimension descriptions:
        eq(EEG[trial, ch, w_ind * s: (w_ind * s) + w], X_4d[trial, w_ind, ch, w])
    """
    X_4d = np.reshape(X, (n_trials, -1, X.shape[-2],  X.shape[-1]), order='C')
    return X_4d


def target_to_2d(target, n_trials):
    return np.reshape(target, (n_trials, -1)).squeeze()

def target_to_1d(target):
    return target.flatten()


def segment_ts(X, y, groups, window, step):
    """ Description: Takes classification data as input and returns all  segmented windows with step size.
        Resulting output are in [trial * num windows,] format.
        input:
            - X: [trial, channel, time]
            - y: [trial,] (-1, +1)
            - groups: [trial,]
            - window: (int)
            - step: (int)
        function:
            - 1 - initialize Segment from seglearn
            - 2 - swap channel (1 -> 2) and time (2 -> 1) axes
            - 3 - segment X, y, groups
            - 4 - swap channel (2 -> 1) and time (1 -> 2) axes back
        output:
            - X: [trial *  w_ind, channel, time]
            - y: [trial * w_ind,]
            - groups : [trial * w_ind,]
    """
    segmenter = Segment(width=window, step=step, order='F')
    X = X.swapaxes(1,2) # switch to [trials, time, channels] <- 3D
    X, y, groups = segmenter.fit_transform(X, y, groups)
    return X.swapaxes(1,2), y, groups

def order_by_trials(X, y, groups, n_trials):
    """ changes output of segment_ts to split axis=0 into [trials, windows].
        arguments:
                X:      [n_trials * n_windows, n_channels, w]
                y:      [n_trials * n_windows]
                groups: [n_trials * n_windows]
        return:
                X:      [n_trials, n_windows, n_channels, w]
                y:      [n_trials, n_windows]
                groups: [n_trials, n_windows]
    """
    X = tw_to_4d(X, n_trials)
    y = target_to_2d(y, n_trials)
    groups = target_to_2d(groups, n_trials)
    return X, y, groups

def segment_3D(X, t, t_start=0,window=None):
    """ segments EEG and time [trials, channels, t_start:t_end]. 
    t is an array of timestamps in ms. 
    t_start: ms (at which to start trial window)
    window: length in ms of window after t_start
        """
    cue_index = np.flatnonzero(t>=t_start)[0]
    if isinstance(window,type(None)):
        end_index = len(t)-1
    else:
        end_index = np.int(cue_index + window)
    X = X[:,:,cue_index:end_index]
    t = t[cue_index:end_index]
    return X, t


def segment(X, y, groups, w, step):
    n_trials = X.shape[0]
    X, y, groups = segment_ts(X, y, groups, w, step)
    X, y, groups = order_by_trials(X, y, groups, n_trials)
    return X, y, groups
