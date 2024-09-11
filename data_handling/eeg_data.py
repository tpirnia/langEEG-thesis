import numpy as np
import pandas as pd
import json

def load_subject_data(data_path, subject,suffix='c_avg'):
    fname = '/'.join([data_path, 'data', subject, 'data_' + suffix + '.npz'])
    data = np.load(fname)
    d = dict(zip(("{}".format(k) for k in data), (data[k] for k in data)))
    d['subject']= np.str(d['subject'])
    return d


def load_X(path, subject, t_end, suffix, t_start=0, **_):
    """Load data, trim EEG epochs, & remove bad trials
    """
    data = load_subject_data(path, subject, suffix=suffix)
    
    data['EEG'], data['t'] = trim_EEG(data['EEG'], data['t'], t_start=t_start, t_end=t_end)
    
    X = remove_bad_channels(data['ch'],data['EEG'])
    return X

def list_bad_channels(ch):
    bad_ch = np.unique([np.where(x==-1)[0] for x in ch])
    return bad_ch

def remove_bad_channels(ch, eeg):
    bad_ch = list_bad_channels(ch)
    return np.delete(eeg, bad_ch, axis=1)

def get_phase_labels(fname):
    """load label table, convert phase labels to binary, set as y.
        Arguments:
                fname: pkl label file (i.e. '/project_data/volume0/tpirnia/data/EEGWS/ \
                       data/<subject>/data_<suffix>_labels.pkl')
        
        Returns:
                y: binary phase labels per trial (+1: reverse, -1: straight)
    """
    labels = pd.read_pickle(fname)
    y = np.zeros(labels['phase'].shape,dtype=int)
    y[labels['phase']=='R'] = 1
    y[labels['phase']=='S'] = -1
    return y

def get_blocks(fname):
    """ block label per each trial.
        Arguments:
                fname: /project_data/volume0/tpirnia/data/EEGWS/ \
                        data/<subject>/data_<suffix>_labels.pkl'
        
        Returns:
                y: binary labels (+1: reverse, -1: straight)
    """
    labels = pd.read_pickle(fname)
    return labels['block']

def load_phase_data(project_path, subject, t_end, suffix,**_):
    """
    path = project_path
    """
    X = load_X(project_path, subject, t_end, suffix=suffix)
    fname = ''.join([project_path, 
                            '/data/', 
                            subject, 
                            '/data_', 
                            suffix, 
                            '_labels.pkl'],
                          )
    y = get_phase_labels(fname)
    groups = get_blocks(fname)
    return X, y, groups

