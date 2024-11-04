import argparse
import json
import numpy as np

def get_phaseEEG(arg, subject=None, suffix=None):
    """ X, y, groups = get_phaseEEG() 
        - if subject not in arg, adds it.
    """
    if isinstance(arg, str): 
        arg = load_args(arg)
    if (isinstance(subject, str)):
        arg.subject = subject
    #elif not ('subject' in vars(arg)):
    #    print('No subject specified')
    #    pass
    if suffix:
        out_dir = arg.project_path + 'data/' + arg.subject + '/' + suffix + '/'
    else:
        out_dir = arg.project_path + 'data/' + arg.subject + '/'
    X = np.load(out_dir + 'X' + arg.suffix_X + '.npy')
    y = np.load(out_dir + 'y' + arg.suffix_y + '.npy')
    groups = np.load(out_dir + 'blocks' + arg.suffix_groups + '.npy')
    return X, y, groups, arg
    
def load_args(fname, subject=None):
    if not fname[-4:] == '.txt':
        fname = fname + '.txt'
    config = json.load(open(fname))
    arg = argparse.Namespace(**config)
    if isinstance(subject,type(np.str)):
        arg.subject = subject
    return arg

def load_multi_subject(arg, subject_list, bad_ch=None):
    """ Loads all subject's x,y, groups data from subject_list with parameters from arg.
        arguments:
                arg:          namespace with project path and data file suffix
                subject_list: list of subjects to load data from
                
        return:
                X:      [n_subjects * n_trials, n_channels, time]
                y:      [n_subjects * n_trials,]
                groups: [n_subjects * n_trials,]
                ind_to_subject: [n_subjects * n_trials,]
                
    """
    ind=0
    X, y, groups, ind_to_subject = [], [], [], []
    for subject in subject_list:
        x_, y_, g_, _ = get_phaseEEG(arg, subject)
        ind_to_subject.append([subject]*len(y_))
        X.append(x_)
        y.append(y_)
        groups.append(g_)
    if not isinstance(bad_ch, type(None)):
        X = [np.delete(x, bad_ch, axis=1) for x in X]
    return np.vstack(X), np.hstack(y), np.hstack(groups), np.hstack(ind_to_subject)
        
def remove_channels(X, ch):
    return np.delete(X, bad_ch, axis=1)
    
# def get_EEG(arg, prefix='', subject=None):
#     """ X, y, groups = get_phaseEEG() 
#         - if subject not in arg, adds it.
#     """
#     if isinstance(arg, type(str)): 
#         arg = load_args(arg)
#     if not ('subject' in vars(arg)):
#         arg.subject = subject
#     out_dir = arg.project_path + '/data/' + arg.subject + '/'
#     X = np.load(out_dir + prefix + 'X.npy')
#     y, groups = np.load(out_dir + prefix + 'y.npy'), np.load(out_dir + 'blocks.npy')
#     return X, y, groups, arg