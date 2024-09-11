#evaluation of results and printing it out:
import os
from numpy import np
from filelock import FileLock

def acc_table(subject, train_acc, test_acc, suffix, trial_len, classifier,**_):
    acc_summary = [np.unique(subject)[0], 
                   np.mean(train_acc),
                   np.std(train_acc),
                   np.mean(test_acc), 
                   np.std(test_acc),
                   np.unique(suffix)[0],
                   np.unique(trial_len)[0],
                   np.unique(classifier)[0],
                  ]
    return acc_summary

def format_stats(x):
    if isinstance(x, str) or isinstance(x, int): x=x
    elif isinstance(x,type(dict())): x=str(np.nan)
    #else: x = '{:.5f}'.format(x)
    return str(x)


def save_summary(fname, subject, results):
    fname = fname + '.txt'
    header = ['subject', 'train_acc_mean','train_acc_std', 'test_acc_mean','test_acc_std', 'suffix', 'trial_len', 'classifier']# list(results.keys())
    acc_summary = acc_table(**results)
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname))
    if not os.path.exists(fname):
         out_ = '\t'.join(map(format_stats, header))+'\n'+'\t'.join(map(format_stats, acc_summary))
    else: 
        out_ = '\n'+'\t'.join(map(format_stats, acc_summary))
    with FileLock(fname + ".lock"): 
        with open(fname, "a+") as f:  _ = f.write(out_)