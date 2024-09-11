import os 
import numpy as np
import pickle as pkl
from filelock import FileLock

import numpy as np
import pandas as pd
import time, random, copy, argparse

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
def format_stats(x):
    if isinstance(x,type(dict())): 
        x=str(np.nan)
    return str(x)   

def save_output(fname, results, header=None,sep='\t'):
    if os.path.exists(fname): 
        header = None
    fill_path(fname)
    out_ = '\n'+ sep.join(map(format_stats, results))
    if header:
        out_ = sep.join(map(format_stats, header)) + out_
    with FileLock(fname + ".lock"): 
        with open(fname, "a+") as f:  _ = f.write(out_)
            
            
def load_pkl(fname):
    with open(fname, 'rb') as pickle_file:
        content = pkl.load(pickle_file)
    return content

def sts_to_csv(sts, arg, w_j, suffix=""):
    test_df = pd.concat(sts,ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    out_name = arg.out_name  + suffix
    fname = (arg.project_path 
             + 'data/' 
             + arg.subject 
             + '/results/w'+ np.str(arg.window)+'_s'+np.str(arg.step)+'/' 
             + out_name 
             + '_'
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    test_df.to_csv(fname, index=False)
    
    #by_cols = list(test_df.columns.values[:5]
    by_cols = ['subject', 'classifier', 'within_condition','cross_condition', 'window', 'step', 'w_train', 'test_trial_val']
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    acc_agg = acc_agg.drop(columns=["fold", "test_trial_num"])
    tmp_col_names = ','.join(list(acc_agg.columns))
    acc_agg = list(acc_agg.values.squeeze())
    acc_agg = ','.join(map(np.str, acc_agg))
    group_fname = arg.project_path + 'data/across_subjects/results/' + out_name + '.csv'
    with FileLock(group_fname+".lock"): 
        if not os.path.exists(group_fname):
            with open(group_fname, "a+") as f:  _ = f.write(tmp_col_names+'\n')
        with open(group_fname, "a+") as f:  _ = f.write(acc_agg+'\n')
            
            
def sts_to_cross_csv(sts, arg, w_j, suffix=""):
    #come back to this: need to copy from ch2_crossclassify_run_and_group_acc
    test_df = pd.concat(sts,ignore_index=True)
    test_df = test_df.apply(pd.to_numeric, errors='ignore')
    out_name = arg.out_name  + suffix
    fname = (arg.project_path 
             + 'data/' 
             + arg.subject 
             + '/results/w'+ np.str(arg.window)+'_s'+np.str(arg.step)+'/' 
             + out_name 
             + '_'
             + str(w_j).zfill(3)
             + '.csv')
    fill_path(fname)
    test_df.to_csv(fname, index=False)
    
    by_cols = ['subject', 'classifier', 'within_condition', 
               'cross_condition', 'window', 'step', 'w_train',
               'test_trial_val']
    if arg.perm:
        by_cols = ['subject', 'classifier', 'within_condition', 
                   'cross_condition', 'window', 'step',
                   'w_train','permutation', 'fold',
                   'test_trial_val']
    acc_agg = test_df.groupby(by=by_cols, as_index=False).mean()
    acc_agg = acc_agg.drop(columns=[ "test_trial_num"])

    tmp_col_names = ','.join(list(acc_agg.columns))
    acc_agg = list(acc_agg.values.squeeze())
    #acc_agg = ','.join(map(np.str, acc_agg))
    group_fname = arg.project_path + 'data/across_subjects/results/' + out_name + '.csv'
    with FileLock(group_fname+".lock"): 
        if not os.path.exists(group_fname):
            with open(group_fname, "a+") as f:  _ = f.write(tmp_col_names+'\n')
        with open(group_fname, "a+") as f:  
            for acc_ in acc_agg:
                acc_ = ','.join(map(np.str, acc_))
                _ = f.write(acc_+'\n')
    