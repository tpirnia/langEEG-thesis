# need to update this to use content from ch1_classify_by_channel_plot_average_all_subjects.ipynb -- 05/10/23
import os, sys
import numpy as np
import itertools, warnings


def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

def nulldist(nulls, yhat):
    """ CALCULATE P-VALUE"""
    pv0 = sum(nulls >= yhat) + 1 
    return pv0/(len(nulls) + 1)



if __name__ == '__main__':
    
    subject = sys.argv[1]
    w = int(sys.argv[2])
    
    project_path='/project_data/volume0/tpirnia/thesis_repository'
    
    sys.path.append(project_path+'/repo')
    from data_handling.preprocessing import get_bad_channels
    from sig.fdr import fdr_correction
    
    data_path='{}/results/phase_accuracy_by_channel_w{}/'.format(project_path, w)
    #input:
    obs_fname ='{}/obs/{}_by_channel_test.npy'.format(data_path, subject)
    nulls_fname = '{}/null/{}_by_channel_test.npy'.format(data_path, subject)
    
    #output:
    pval_fname = '{}/sig/{}_by_channel-pval.npy'.format(data_path, subject)
    fdr_path = '{}/fdr/{}_by_channel-'.format(data_path, subject)
    
    obs_,nulls_ = np.load(obs_fname),np.load(nulls_fname)

    nulls_ = nulls_.swapaxes(0,1)
    nulls_ = np.mean(nulls_, axis=2).squeeze()[:,:,np.newaxis] 
    obs_ = obs_.mean(axis=1)
    testd1,testd2 = nulls_.shape[0], nulls_.shape[-1] # test accuracy - dim 1 & 2
    
    d1 = np.arange(testd1) #1st dimension = channels
    bad_d1 = get_bad_channels(project_path,subject) #bad channels
    good_d1 = d1[np.isin(d1, bad_d1, invert=True)]

    pvalues=np.zeros((testd1, testd2))
    for j,k in itertools.product(good_d1,np.arange(testd2)): #included 2nd dimension for future
            pvalues[j,k] = nulldist(nulls_[j,:,k].squeeze(),obs_[j,k])
    pvalues[bad_d1] = np.nan
    
    fill_path(pval_fname)
    np.save(pval_fname,pvalues)
    
    #fdr correction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fdr_ind, fdr_pvalues = np.zeros((testd1, testd2), dtype=int), np.zeros((testd1, testd2))
        fdr_ind[good_d1], fdr_pvalues[good_d1] = fdr_correction(pvalues[good_d1], alpha=0.05, method="by")
        fdr_pvalues[bad_d1] = np.nan

    fill_path(fdr_path)
    np.save(fdr_path + 'ind-by.npy',fdr_ind)
    np.save(fdr_path + 'pval-by.npy',fdr_pvalues)
