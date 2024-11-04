import sys, os
import numpy as np
import copy 

from collections import deque
import pickle as pkl

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

def load_tg(test_subject, 
            train_subject = '',
            suffix = '',
            pathname = '',
           ):
    
    if not train_subject:
        train_subject=test_subject
        
    fname = pathname +'{}_trained{}{}.npy'.format(test_subject, 
                                                  train_subject, 
                                                  suffix) 
    return np.load(fname)

def make_mask(n_times, n_permutations=0):
    mask_mat = np.zeros((n_times,n_times))
    for i in np.arange(0,n_times):
        for j in np.arange(0,n_times):
            if np.abs(i-j)> mask_dist_: # distance from diagonal
                mask_mat[i,j] = 1

    # transpose to be train_windows x perm_num x test_windows
    if n_permutations:
        mask_mat=np.transpose( mask_mat[:,:,np.newaxis] * np.ones((n_permutations)),
                              (0,2,1)) #n_permutations * mask matrix
    return mask_mat

def _mask(mtx, mask):
    mtx[mask==1] = 0
    return mtx

def _threshold(mtx, val):
    mtx[ np.where(mtx<val) ] = np.nan
    return mtx

def format_observations(obs, thresh, mask):
    '''
    dimensions are probably wrong here since it is pulled from null/permutation outputs'''
    # 2.1 - start at cue
    # r, c = nl.shape[0], nl.shape[-1]
    if obs.shape[0] == 146: #maybe switch to obs[-96:] instead of obs[50:]
        obs = obs[50:]
    if obs.shape[1] == 146:
        obs = obs[:,50:]

    # 2.2 - mask TG by 250ms
    obs = _mask(copy.deepcopy(obs), mask)

    # 2.3 - threshold accuracy
    obs = _threshold(obs, thresh)
    
    return obs

def get_full_matrix(adj):
    d = {}
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] == 0:
                continue
            d[(i,j)] = []
            for di,dj in [(0,1), (1,0), (0,-1), (-1,0), (-1,-1), (-1, 1), (1, -1), (1,1)]:
                ni,nj = i+di, j+dj
                if ni<0 or nj<0 or ni>=len(adj) or nj>=len(adj[ni]):
                    continue
                if adj[ni][nj] == 0:
                    continue
                d[(i,j)].append((ni,nj))
    return d

def get_clusters(adj):
    islands = []
    seen = set()
    for coordinate in adj:
        if coordinate in seen:
            continue
        queue = deque([coordinate])
        island = set()
        while queue:
            current = queue.popleft() #queue.pop(0)#
            island.add(current)
            neighbours = adj[current]
            tmp = set(neighbours) - set(island)
            if tmp:
                queue.extend(list(tmp))
                queue = deque(list(set(queue)))
        islands.append(island)
        seen.update(island)
    return islands

if __name__ == '__main__':
    subject = sys.argv[1]
    thresh = np.float(sys.argv[2])

    # subject = 'EP03'
    print(subject)

    ## time points for plot
    times_ = np.arange(0,960,10)
    n_times, mask_dist_= len(times_), 25

    datapath='/project_data/volume0/tpirnia/data/EEGWS/data/AIM3/data_interp_loo/'

    out_fname = (datapath +'clusters/' 
             + 'threshold_accuracy_{}/{}_trained{}'.format(thresh,
                                                           subject, 
                                                           '-loo'))

    fill_path(out_fname)
    # 2.0 - load results
    obs = load_tg(subject, '-loo',
                      suffix='',
                      pathname = datapath)


    mask = make_mask(n_times)

    obs = format_observations(obs, thresh, mask)

    tmp = np.nan_to_num(obs)
    am = get_full_matrix(tmp)
    clust = get_clusters(am)


    clust_size=list(map(len, clust))

    fill_path(out_fname)

    #out_fname+'-clusters_size.npy'

    np.save(out_fname+'-clusters_size.npy',np.asarray(clust_size))
    with open((out_fname+'-clusters.pkl'),'wb') as ff: 
        pkl.dump(clust,ff)