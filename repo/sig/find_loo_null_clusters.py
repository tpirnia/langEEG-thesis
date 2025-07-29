import sys,os
import numpy as np
import copy 

from collections import deque
import pickle as pkl

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

def load_tg(test_subject, 
            train_subject,
            suffix = '',
            pathname = '',
           ):
    
    if not train_subject:
        train_subject=test_subject
        
    fname = pathname +'{}_trained{}{}.npy'.format(test_subject, 
                                                  train_subject, 
                                                  suffix) 
    return np.load(fname)

def _mask(mtx, mask):
    mtx[mask==1] = 0
    return mtx

def _threshold(mtx, val):
    mtx[ np.where(mtx<val) ] = np.nan
    return mtx

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

def make_mask(n_times, n_permutations):
    mask_mat = np.zeros((n_times,n_times))
    for i in np.arange(0,n_times):
        for j in np.arange(0,n_times):
            if np.abs(i-j)> mask_dist_: # distance from diagonal
                mask_mat[i,j] = 1

    # transpose to be train_windows x perm_num x test_windows
    mask_mat=np.transpose( mask_mat[:,:,np.newaxis] * np.ones((n_permutations)),
                          (0,2,1)) #n_permutations * mask matrix
    return mask_mat


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

def format_null(nl, thresh, mask):
    # 2.1 - average across folds:
    nl = nl.squeeze()

    # 2.2 - start at cue
    #nl = nl[50:,:,50:]

    # 2.3,2.4 - mask TG by 250ms
    nl = _mask(copy.deepcopy(nl), mask)

    # 2.5 - threshold accuracy
    nl = _threshold(nl, thresh)

    nl = np.swapaxes(nl, 0, 1) # n x train_time x test_time
    return nl

if __name__ == '__main__':
    subject = sys.argv[1]
    thresh = float(sys.argv[2])
    train_subject='-loo'
    print(subject)

    ## time points for plot
    times_ = np.arange(0,960,10)
    n_times, mask_dist_= len(times_), 25


    datapath='/project_data/volume0/tpirnia/data/EEGWS/thesis_repository/results/TG/'

    out_fname = (datapath 
                 + 'clusters/' 
                 + '{}_threshold_accuracy_{}/{}_trained{}{}'.format( 'null',
                                                   thresh,
                                                   subject, 
                                                   train_subject, 
                                                   '-null'))

    fill_path(datapath
                 + 'clusters/' 
                 + '{}_threshold_accuracy_{}/'.format('null', thresh))

    # 2.0 - load each permutation
    nulls = load_tg(subject, train_subject,
                      suffix = '-null',
                      pathname = datapath + 'null/')

    n_permutations = nulls.shape[1]

    mask = make_mask(n_times, n_permutations)

    nulls = format_null(nulls, thresh, mask)

    max_clusts = []
    clusters = []
    for pp in range(n_permutations):
        tmp = np.nan_to_num( nulls[pp])
        am = get_full_matrix(tmp)
        clust = get_clusters(am)


        clust_size=list(map(len, clust))
        max_clusts.append(max(clust_size,default=0))
        clusters.append(clust)

    np.save(out_fname +'-maxdist.npy', np.asarray(max_clusts))
    with open((out_fname + '-clusters.pkl'),'wb') as ff: 
        pkl.dump(clusters,ff)