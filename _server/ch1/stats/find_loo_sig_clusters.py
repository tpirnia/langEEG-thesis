import sys,os
import numpy as np
import pickle as pkl

def fill_path(fname):
    if not os.path.exists(os.path.dirname(fname)): 
        os.makedirs(os.path.dirname(fname), exist_ok=True)

def load_pkl(fname):
    with open(fname, 'rb') as pickle_file:
        content = pkl.load(pickle_file)
    return content

def _clusters_size(clusters):
    return list(map(len, clusters))

def _cluster_size(cluster):
    return len(list(cluster))

def _cluster_pvalue(cluster_size, null_cluster_dist, n_permutations):
    #sum_ind = sum((mc >= size_clust for mc in null_cluster_dist))
    sum_ind = sum(null_cluster_dist >= cluster_size)
    return (sum_ind + 1) / (n_permutations + 1)

def fix_xy(cluster):
    return list(c[::-1] for c in cluster)

def add_sig(full_arr, cluster, sig_arr=None):
    if sig_arr is None:
        sig_arr = np.zeros(full_arr.shape)
    for x,y in cluster:
        sig_arr[y,x] = full_arr[y,x] 
    return sig_arr


if __name__ == '__main__':
    subject = sys.argv[1]
    thresh = np.float(sys.argv[2])
    print(subject)
    n_perms = 500
    p_thresh = 0.05

    datapath='/project_data/volume0/tpirnia/data/EEGWS/data/AIM3/data_interp_loo/'
    
    obs_fname = (datapath 
                 +'clusters/' 
                 + 'threshold_accuracy_{}/{}_trained{}-clusters.pkl'.format(thresh,
                                                                            subject, 
                                                                            '-loo'))
    null_fname = (datapath 
                  + 'clusters/' 
                  + '{}_threshold_accuracy_{}/{}_trained{}{}-maxdist.npy'.format( 'null',
                                                                                 thresh,
                                                                                 subject, 
                                                                                 '-loo', 
                                                                                 '-null'))
    tg_path = datapath + '{}_trained{}.npy'.format(subject,'-loo')


    outpath = (datapath + 'clusters/sig_threshold_accuracy_{}/'.format(thresh))
    fill_path(outpath)

    #load data:
    obs_clusters = load_pkl(obs_fname)

    max_nulls = np.load(null_fname)

    tg = np.load(tg_path)

    if tg.shape[0] > 96: tg = tg[50:]
    if tg.shape[1] > 96: tg = tg[:,50:]

    sig_tg = np.zeros(tg.shape)
    sig_clusters, insig_clusters = [], []

    for i, clust in enumerate(obs_clusters):
        size_c = _cluster_size(clust)

        pvalue_c = _cluster_pvalue( size_c, 
                                   max_nulls, 
                                   n_perms )

        if pvalue_c > p_thresh: 
            continue

        r_clust = fix_xy(clust)
        sig_clusters.append([pvalue_c,
                             size_c, 
                             r_clust, 
                             i])
        sig_tg = add_sig(tg, r_clust, sig_tg)

    sig_tg[sig_tg==0]= np.nan

    out_fname = outpath + '{}_trained{}'.format(subject, '-loo')

    np.save((out_fname + '.npy'), sig_tg)

    with open((out_fname + '-clusters.pkl'),'wb') as ff: 
        pkl.dump(sig_clusters,ff)