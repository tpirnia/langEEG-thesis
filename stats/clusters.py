import numpy as np


def thresh_clusters(x, tail=1, t_power=0, adjacency=None, partitions=None, max_step=None, threshold=0.05):
    from scipy import ndimage

    x = np.asanyarray(x)
    
    thresholds = [threshold]
    # set these here just in case thresholds == []
    clusters = list()
    sums = list()
    include = np.ones(x.shape, dtype=bool)
    for ti, thresh in enumerate(thresholds): #use threshold = 0.05
        # these need to be reset on each run
        clusters = list()
        if tail == 0:
            x_ins = [np.logical_and(x > thresh, include),
                     np.logical_and(x < -thresh, include)]
        elif tail == -1:
            x_ins = [np.logical_and(x < thresh, include)]
        else:  # tail == 1
            x_ins = [np.logical_and(x > thresh, include)]
        # loop over tails
        for x_in in x_ins:
            if np.any(x_in):
                _find_clusters_1dir(x, x_in, adjacency, max_step,
                                    t_power, ndimage)
                clusters += out[0]
                sums.append(out[1])
    sums = np.concatenate(sums) if sums else np.array([])
    return clusters, sums


def _find_clusters_1dir(x, x_in, adjacency, max_step, t_power, ndimage):
    """Actually call the clustering algorithm.
    
    
        t_power : float
        Power to raise the statistical values (usually t-values) by before
        summing (sign will be retained). Note that t_power == 0 will give a
        count of nodes in each cluster, t_power == 1 will weight each node by
        its statistical score.
    """
    from scipy import sparse
    if adjacency is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            # slices
            clusters = ndimage.find_objects(labels, n_labels)
            # equivalent to if len(clusters) == 0 but faster
            if not clusters:
                sums = list()
            else:
                index = list(range(1, n_labels + 1))
                if t_power == 1:
                    sums = ndimage.sum(x, labels, index=index)
                else:
                    sums = ndimage.sum(np.sign(x) * np.abs(x) ** t_power,
                                       labels, index=index)
        else:
            # boolean masks (raveled)
            clusters = list()
            sums = np.empty(n_labels)
            for label in range(n_labels):
                c = labels == label + 1
                clusters.append(c.ravel())
                if t_power == 1:
                    sums[label] = np.sum(x[c])
                else:
                    sums[label] = np.sum(np.sign(x[c]) *
                                         np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception("Data should be 1D when using a adjacency "
                            "to define clusters.")
        if isinstance(adjacency, sparse.spmatrix) or adjacency is False:
            clusters = _get_components(x_in, adjacency)
        elif isinstance(adjacency, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, adjacency, max_step)
        else:
            raise ValueError('adjacency must be a sparse matrix or list')
        if t_power == 1:
            sums = [_masked_sum(x, c) for c in clusters]
        else:
            sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)



