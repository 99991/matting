from .util import solve_cg, make_system, weights_to_laplacian
from .knn import knn
import numpy as np
import scipy.sparse

def knn_laplacian(image):
    h,w,depth = image.shape
    n = h*w
    assert(depth == 3)
    x = np.arange(1, w + 1)
    y = np.arange(1, h + 1)
    x,y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    level = 2

    d = np.sqrt(w*w + h*h)/level

    r,g,b = image.reshape(n, 3).T
    features = np.stack([r,g,b,x/d,y/d], axis=1)
    
    features = features.astype(np.float32)
    
    row = np.zeros((0, 2), dtype=np.int64)

    for n_neighbors in [10, 2]:
        a = np.repeat(np.arange(n), n_neighbors)
        b = knn(features, features, n_neighbors).flatten()

        more_rows = np.stack([
            np.minimum(a, b),
            np.maximum(a, b),
        ], axis=1)

        row = np.concatenate([row, more_rows], axis=0)
        
        # decrease distance cost by scaling coordinates
        features[:, -2:] *= 0.01

    row = np.unique(row, axis=0)

    i_inds = row[:, 0]
    j_inds = row[:, 1]

    values = np.ones(len(i_inds))

    W = scipy.sparse.coo_matrix((values, (i_inds, j_inds)), shape=(n, n))
    
    W = W + W.T
    
    return weights_to_laplacian(W)
