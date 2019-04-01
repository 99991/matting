from .util import weights_to_laplacian
from .knn import knn
import numpy as np
import scipy.sparse

def knn_laplacian(image, normalize=True):
    # build knn laplacian from image using nearest neighbor information
    h,w,depth = image.shape
    n = h*w
    assert(depth == 3)
    # calculate pixel positions
    x = np.arange(1, w + 1)
    y = np.arange(1, h + 1)
    x,y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    # calculate feature matrix from colors and pixel positions
    # distance scale factor
    d = np.sqrt(w*w + h*h)/2
    r,g,b = image.reshape(n, 3).T
    features = np.stack([r,g,b,x/d,y/d], axis=1)
    features = features.astype(np.float32)

    # empty list of nearest neighbor pairs
    ij = np.zeros((0, 2), dtype=np.int64)

    for n_neighbors in [10, 2]:
        # for each pixel
        a = np.repeat(np.arange(n), n_neighbors)
        # find nearest neighbor
        b = knn(features, features, n_neighbors).flatten()

        # sort nearest neighbor pairs by i-index
        # (needed for uniqueness test latter)
        more_ij = np.stack([
            np.minimum(a, b),
            np.maximum(a, b),
        ], axis=1)

        # add new nearest neighbor pairs
        ij = np.concatenate([ij, more_ij], axis=0)
        
        # decrease distance cost by scaling coordinates
        features[:, -2:] *= 0.01

    # discard duplicate nearest neighbor pairs
    ij = np.unique(ij, axis=0)

    i_inds = ij[:, 0]
    j_inds = ij[:, 1]

    # sophisticated weight function
    values = np.ones(len(i_inds))

    # build weight matrix
    W = scipy.sparse.coo_matrix((values, (i_inds, j_inds)), shape=(n, n))
    
    W = W + W.T
    
    return weights_to_laplacian(W, normalize=normalize)
