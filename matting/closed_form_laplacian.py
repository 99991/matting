import numpy as np
from .util import pad, make_windows, weights_to_laplacian
import scipy.sparse


def closed_form_laplacian(image, epsilon):
    h, w, depth = image.shape
    n = h * w

    indices = np.arange(n).reshape(h, w)
    neighbor_indices = make_windows(pad(indices))

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(h, w, 1, depth)
    # shape: h w 3 3
    covariance = np.matmul(centered_neighbors.transpose(0, 1, 3, 2), centered_neighbors) / (3 * 3)

    inv_cov = np.linalg.inv(covariance + epsilon / (3 * 3) * np.eye(3, 3))

    # shape: h w 9 3
    weights = np.matmul(centered_neighbors, inv_cov)
    # shape: h w 9 9
    weights = 1 + np.matmul(weights, centered_neighbors.transpose(0, 1, 3, 2))

    i_inds = np.tile(neighbor_indices, 3 * 3).flatten()
    j_inds = np.repeat(neighbor_indices, 3 * 3).flatten()
    weights = weights.flatten()

    W = scipy.sparse.csc_matrix((weights, (i_inds, j_inds)), shape=(n, n))

    L = weights_to_laplacian(W)

    return L
