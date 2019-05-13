import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def make_P(shape):
    h, w = shape
    n = h * w
    h2 = h // 2
    w2 = w // 2
    n2 = w2 * h2
    weights = np.float64([
        1, 2, 1,
        2, 4, 2,
        1, 2, 1]) / 16

    x2 = np.arange(w2)
    y2 = np.arange(h2)
    x2, y2 = np.meshgrid(x2, y2)
    x2 = np.repeat(x2.flatten(), 9)
    y2 = np.repeat(y2.flatten(), 9)

    x = x2 * 2 + np.tile([-1, 0, 1, -1, 0, 1, -1, 0, 1], n2)
    y = y2 * 2 + np.tile([-1, -1, -1, 0, 0, 0, 1, 1, 1], n2)

    mask = (0 <= x) & (x < w) & (0 <= y) & (y <= h)

    i_inds = (x2 + y2 * w2)[mask]
    j_inds = (x + y * w)[mask]
    values = np.tile(weights, n2)[mask]

    downsample = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n2, n))
    upsample = downsample.T

    return upsample, downsample


def vcycle(A, b, shape, cache):
    h, w = shape
    n = h * w

    omega = 0.8
    num_pre_iter = 0
    num_post_iter = 1

    if n <= 32:
        return scipy.sparse.linalg.spsolve(A, b)

    if shape not in cache:
        upsample, downsample = make_P(shape)

        coarse_A = downsample @ A @ upsample

        A_diag = A.diagonal()

        inv_A_diag = 1 / A_diag

        cache[shape] = (upsample, downsample, coarse_A, A_diag, inv_A_diag)
    else:
        upsample, downsample, coarse_A, A_diag, inv_A_diag = cache[shape]

    # dampened jacobi iteration on x0 = 0
    x = omega * inv_A_diag * b
    # more dampened jacobi iterations on x
    for _ in range(num_pre_iter):
        x = omega * inv_A_diag * (b - A @ x + A_diag * x) + (1 - omega) * x

    # calculate residual error to perfect solution
    residual = b - A @ x

    # downsample residual error
    coarse_residual = downsample @ residual

    # calculate coarse solution for residual
    coarse_x = vcycle(coarse_A, coarse_residual, (h // 2, w // 2), cache)

    # apply coarse correction
    x += upsample @ coarse_x

    # dampened jacobi iterations on x
    for _ in range(num_post_iter):
        x = omega * inv_A_diag * (b - A @ x + A_diag * x) + (1 - omega) * x

    return x
