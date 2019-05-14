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


def jacobi(A, A_diag, b, x, num_iter, omega):
    if x is None:
        if num_iter > 0:
            x = omega * b / A_diag
            num_iter -= 1
        else:
            x = np.zeros_like(b)

    for _ in range(num_iter):
        x = x + omega * (b - A @ x) / A_diag

    return x


def gauss_seidel(L, U, b, x, num_iter):
    from .ichol import backsub_L_csc_inplace

    if x is None:
        if num_iter > 0:
            x = b.copy()
            backsub_L_csc_inplace(L, x)
            num_iter -= 1
        else:
            x = np.zeros_like(b)

    for _ in range(num_iter):
        x = b - U @ x
        backsub_L_csc_inplace(L, x)

    return x


def vcycle(
    A,
    b,
    shape,
    cache,
    num_pre_iter=1,
    num_post_iter=1,
    omega=0.8,
    smoothing="jacobi",
):
    h, w = shape
    n = h * w

    omega = 0.8

    if n <= 32:
        return scipy.sparse.linalg.spsolve(A, b)

    if shape not in cache:
        upsample, downsample = make_P(shape)

        coarse_A = downsample @ A @ upsample

        A_diag = A.diagonal()

        L = scipy.sparse.tril(A).tocsc()
        U = scipy.sparse.triu(A, 1)

        cache[shape] = (upsample, downsample, coarse_A, A_diag, L, U)
    else:
        upsample, downsample, coarse_A, A_diag, L, U = cache[shape]

    # smooth error
    if smoothing == "jacobi":
        x = jacobi(A, A_diag, b, None, num_pre_iter, omega)
    elif smoothing == "gauss-seidel":
        x = gauss_seidel(L, U, b, None, num_pre_iter)
    else:
        raise ValueError("Invalid smoothing method %s" % smoothing)

    # calculate residual error to perfect solution
    residual = b - A @ x

    # downsample residual error
    coarse_residual = downsample @ residual

    # calculate coarse solution for residual
    coarse_x = vcycle(coarse_A, coarse_residual, (h // 2, w // 2), cache)

    # apply coarse correction
    x += upsample @ coarse_x

    # smooth error
    if smoothing == "jacobi":
        x = jacobi(A, A_diag, b, x, num_post_iter, omega)
    elif smoothing == "gauss-seidel":
        x = gauss_seidel(L, U, b, x, num_post_iter)
    else:
        raise ValueError("Invalid smoothing method %s" % smoothing)

    return x
