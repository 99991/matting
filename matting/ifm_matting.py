# The information flow alpha matting method implementation in this file
# is based on
# https://github.com/yaksoy/AffinityBasedMattingToolbox
# by Yağız Aksoy.
#
############################################################################################
# Copyright 2017, Yagiz Aksoy. All rights reserved.                                        #
#                                                                                          #
# This software is for academic use only. A redistribution of this                         #
# software, with or without modifications, has to be for academic                          #
# use only, while giving the appropriate credit to the original                            #
# authors of the software. The methods implemented as a part of                            #
# this software may be covered under patents or patent applications.                       #
#                                                                                          #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR IMPLIED             #
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND #
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR         #
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR      #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF     #
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                               #
############################################################################################

r"""
The information flow alpha matting method is provided for academic use only.
If you use the information flow alpha matting method for an academic
publication, please cite corresponding publications referenced in the
description of each function:

@INPROCEEDINGS{ifm,
author={Aksoy, Ya\u{g}{\i}z and Ayd{\i}n, Tun\c{c} Ozan and Pollefeys, Marc},
booktitle={Proc. CVPR},
title={Designing Effective Inter-Pixel Information Flow for Natural Image Matting},
year={2017},
}
"""

import numpy as np
import numpy.linalg
import scipy.sparse.linalg
from .util import pad, weights_to_laplacian, make_windows
from ctypes import c_int, c_double, c_char_p, c_ubyte, POINTER
from .load_libmatting import load_libmatting
from .knn import knn

c_ubyte_p = POINTER(c_ubyte)
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

library = load_libmatting()

_label_expand = library.label_expand
_label_expand.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_int,
    c_int,
    c_int,
    c_ubyte_p,
    c_double,
    c_double_p,
    c_double_p]

np.random.seed(0)


def mul_vec_mat_vec(v, A, w):
    # calculates v' A w
    return np.einsum("...i,...ij,...j->...", v, A, w)


def mul_mat_mat_matT(A, B, C):
    # calculates A B C.T
    return np.einsum("...ij,...jk,...lk->...il", A, B, C)


def mul_matT_mat(A, B):
    # calculates A.T B
    return np.einsum("...ji,...jk->...ik", A, B)


def mul_mat_matT(A, B):
    # calculates A B.T
    return np.einsum("...ij,...kj->...ik", A, B)


def boxfilter(image, r, mode='full', fill_value=0.0):
    height, width = image.shape[:2]
    size = 2 * r + 1

    pad = {
        'full': 2 * r,
        'valid': 0,
        'same': r,
    }[mode]

    shape = [1 + pad + height + pad, 1 + pad + width + pad] + list(image.shape[2:])
    image_padded = np.full(shape, fill_value, dtype=image.dtype)
    image_padded[1 + pad:1 + pad + height, 1 + pad:1 + pad + width] = image

    c = np.cumsum(image_padded, axis=0)
    c = c[size:, :] - c[:-size, :]
    c = np.cumsum(c, axis=1)
    c = c[:, size:] - c[:, :-size]

    return c


def imdilate(image, radius):
    return boxfilter(image, radius, 'same') > 0


def make_windows_at(image, r, x, y):
    if len(image.shape) == 3:
        height, width, depth = image.shape
        return np.array([
            image[y + dy, x + dx]
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
        ]).transpose(1, 0, 2)
    else:
        return np.array([
            image[y + dy, x + dx]
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
        ]).transpose(1, 0)


def label_expand(image, rows, cols, r, w, h, knownReg, colorThresh, trimap, extendedTrimap):
    _label_expand(
        np.ctypeslib.as_ctypes(image.ravel()),
        np.ctypeslib.as_ctypes(rows.astype(np.int32)),
        np.ctypeslib.as_ctypes(cols.astype(np.int32)),
        len(rows),
        r,
        w,
        h,
        np.ctypeslib.as_ctypes(knownReg.astype(np.uint8).ravel()),
        colorThresh,
        np.ctypeslib.as_ctypes(trimap.ravel()),
        np.ctypeslib.as_ctypes(extendedTrimap.ravel()))


def LabelExpansion(image, trimap, maxDist, colorThresh):
    fg = trimap > 0.8
    bg = trimap < 0.2
    not_fg = np.logical_not(fg)
    not_bg = np.logical_not(bg)

    knownReg = np.logical_or(fg, bg)

    searchReg = np.logical_or(
        np.logical_and(imdilate(fg, maxDist), not_fg),
        np.logical_and(imdilate(bg, maxDist), not_bg))

    h, w = trimap.shape
    cols = np.arange(w)
    rows = np.arange(h)
    cols, rows = np.meshgrid(cols, rows)

    cols = cols[searchReg]
    rows = rows[searchReg]

    extendedTrimap = trimap.copy()

    label_expand(image, rows, cols, maxDist, w, h, knownReg, colorThresh, trimap, extendedTrimap)

    return extendedTrimap


def trimmingFromKnownUnknownEdges(image, trimap):
    for i, threshold in enumerate(np.linspace(8.11 / 256, 1 / 256, 9)):
        trimap = LabelExpansion(image, trimap, i + 1, threshold)
    return trimap


def findNonlocalNeighbors(image, K, xyWeight=1.0, inMap=None, outMap=None):
    h, w, c = image.shape

    if inMap is None:
        inMap = np.ones((h, w), dtype=np.bool8)

    if outMap is None:
        outMap = np.ones((h, w), dtype=np.bool8)

    if xyWeight > 0.0:
        x = np.arange(1, w + 1)
        y = np.arange(1, h + 1)
        x, y = np.meshgrid(x, y)

        x = xyWeight * x.astype(np.float64) / w
        y = xyWeight * y.astype(np.float64) / h

        features = np.stack([
            image[:, :, 0].flatten(),
            image[:, :, 1].flatten(),
            image[:, :, 2].flatten(),
            x.flatten(),
            y.flatten(),
        ], axis=1)
    else:
        features = np.stack([
            image[:, :, 0].flatten(),
            image[:, :, 1].flatten(),
            image[:, :, 2].flatten(),
        ], axis=1)

    indices = np.arange(h * w)

    inMap = inMap.flatten()
    outMap = outMap.flatten()

    inInd = indices[inMap]
    outInd = indices[outMap]

    neighbors = knn(features[outMap], features[inMap], K)

    neighInd = outInd[neighbors]

    return inInd, neighInd, features


def solve_for_weights(z, regularization_factor=1e-3):
    n, n_neighbors, _ = z.shape

    # calculate covariance matrices
    C = mul_mat_matT(z, z)

    # regularization
    C += regularization_factor * np.eye(n_neighbors)

    # solve for weights
    weights = np.linalg.solve(C, np.ones((n, n_neighbors)))
    # normalize rows
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def colorMixtureAffinities(image, K, inMap=None, outMap=None, xyWeight=1.0, useXYinLLEcomp=False):
    h, w, c = image.shape
    N = h * w

    if inMap is None:
        inMap = np.ones((h, w), dtype=np.bool8)

    if outMap is None:
        outMap = np.ones((h, w), dtype=np.bool8)

    inInd, neighInd, features = findNonlocalNeighbors(image, K, xyWeight, inMap, outMap)

    if not useXYinLLEcomp:
        features = features[:, :-2]

    inInd = np.repeat(inInd, K).reshape(-1, K)
    flows = solve_for_weights(features[inInd] - features[neighInd], regularization_factor=1e-10)

    i = inInd.flatten()
    j = neighInd.flatten()
    v = flows.flatten()

    W = scipy.sparse.csr_matrix((flows.flatten(), (inInd.flatten(), neighInd.flatten())), shape=(N, N))

    return W


def mattingAffinity(image, inMap, windowRadius, eps):
    height, width, depth = image.shape
    n = height * width

    window_size = (2 * windowRadius + 1)**2

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(height, width, 1, depth)
    # shape: h w 3 3
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / window_size

    inv_cov = np.linalg.inv(covariance + eps / window_size * np.eye(3, 3))

    indices = np.arange(width * height).reshape(height, width)
    neighInd = make_windows(indices)

    inMap = inMap[windowRadius:-windowRadius, windowRadius:-windowRadius]

    neighInd = neighInd.reshape(-1, window_size)

    neighInd = neighInd[inMap.flatten()]

    inInd = neighInd[:, window_size // 2]

    image = image.reshape(-1, 3)
    means = means.reshape(-1, 3)
    inv_cov = inv_cov.reshape(-1, 3, 3)

    centered_neighbors = image[neighInd] - means[inInd].reshape(-1, 1, 3)

    weights = mul_mat_mat_matT(centered_neighbors, inv_cov[inInd], centered_neighbors)

    flowCols = np.repeat(neighInd, window_size, axis=1).reshape(-1, window_size, window_size)
    flowRows = flowCols.transpose(0, 2, 1)

    weights = (weights + 1) / window_size

    flowRows = flowRows.flatten()
    flowCols = flowCols.flatten()
    weights = weights.flatten()

    W = scipy.sparse.csc_matrix((weights, (flowRows, flowCols)), shape=(n, n))

    W = W + W.T

    W_row_sum = np.array(W.sum(axis=1)).flatten()
    W_row_sum[W_row_sum < 0.05] = 1.0

    return scipy.sparse.diags(1 / W_row_sum).dot(W)


def colorSimilarityAffinities(image, K, inMap=None, outMap=None, xyWeight=0.05):
    h, w, c = image.shape
    N = h * w

    if inMap is None:
        inMap = np.ones((h, w), dtype=np.bool8)

    if outMap is None:
        outMap = np.ones((h, w), dtype=np.bool8)

    _, neighInd, _ = findNonlocalNeighbors(image, K, xyWeight, inMap, outMap)

    # This behaviour below, decreasing the xy-weight and finding a new set of neighbors, is taken
    # from the public implementation of KNN matting by Chen et al.
    inInd, neighInd2, features = findNonlocalNeighbors(image, int(np.ceil(K / 5)), xyWeight / 100, inMap, outMap)

    neighInd = np.concatenate([neighInd, neighInd2], axis=1)
    features[-2:] /= 100.0

    inInd = np.repeat(inInd, neighInd.shape[1]).reshape(-1, neighInd.shape[1])
    flows = 1 - np.mean(np.abs(features[inInd] - features[neighInd]), axis=2)
    flows[flows < 0] = 0

    W = scipy.sparse.csr_matrix((flows.flatten(), (inInd.flatten(), neighInd.flatten())), shape=(N, N))

    W = 0.5 * (W + W.T)

    return W


def affinityMatrixToLaplacian(A):
    return weights_to_laplacian(A, normalize=False)


def patchBasedTrimming(image, trimap, minDist, maxDist, windowRadius, K):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    trimap = trimap.copy()
    height, width, depth = image.shape

    eps = 1e-8

    # shape: h w 3
    means = make_windows(pad(image)).mean(axis=2)
    # shape: h w 9 3
    centered_neighbors = make_windows(pad(image)) - means.reshape(height, width, 1, depth)
    # shape: h w 3 3
    covariance = mul_matT_mat(centered_neighbors, centered_neighbors) / (3 * 3) \
        + eps / (3 * 3) * np.eye(3, 3)

    unkInd, fgNeigh, _ = findNonlocalNeighbors(means, K, -1, is_unknown, is_fg)
    _, bgNeigh, _ = findNonlocalNeighbors(means, K, -1, is_unknown, is_bg)

    meanImage = means.transpose(0, 1, 2).reshape(height * width, depth)

    covariance = covariance.transpose(0, 1, 2, 3).reshape(width * height, 3, 3)

    pixMeans = meanImage[unkInd]
    pixCovars = covariance[unkInd]
    pixDets = np.linalg.det(pixCovars)
    pixCovars = pixCovars.reshape(unkInd.shape[0], 1, 3, 3)

    nMeans = meanImage[fgNeigh] - pixMeans.reshape(unkInd.shape[0], 1, 3)
    nCovars = covariance[fgNeigh]
    nDets = np.linalg.det(nCovars)
    nCovars = (pixCovars + nCovars) / 2

    fgBhatt = 0.125 * mul_vec_mat_vec(nMeans, np.linalg.inv(nCovars), nMeans) \
        + 0.5 * np.log(np.linalg.det(nCovars) / np.sqrt(pixDets[:, None] * nDets))

    nMeans = meanImage[bgNeigh] - pixMeans.reshape(unkInd.shape[0], 1, 3)
    nCovars = covariance[bgNeigh]
    nDets = np.linalg.det(nCovars)
    nCovars = (pixCovars + nCovars) / 2

    bgBhatt = 0.125 * mul_vec_mat_vec(nMeans, np.linalg.inv(nCovars), nMeans) \
        + 0.5 * np.log(np.linalg.det(nCovars) / np.sqrt(pixDets[:, None] * nDets))

    shape = trimap.shape

    minFGdist = np.min(fgBhatt, axis=1)
    minBGdist = np.min(bgBhatt, axis=1)

    mask0 = np.logical_and(minBGdist < minDist, minFGdist > maxDist)
    mask1 = np.logical_and(minFGdist < minDist, minBGdist > maxDist)

    trimap[np.unravel_index(unkInd[mask0], shape)] = 0
    trimap[np.unravel_index(unkInd[mask1], shape)] = 1

    return trimap


def knownToUnknownColorMixture(image, trimap, K, xyWeight):
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    # Find neighbors of unknown pixels in FG and BG
    inInd, bgInd, features = findNonlocalNeighbors(image, K, xyWeight, is_unknown, is_bg)
    _, fgInd, _ = findNonlocalNeighbors(image, K, xyWeight, is_unknown, is_fg)

    neighInd = np.concatenate([fgInd, bgInd], axis=1)

    # Compute LLE weights and estimate FG and BG colors that got into the mixture
    features = features[:, :-2]
    flows = np.zeros((inInd.shape[0], neighInd.shape[1]))
    fgCols = np.zeros((inInd.shape[0], 3))
    bgCols = np.zeros((inInd.shape[0], 3))

    flows = solve_for_weights(features[inInd].reshape(-1, 1, 3) - features[neighInd], 1e-10)

    fgCols = np.sum(features[neighInd[:, :K]] * flows[:, :K, np.newaxis], axis=1)
    bgCols = np.sum(features[neighInd[:, K:]] * flows[:, K:, np.newaxis], axis=1)

    alphaEst = trimap.copy()
    alphaEst[is_unknown] = np.sum(flows[:, :K], axis=1)

    # Compute the confidence based on FG - BG color difference
    confidence_of_unknown = np.sum(np.square(fgCols - bgCols), 1) / 3

    conf = is_known.astype(np.float64)
    conf[is_unknown] = confidence_of_unknown

    return alphaEst, conf


def ifm_system(image, trimap):
    params = {
        "lambda": 100.0,
        "mattePostTrim": 0,
        "cm_K": 20,
        "cm_xyw": 1,
        "cm_mult": 1,
        "ku_K": 7,
        "ku_xyw": 10,
        "ku_mult": 0.05,
        "iu_K": 5,
        "iu_xyw": 0.05,
        "iu_mult": 0.01,
        "loc_win": 1,
        "loc_eps": 1.0000e-06,
        "loc_mult": 1,
        "refinement_mult": 0.1,
    }

    h, w = trimap.shape
    n = h * w
    is_fg = trimap > 0.8
    is_bg = trimap < 0.2
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)

    L = affinityMatrixToLaplacian(colorMixtureAffinities(image, params["cm_K"], is_unknown, None, params["cm_xyw"]))

    L = params["cm_mult"] * L.T.dot(L)

    dilUnk = imdilate(is_unknown, params["loc_win"])

    L = L + params["loc_mult"] * affinityMatrixToLaplacian(mattingAffinity(image, dilUnk, params["loc_win"], params["loc_eps"]))
    L = L + params["iu_mult"] * affinityMatrixToLaplacian(colorSimilarityAffinities(image, params["iu_K"], is_unknown, is_unknown, params["iu_xyw"]))

    edgeTrimmed = trimmingFromKnownUnknownEdges(image, trimap)

    patchTrimmed = patchBasedTrimming(image, trimap, 0.25, 0.9, 1, 5)

    kToU, kToUconf = knownToUnknownColorMixture(image, patchTrimmed, params["ku_K"], params["ku_xyw"])
    kToU[edgeTrimmed < 0.2] = 0
    kToU[edgeTrimmed > 0.8] = 1
    kToUconf[edgeTrimmed < 0.2] = 1
    kToUconf[edgeTrimmed > 0.8] = 1

    d = is_known.flatten().astype(np.float64)

    A = params["lambda"] * scipy.sparse.diags(d)

    kToUconf[is_known] = 0

    A = A + params["ku_mult"] * scipy.sparse.diags(kToUconf.flatten())

    b = A.dot(kToU.flatten())

    A = A + L

    return A, b
