import numpy as np
import scipy.sparse
from .boxfilter import boxfilter
from .util import vec_vec_dot, mat_vec_dot, vec_vec_outer

def make_lkm_operators(image, eps=1e-4, radius=2):
    window_size = 2*radius + 1
    window_area = window_size*window_size
    I = image
    h, w, depth = I.shape
    n = w*h
    w2 = w - 2*radius
    h2 = h - 2*radius
    r,g,b = I.transpose(2, 0, 1)
    
    means = boxfilter(I, radius, mode='valid')/window_area
    covs = boxfilter(vec_vec_outer(I, I), radius, mode='valid')
    I_sum = boxfilter(I, radius, mode='valid')
    covs -= 2*vec_vec_outer(means, I_sum)
    covs += vec_vec_outer(means, means)*window_area
    covs /= window_area
    inv_covs = np.linalg.inv(covs + eps/window_area*np.eye(depth))
    Vmeans = mat_vec_dot(inv_covs, means)
    VI = mat_vec_dot(inv_covs, I[radius:-radius, radius:-radius])
    meanVmeans = vec_vec_dot(means, Vmeans)
    meanVmeansPlusOne = meanVmeans + 1
    filterweights = boxfilter(np.full((h2, w2), -1.0*window_area), radius, mode='full')

    L_diag = boxfilter(window_area - meanVmeansPlusOne, radius, mode='full')
    temp = 2*boxfilter(Vmeans, radius, mode='full')
    temp -= mat_vec_dot(boxfilter(inv_covs, radius, mode='full'), I)
    L_diag += vec_vec_dot(I, temp)
    L_diag /= window_area

    L_diag = L_diag.flatten()

    def L_dot(p):
        p = p.reshape(h, w)

        pr = boxfilter(p*r, radius, mode='valid')
        pg = boxfilter(p*g, radius, mode='valid')
        pb = boxfilter(p*b, radius, mode='valid')
        
        p_means = boxfilter(p, radius, mode='valid')

        temp = p_means*meanVmeansPlusOne
        temp -= pr*Vmeans[:, :, 0]
        temp -= pg*Vmeans[:, :, 1]
        temp -= pb*Vmeans[:, :, 2]
        Lp = boxfilter(temp, radius, mode='full')
        
        Lp += r*boxfilter(inv_covs[:, :, 0, 0]*pr + inv_covs[:, :, 0, 1]*pg + inv_covs[:, :, 0, 2]*pb - p_means*Vmeans[:, :, 0], radius, mode='full')
        Lp += g*boxfilter(inv_covs[:, :, 1, 0]*pr + inv_covs[:, :, 1, 1]*pg + inv_covs[:, :, 1, 2]*pb - p_means*Vmeans[:, :, 1], radius, mode='full')
        Lp += b*boxfilter(inv_covs[:, :, 2, 0]*pr + inv_covs[:, :, 2, 1]*pg + inv_covs[:, :, 2, 2]*pb - p_means*Vmeans[:, :, 2], radius, mode='full')
        
        Lp += p*filterweights
        Lp *= -1/window_area
        
        return Lp.flatten()

    L = scipy.sparse.linalg.LinearOperator(matvec=L_dot, shape=(n, n))

    return L, L_diag

