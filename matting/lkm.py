import numpy as np
import scipy.sparse
from .boxfilter import boxfilter
from .util import vec_vec_dot, mat_vec_dot, vec_vec_outer

def make_lkm_operators(image, eps=1e-4, radius=2):
    window_size = 2*radius + 1
    window_area = window_size*window_size
    I = image
    h, w, depth = I.shape
    
    # means over neighboring pixels
    means = boxfilter(I, radius, mode='valid')/window_area
    
    # color covariance over neighboring pixels
    covs = boxfilter(vec_vec_outer(I, I), radius, mode='valid')/window_area
    covs -= vec_vec_outer(means, means)
    
    # precompute values which do not depend on p
    V = np.linalg.inv(covs + eps/window_area*np.eye(depth))/window_area
    Vm = mat_vec_dot(V, means)
    mVm = 1/window_area + vec_vec_dot(means, Vm)
    c = boxfilter(np.ones((h - 2*radius, w - 2*radius)), radius, mode='full')

    # compute diagonal of L
    d_L = boxfilter(1.0 - mVm, radius, mode='full')
    temp = 2*boxfilter(Vm, radius, mode='full')
    temp -= mat_vec_dot(boxfilter(V, radius, mode='full'), I)
    d_L += vec_vec_dot(I, temp)

    def L_dot(p):
        p = p.reshape(h, w)
        
        p_sums = boxfilter(p, radius, mode='valid')
        pI_sums = boxfilter(p[:, :, np.newaxis]*I, radius, mode='valid')
        
        p_L = c*p
        
        temp = p_sums[:, :, np.newaxis]*Vm - mat_vec_dot(V, pI_sums)
        p_L += vec_vec_dot(I, boxfilter(temp, radius, mode='full'))
        
        temp = p_sums*mVm - vec_vec_dot(pI_sums, Vm)
        p_L -= boxfilter(temp, radius, mode='full')
        
        return p_L.flatten()

    L = scipy.sparse.linalg.LinearOperator(matvec=L_dot, shape=(w*h, w*h))

    return L, d_L.flatten()

