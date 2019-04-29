from .util import resize_nearest, resize_bilinear, vec_vec_outer
from .boxfilter import boxfilter as _boxfilter
import numpy as np

def blur_valid(image, radius):
    return 1/(2*radius + 1)**2 * _boxfilter(image, radius, mode='valid')

def blur_full(image, radius):
    return 1/(2*radius + 1)**2 * _boxfilter(image, radius, mode='full')

def local_model(I, p, r, eps):
    mean_I = blur_valid(I, r)
    mean_p = blur_valid(p, r)
    mean_pI = blur_valid(p[:, :, np.newaxis] * I, r)
    
    v = mean_pI - mean_I * mean_p[:, :, np.newaxis]
    
    covariance = blur_valid(vec_vec_outer(I, I), r) - vec_vec_outer(mean_I, mean_I)
    
    a = np.linalg.solve(covariance + eps*np.eye(3), v)
    b = mean_p - (a * mean_I).sum(axis=2)

    return a, b

def guided_filter(I, p, r, eps=1e-4):
    a, b = local_model(I, p, r, eps)
    
    c = blur_full(np.ones_like(b), r)
    
    return ((blur_full(a, r)*I).sum(axis=2) + blur_full(b, r))/c

def fast_guided_filter(I, p, r, eps=1e-4, scale=0.125):
    h,w = I.shape[:2]
    
    r_small = int(r*scale)
    w_small = int(w*scale)
    h_small = int(h*scale)
    
    I_small = resize_nearest(I, w_small, h_small)
    p_small = resize_nearest(p, w_small, h_small)
    
    a_small,b_small = local_model(I_small, p_small, r_small, eps)
    
    mean_a_small = blur_full(a_small, r_small)
    mean_b_small = blur_full(b_small, r_small)
    c            = blur_full(np.ones_like(b_small), r_small)
    
    mean_a = resize_bilinear(mean_a_small, w, h)
    mean_b = resize_bilinear(mean_b_small, w, h)
    c      = resize_bilinear(           c, w, h)
    
    return ((mean_a*I).sum(axis=2) + mean_b) / c
