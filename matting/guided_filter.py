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

def fast_guided_filter(I0, p0, r0, eps=1e-4, scale=0.125):
    h0,w0 = I0.shape[:2]
    
    r = int(r0*scale)
    w = int(w0*scale)
    h = int(h0*scale)
    
    I = resize_nearest(I0, w, h)
    p = resize_nearest(p0, w, h)
    
    a,b = local_model(I, p, r, eps)
    
    a = resize_bilinear(a, w0 - 2*r0, h0 - 2*r0)
    b = resize_bilinear(b, w0 - 2*r0, h0 - 2*r0)

    c = blur_full(np.ones_like(b), r0)
    
    return ((blur_full(a, r0)*I0).sum(axis=2) + blur_full(b, r0))/c
