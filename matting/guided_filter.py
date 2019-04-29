from .util import resize_nearest, resize_bilinear, vec_vec_outer
from .boxfilter import boxfilter as _boxfilter
import numpy as np

def blur(image, radius):
    return _boxfilter(image, radius, mode='same') / _boxfilter(np.ones_like(image), radius, mode='same')

def _guided_filter(I, p, r, eps):
    mean_I = blur(I, r)
    mean_p = blur(p, r)
    mean_Ip = blur(I * p[:, :, np.newaxis], r)
    
    cov_Ip = mean_Ip - mean_I * mean_p[:, :, np.newaxis]
    var_I = blur(vec_vec_outer(I, I), r) - vec_vec_outer(mean_I, mean_I)
    
    a = np.linalg.solve(var_I + eps*np.eye(3), cov_Ip)
    b = mean_p - (a * mean_I).sum(axis=2)

    return a, b

def guided_filter(I, p, r, eps):
    a, b = _guided_filter(I, p, r, eps)
    
    return (blur(a, r)*I).sum(axis=2) + blur(b, r)

def fast_guided_filter(I0, p0, r0, eps, scale=0.125):
    h0,w0 = I0.shape[:2]
    
    r = int(r0*scale)
    w = int(w0*scale)
    h = int(h0*scale)
    
    I = resize_nearest(I0, w, h)
    p = resize_nearest(p0, w, h)
    
    a,b = _guided_filter(I, p, r, eps)
    
    a = resize_bilinear(a, w0, h0)
    b = resize_bilinear(b, w0, h0)

    return (blur(a, r)*I0).sum(axis=2) + blur(b, r)
