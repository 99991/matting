import numpy as np
from ctypes import CDLL, c_int, c_double, POINTER
from .load_libmatting import load_libmatting

library = load_libmatting()

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

_boxfilter_valid = library.boxfilter_valid
_boxfilter_valid.argtypes = [
    c_double_p,
    c_double_p,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int]

_boxfilter_full = library.boxfilter_full
_boxfilter_full.argtypes = [
    c_double_p,
    c_double_p,
    c_int,
    c_int,
    c_int]

def boxfilter_valid(src, r):
    assert(src.flags["C_CONTIGUOUS"])
    assert(src.flags["ALIGNED"])
    assert(src.dtype == np.float64)
    
    h,w = src.shape
    
    dst = np.zeros((h - 2*r, w - 2*r), dtype=np.float64)
    
    _boxfilter_valid(
        np.ctypeslib.as_ctypes(src.ravel()),
        np.ctypeslib.as_ctypes(dst.ravel()),
        w,
        w - 2*r,
        w,
        h,
        r)
    
    return dst

def boxfilter_full(src, r):
    assert(src.flags["C_CONTIGUOUS"])
    assert(src.flags["ALIGNED"])
    assert(src.dtype == np.float64)
    
    h,w = src.shape
    
    dst = np.zeros((h + 2*r, w + 2*r), dtype=np.float64)
    
    _boxfilter_full(
        np.ctypeslib.as_ctypes(src.ravel()),
        np.ctypeslib.as_ctypes(dst.ravel()),
        w,
        h,
        r)
    
    return dst

def apply_to_channels(single_channel_func):
    def multi_channel_func(image, *args, **kwargs):
        if len(image.shape) == 2:
            return single_channel_func(image, *args, **kwargs)
        else:
            shape = image.shape
            image = image.reshape(shape[0], shape[1], -1)
            
            result = np.stack([
                single_channel_func(image[:, :, c].copy(), *args, **kwargs)
                for c in range(image.shape[2])], axis=2)
            
            return result.reshape(list(result.shape[:2]) + list(shape[2:]))
            
    return multi_channel_func

@apply_to_channels
def boxfilter(src, r, mode):
    if mode == "valid":
        return boxfilter_valid(src, r)
    elif mode == "full":
        return boxfilter_full(src, r)
    else:
        raise ValueError('boxfilter mode must be "valid" of "full", not "%s"'%mode)
