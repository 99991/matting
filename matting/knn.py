from ctypes import c_int, c_float, pointer, POINTER
import numpy as np
from .load_libmatting import load_libmatting

library = load_libmatting()

# Need those types for function signatures:
c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)

# Declare function signatures
_knn = library.knn
_knn.argtypes = [
    c_float_p,
    c_float_p,
    c_int_p,
    c_float_p,
    c_int,
    c_int,
    c_int,
    c_int]

def knn(data_points, query_points, k, overwrite_data_points=False):
    if data_points.dtype == np.float64:
        data_points = data_points.astype(np.float32)
    
    if query_points.dtype == np.float64:
        query_points = query_points.astype(np.float32)
    
    assert(len(data_points.shape) == 2)
    assert(len(query_points.shape) == 2)
    assert(data_points.dtype == np.float32)
    assert(query_points.dtype == np.float32)
    assert(len(data_points) >= k)
    
    n_data_points, data_points_dim = data_points.shape
    n_query_points, query_points_dim = query_points.shape
    
    assert(data_points_dim == query_points_dim)
    
    indices = np.empty(n_query_points*k, dtype=np.int32)
    squared_distances = np.empty(n_query_points*k, dtype=np.float32)
    
    data_points = data_points.ravel()
    query_points = query_points.ravel()
    
    if not overwrite_data_points:
        data_points = data_points.copy()
    
    assert(data_points.flags['C_CONTIGUOUS'])
    assert(query_points.flags['C_CONTIGUOUS'])
    
    _knn(
        np.ctypeslib.as_ctypes(data_points),
        np.ctypeslib.as_ctypes(query_points),
        np.ctypeslib.as_ctypes(indices),
        np.ctypeslib.as_ctypes(squared_distances),
        n_data_points,
        n_query_points,
        data_points_dim,
        k)
    
    indices = indices.reshape(n_query_points, k)
    squared_distances = squared_distances.reshape(n_query_points, k)
    
    return indices

def test():
    for d in range(1, 10):
        for k in range(1, 5):
            na = np.random.randint(1, 100)
            nb = np.random.randint(1, 100)
            
            if k > na: continue
            
            a = np.random.randn(na, d).astype(np.float32)
            b = np.random.randn(nb, d).astype(np.float32)

            indices1 = knn(a, b, k)
            indices2 = np.empty((nb, k), dtype=np.int32)

            for i in range(nb):
                distances = np.linalg.norm(a - b[i].reshape(1, d), axis=1)
                indices2[i] = np.argsort(distances)[:k]

            assert(indices1.shape == indices2.shape)
            assert(np.allclose(indices1, indices2))
    print("tests passed")

if __name__ == "__main__":
    test()
