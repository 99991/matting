from ctypes import c_int, c_double, c_void_p, pointer, POINTER
import numpy as np
import scipy.sparse
from .load_libmatting import load_libmatting

library = load_libmatting()

# Need those types for function signatures:
c_int_p = POINTER(c_int)
c_int_pp = POINTER(c_int_p)
c_double_p = POINTER(c_double)
c_double_pp = POINTER(c_double_p)

# Declare function signatures
_ichol_free = library.ichol_free
_ichol_free.argtypes = [c_void_p]

_ichol = library.ichol
_ichol.restype = c_int
_ichol.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_pp,
    c_int_pp,
    c_int_pp,
    c_int,
    c_double]

_backsub_L_csc_inplace = library.backsub_L_csc_inplace
_backsub_L_csc_inplace.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int]

_backsub_LT_csc_inplace = library.backsub_LT_csc_inplace
_backsub_LT_csc_inplace.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int]

_backsub_L_csr_inplace = library.backsub_L_csr_inplace
_backsub_L_csr_inplace.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int]


def backsub_L_csc_inplace(L, x):
    n = len(x)
    _backsub_L_csc_inplace(
        np.ctypeslib.as_ctypes(L.data),
        np.ctypeslib.as_ctypes(L.indices),
        np.ctypeslib.as_ctypes(L.indptr),
        np.ctypeslib.as_ctypes(x),
        n)


def backsub_LT_csc_inplace(L, x):
    n = len(x)
    _backsub_LT_csc_inplace(
        np.ctypeslib.as_ctypes(L.data),
        np.ctypeslib.as_ctypes(L.indices),
        np.ctypeslib.as_ctypes(L.indptr),
        np.ctypeslib.as_ctypes(x),
        n)


def backsub_L_csr_inplace(L, x):
    assert(isinstance(L, scipy.sparse.csr.csr_matrix))
    n = len(x)
    _backsub_L_csr_inplace(
        np.ctypeslib.as_ctypes(L.data),
        np.ctypeslib.as_ctypes(L.indices),
        np.ctypeslib.as_ctypes(L.indptr),
        np.ctypeslib.as_ctypes(x),
        n)


def backsub_L_csr(L, b):
    x = b.copy()
    backsub_L_csr_inplace(L, x)
    return x


def ichol_solve_inplace(L, x):
    backsub_L_csc_inplace(L, x)
    backsub_LT_csc_inplace(L, x)


def ichol_solve(L, b):
    x = b.copy()
    ichol_solve_inplace(L, x)
    return x


def ichol(A, threshold):
    if not isinstance(A, scipy.sparse.csc.csc_matrix):
        raise ValueError("Matrix A must be of type scipy.sparse.csc.csc_matrix")

    n = A.shape[0]

    # Result pointers
    L_data_ptr = (c_double_p)()
    L_indices_ptr = (c_int_p)()
    L_indptr_ptr = (c_int_p)()

    # Call C ichol function
    err = _ichol(
        np.ctypeslib.as_ctypes(A.data),
        np.ctypeslib.as_ctypes(A.indices),
        np.ctypeslib.as_ctypes(A.indptr),
        pointer(L_data_ptr),
        pointer(L_indices_ptr),
        pointer(L_indptr_ptr),
        n,
        threshold)

    if err == -1:
        raise ValueError("ichol failed because matrix A was not positive definite enough for threshold")
    elif err == -2:
        raise ValueError("ichol failed (out of memory)")

    L_indptr = np.ctypeslib.as_array(L_indptr_ptr, shape=(n + 1,))
    L_nnz = L_indptr[n]
    L_indices = np.ctypeslib.as_array(L_indices_ptr, shape=(L_nnz,))
    L_data = np.ctypeslib.as_array(L_data_ptr, shape=(L_nnz,))

    # Copy array to make it numpy object with memory ownership
    L_indptr = L_indptr.copy()
    L_indices = L_indices.copy()
    L_data = L_data.copy()

    # Free C-allocated memory.
    _ichol_free(L_indptr_ptr)
    _ichol_free(L_indices_ptr)
    _ichol_free(L_data_ptr)

    L = scipy.sparse.csc_matrix((L_data, L_indices, L_indptr), shape=(n, n))

    return L


if __name__ == "__main__":
    for n in [10, 100, 5, 4, 3, 2, 1]:
        A = np.random.rand(n, n)
        # Make positive definite matrix.
        A = A + A.T + n * np.eye(n)
        A = scipy.sparse.csc_matrix(A)

        L = ichol(A, 0.0)

        assert(abs(L.dot(L.T) - A).max() < 1e-10)

        import scipy.sparse.linalg
        b = np.random.rand(n)
        x_true = scipy.sparse.linalg.spsolve(A, b)
        x = b.copy()
        ichol_solve_inplace(L, x)

        assert(np.max(np.abs(x - x_true)) < 1e-10)

        L = np.random.rand(n, n)
        L[np.abs(L) < 0.9] = 0
        L = L + n * np.eye(n)
        L = np.tril(L)
        L = scipy.sparse.csr_matrix(L)

        x_true = scipy.sparse.linalg.spsolve(L, b)
        x = backsub_L_csr(L, b)
        assert(np.max(np.abs(x_true - x)) < 1e-10)
    print("tests passed")
