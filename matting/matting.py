from .util import make_system, solve_cg
from .closed_form_laplacian import closed_form_laplacian
from .knn_laplacian import knn_laplacian
from .ichol import ichol, ichol_solve
from .lkm import make_lkm_operators
from .matting_ifm import ifm_system
import numpy as np
import scipy.sparse.linalg

def matting(
    image,
    trimap,
    method,
    ichol_regularization=0.0,
    ichol_threshold=1e-4,
    lkm_radius=10,
    lambd=100.0,
    epsilon=1e-7,
    max_iter=2000,
    relative_tolerance=1e-6,
    print_info=False,
):
    """
    Closed form (cf) matting based on:
        Levin, Anat, Dani Lischinski, and Yair Weiss.
        "A closed-form solution to natural image matting."
        IEEE transactions on pattern analysis and machine intelligence
        30.2 (2008): 228-242.
    
    K-nearest neighbors (knn) matting based on:
        Q. Chen, D. Li, C.-K. Tang.
        "KNN Matting."
        Conference on Computer Vision and Pattern Recognition (CVPR), June 2012.
    
    Large kernel matting (lkm) based on:
        He, Kaiming, Jian Sun, and Xiaoou Tang.
        "Fast matting using large kernel matting laplacian matrices."
        Computer Vision and Pattern Recognition (CVPR),
        2010 IEEE Conference on. IEEE, 2010.
    
    Information flow matting (ifm) based on:
        Aksoy, Yagiz, Tunc Ozan Aydin, and Marc Pollefeys.
        "Designing effective inter-pixel information flow for natural image matting."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
    
    Propagates approximate alpha values of trimap into unknown regions
    based on image color.
    A system of linear equations is assembled and then solved with
    conjugate gradient descent.
    To accelerate convergence, an incomplete Cholesky preconditioner is
    used.
    The effectiveness of this preconditioner can be controlled with the
    "ichol_*" parameters.
    
    Parameters
    ----------
    image: ndarray, dtype float64, shape [height, width, 3]
        Values should be in range [0 - 1]
    trimap: ndarray, dtype float64, shape [height, width]
        Values should be
            0 for background,
            1 for foreground, and
            other for unknown.
    method: string
        Possible methods are:
            "closed_form"
            "knn"
    ichol_regularization: float64
        Increase to increase probability that incomplete
        Cholesky decomposition can be built successfully.
        Increasing regularization decreases convergence rate.
    ichol_threshold: float64
        Increase to discard more values of incomplete Cholesky
        decomposition.
        Leads to faster build times and lower memory use,
        but decreases convergence rate and decomposition might fail.
    lkm_radius: int
        Radius for matting kernel used in lkm matting.
        Converges faster with larger radius, but result is more blurry.
    lambd: float64
        Weighting factor to constrain known trimap values.
    epsilon: float64
        Regularization factor for closed-form matting.
        Larger values lead to faster convergence but more blurry alpha.
    max_iter: int
        Maximum number of iterations of conjugate gradient descent.
    relative_tolerance: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < relative_tolerance norm(b)
    print_info: bool
        If to print convergence information.
        
    Returns
    -------
    alpha: ndarray, dtype float64, shape [height, width]
    """
    
    assert(image.dtype == np.float64)
    assert(trimap.dtype == np.float64)
    assert(len(image.shape) == 3)
    assert(image.shape[2] == 3)
    assert(len(trimap.shape) == 2)
    assert(image.shape[:2] == trimap.shape)
    assert(0 <= image.min() and image.max() <= 1)
    assert(0 <= trimap.min() and trimap.max() <= 1)
    
    methods = ["cf", "knn", "lkm", "ifm"]
    
    if method == "cf":
        L = closed_form_laplacian(image, epsilon)
        
        A, b = make_system(L, trimap, lambd)
        
        while True:
            try:
                L = ichol(A.tocsc(), ichol_threshold)
                break
            except ValueError as e:
                if ichol_regularization <= 0:
                    ichol_regularization = 1e-4
                else:
                    ichol_regularization *= 5
                
                print("""WARNING: ichol failed (%s).
    Retrying with ichol_regularization = %f.
    See help of matting_closed_form for more info."""%(e, ichol_regularization))
        
        def precondition(r):
            return ichol_solve(L, r)
    
    elif method == "knn":
        L = knn_laplacian(image)
        
        A, b = make_system(L, trimap, lambd)
        
        inv_diag_A = 1/A.diagonal()
        
        def precondition(r):
            return r * inv_diag_A
    
    elif method == "lkm":
        L, L_diag = make_lkm_operators(
            image,
            radius=lkm_radius,
            eps=epsilon)
        
        from .util import trimap_split
        is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
        
        d = lambd*is_known.astype(np.float64)
        
        inv_A_diag = 1/(L_diag + d)
        
        def A_dot(x):
            return L @ x + d * x
        
        n = len(d)
        
        A = scipy.sparse.linalg.LinearOperator(matvec=A_dot, shape=(n, n))
        
        b = lambd*is_fg.astype(np.float64)
        
        def precondition(r):
            return r * inv_A_diag
    
    elif method == "ifm":
        A,b = ifm_system(image, trimap)
        
        def precondition(r):
            return r
    else:
        raise ValueError("Invalid matting method: %s\nValid methods are:\n%s"%(
            method,
            "\n".join("    " + method for method in methods)))
    
    x, info = solve_cg(
        A,
        b,
        max_iter=max_iter,
        rtol=relative_tolerance,
        precondition=precondition,
        print_info=print_info)
    
    alpha = np.clip(x, 0, 1).reshape(trimap.shape)
    
    return alpha
