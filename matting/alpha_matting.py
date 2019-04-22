from .util import make_system, solve_cg
from .closed_form_laplacian import closed_form_laplacian
from .knn_laplacian import knn_laplacian
from .ichol import ichol, ichol_solve
from .lkm import make_lkm_operators
from .ifm_matting import ifm_system
from .vcycle import vcycle
import numpy as np
import scipy.sparse.linalg

def alpha_matting(
    image,
    trimap,
    method="vcycle",
    cf_preconditioner="ichol",
    ichol_regularization=0.0,
    ichol_threshold=1e-4,
    lkm_radius=10,
    lambd=100.0,
    epsilon=1e-7,
    max_iterations=2000,
    relative_tolerance=None,
    absolute_tolerance=None,
    callback=None,
    x0=None,
    print_info=False,
):
    r"""
    Propagates approximate alpha values of trimap into unknown regions
    based on image color.
    A system of linear equations is assembled and then solved with
    conjugate gradient descent.
    To accelerate convergence, an incomplete Cholesky preconditioner is
    used.
    The effectiveness of this preconditioner can be controlled with the
    "ichol_*" parameters.
    
    The information flow alpha matting method is provided for academic use only.
    If you use the information flow alpha matting method for an academic
    publication, please cite corresponding publications referenced in the
    description of each function, as well as this toolbox itself:
    
    @INPROCEEDINGS{ifm,
    author={Aksoy, Ya\u{g}{\i}z and Ayd{\i}n, Tun\c{c} Ozan and Pollefeys, Marc}, 
    booktitle={Proc. CVPR}, 
    title={Designing Effective Inter-Pixel Information Flow for Natural Image Matting}, 
    year={2017}, 
    }
    
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
    
    Vcycle matting based on:
        Lee, Philip G., and Ying Wu.
        "Scalable matting: A sub-linear approach."
        arXiv preprint arXiv:1404.3933 (2014).
    
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
            "cf"
            "knn"
            "lkm"
            "ifm"
            "vcycle"
    cf_preconditioner: string
        Possible preconditioners are:
            None
            "jacobi"
            "ichol"
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
    max_iterations: int
        Maximum number of iterations of conjugate gradient descent.
    relative_tolerance: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < relative_tolerance norm(b).
        If either relative_tolerance or absolute_tolerance is None,
        the other is set to 0.0.
    absolute_tolerance: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < absolute_tolerance.
        The default value is
        0.1/(width * height).
    callback: func(A, x, b)
        callback to inspect temporary result after each iteration.
    x0: np.ndarray of dtype np.float64
        Initial guess for alpha matte.
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
    
    if relative_tolerance is None:
        if absolute_tolerance is None:
            relative_tolerance = 0.0
            absolute_tolerance = 0.1/(image.shape[0] * image.shape[1])
        else:
            relative_tolerance = 0.0
    else:
        if absolute_tolerance is None:
            absolute_tolerance = 0.0
    
    methods = ["cf", "knn", "lkm", "ifm", "vcycle"]
    
    if method == "cf":
        L = closed_form_laplacian(image, epsilon)
        
        A, b = make_system(L, trimap, lambd)
        
        if cf_preconditioner is None:
            def precondition(r):
                return r
        
        elif cf_preconditioner == "ichol":
            params = [
                (ichol_regularization, ichol_threshold),
                (1e-4, 1e-4),
                (0.0, 1e-5),
                (1e-4, 1e-5),
                (0.0, 0.0),
            ]
            
            for ichol_regularization, ichol_threshold in params:
                try:
                    A_regularized = A if ichol_regularization == 0.0 else \
                        A + ichol_regularization*scipy.sparse.identity(A.shape[0])
                    
                    L = ichol(A_regularized.tocsc(), ichol_threshold)
                    break
                except ValueError as e:
                    print("""WARNING:
Incomplete Cholesky decomposition failed (%s) with:
    ichol_regularization = %f
    ichol_threshold = %f
    
A smaller value for ichol_threshold might help if sufficient memory is available.
A larger value for ichol_threshold might help if more time is available.

See help of matting_closed_form for more info.
"""%(e, ichol_regularization, ichol_threshold))
            
            def precondition(r):
                return ichol_solve(L, r)
        
        elif cf_preconditioner == "jacobi":
            inv_diag = 1/A.diagonal()
            
            def precondition(r):
                return r * inv_diag

        else:
            raise ValueError('cf_preconditioner must be None, "jacobi" or "ichol", but not %s'%cf_preconditioner)

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
    
    elif method == "vcycle":
        cache = {}
        
        L = closed_form_laplacian(image, epsilon)
        
        A, b = make_system(L, trimap, lambd)

        def precondition(r):
            return vcycle(A, r, trimap.shape, cache)

    else:
        raise ValueError("Invalid matting method: %s\nValid methods are:\n%s"%(
            method,
            "\n".join("    " + method for method in methods)))
    
    x = solve_cg(
        A,
        b,
        x0=x0,
        max_iter=max_iterations,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        precondition=precondition,
        print_info=print_info,
        callback=callback)
    
    alpha = np.clip(x, 0, 1).reshape(trimap.shape)
    
    return alpha
