from .util import solve_cg, vec_vec_outer, pixel_coordinates, inv2
from .util import resize_nearest, sparse_conv_matrix, uniform_laplacian
from .ichol import ichol, ichol_solve
import numpy as np
import scipy.sparse

def estimate_fb_cf(
    image,
    alpha,
    ichol_threshold = 1e-4,
    regularization = 1e-5,
    print_info = False,
):
    """
    Estimate foreground and background of an image using a closed form approach.
    
    Based on:
        Levin, Anat, Dani Lischinski, and Yair Weiss.
        "A closed-form solution to natural image matting."
        IEEE transactions on pattern analysis and machine intelligence
        30.2 (2008): 228-242.
    
    ichol_threshold: float64
        Incomplete Cholesky decomposition threshold.
    
    regularization: float64
        Smoothing factor for undefined foreground/background regions.
    
    print_info:
        Wheter to print debug information during iterations.
    
    Returns
    -------
    
    foreground: np.ndarray of dtype np.float64
        Foreground image.
    
    background: np.ndarray of dtype np.float64
        Background image.
    """
    
    h,w = image.shape[:2]
    n = w*h

    a = alpha.flatten()
    
    # Build sparse linear equation system
    AF = scipy.sparse.diags(a)
    AB = scipy.sparse.diags(1 - a)
    AA = scipy.sparse.bmat([[AF, AB]])
    
    Dx = sparse_conv_matrix(w, h, [1, 0], [0, 0], [0.5, -0.5])
    Dy = sparse_conv_matrix(w, h, [0, 0], [1, 0], [0.5, -0.5])
    
    Dax = scipy.sparse.diags(np.abs(Dx @ a))
    Day = scipy.sparse.diags(np.abs(Dy @ a))
    
    Dxy = Dx.T @ Dax @ Dx + Dy.T @ Day @ Dy
    
    Dxy2 = scipy.sparse.bmat([
        [Dxy, None],
        [None, Dxy],
    ])
    
    L = uniform_laplacian(w, h, 1)
    
    AD = Dxy + regularization*L
    
    A = scipy.sparse.bmat([
        [AF*AF + AD, AF*AB],
        [AF*AB, AB*AB + AD],
    ]).tocsc()
    
    if print_info:
        print("computing incomplete Cholesky decomposition")
    
    # Build incomplete Cholesky decomposition
    L_ichol = ichol(A, ichol_threshold)
    
    if print_info:
        print("incomplete Cholesky decomposition computed")
    
    # Use incomplete Cholesky decomposition as preconditioner
    def precondition(x):
        return ichol_solve(L_ichol, x)
    
    foreground = np.zeros((h, w, 3))
    background = np.zeros((h, w, 3))
    
    # For each color channel
    for channel in range(3):
        if print_info:
            print("solving channel %d"%(1 + channel))
        
        I = image[:, :, channel].flatten()
        
        b = AA.T @ I
        
        # Solve large sparse linear equation system
        fb = solve_cg(A, b, precondition=precondition, max_iter=10000, atol=1e-6, rtol=0, print_info=print_info)
        
        foreground[:, :, channel] = fb[:n].reshape(h, w)
        background[:, :, channel] = fb[n:].reshape(h, w)
    
    foreground = np.clip(foreground, 0, 1)
    background = np.clip(background, 0, 1)
    
    return foreground, background

def estimate_fb_ml(
    input_image,
    input_alpha,
    min_size = 2,
    growth_factor = 2,
    regularization = 1e-5,
    n_iter_func = lambda w,h: 5 if max(w, h) <= 64 else 1,
    print_info = False,
):
    """
    Estimate foreground and background of an image using a multilevel
    approach.
    
    min_size: int > 0
        Minimum image size at which to start solving.
    
    growth_factor: float64 > 1.0
        Image size is increased by growth_factor each level.
    
    regularization: float64
        Smoothing factor for undefined foreground/background regions.
    
    n_iter_func: func(width: int, height: int) -> int
        How many iterations to perform at a given image size.
    
    print_info:
        Wheter to print debug information during iterations.
    
    Returns
    -------
    
    F: np.ndarray of dtype np.float64
        Foreground image.
    
    B: np.ndarray of dtype np.float64
        Background image.
    """
    
    assert(min_size >= 1)
    assert(growth_factor > 1.0)
    h0,w0 = input_image.shape[:2]
    
    if print_info:
        print("Solving for foreground and background using multilevel method")
    
    # Find initial image size.
    if w0 < h0:
        w = min_size
        h = int(min_size*h0/w0)
    else:
        w = int(min_size*w0/h0)
        h = min_size
    
    if print_info:
        print("Initial size: %d-by-%d"%(w, h))
    
    # Generate initial foreground and background from input image
    F = resize_nearest(input_image, w, h)
    B = F.copy()
    
    while True:
        if print_info:
            print("New level of size: %d-by-%d"%(w, h))
        
        # Resize image and alpha to size of current level
        image = resize_nearest(input_image, w, h)
        alpha = resize_nearest(input_alpha, w, h)
        
        # Iterate a few times
        n_iter = n_iter_func(w, h)
        for iteration in range(n_iter):
            if print_info:
                print("Iteration %d of %d"%(iteration + 1, n_iter))
            
            x,y = pixel_coordinates(w, h, flat=True)
            
            # Make alpha into a vector
            a = alpha.reshape(w*h)
            
            # Build system of linear equations
            A = np.stack([a, 1 - a], axis=1)
            mat = vec_vec_outer(A, A)
            rhs = vec_vec_outer(A, image.reshape(w*h, 3))
        
            # For each neighbor
            for dx,dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x2 = np.clip(x + dx, 0, w - 1)
                y2 = np.clip(y + dy, 0, h - 1)
                
                # Vectorized neighbor coordinates
                j = x2 + y2*w
                
                # Gradient of alpha
                da = regularization + np.abs(a - a[j])
                
                # Update matrix of linear equation system
                mat[:, 0, 0] += da
                mat[:, 1, 1] += da
                
                # Update rhs of linear equation system
                rhs[:, 0, :] += da.reshape(w*h, 1) * F.reshape(w*h, 3)[j]
                rhs[:, 1, :] += da.reshape(w*h, 1) * B.reshape(w*h, 3)[j]
            
            # Solve linear equation system for foreground and background
            fb = np.clip(np.matmul(inv2(mat), rhs), 0, 1)
            
            F = fb[:, 0, :].reshape(h, w, 3)
            B = fb[:, 1, :].reshape(h, w, 3)
        
        # If original image size is reached, return result
        if w >= w0 and h >= h0: return F, B
        
        # Grow image size to next level
        w = min(w0, int(w*growth_factor))
        h = min(h0, int(h*growth_factor))
        
        F = resize_nearest(F, w, h)
        B = resize_nearest(B, w, h)

def estimate_foreground_background(
    image,
    alpha,
    method="ml",
    **kwargs
):
    if method == "cf":
        return estimate_fb_cf(image, alpha, **kwargs)
    elif method == "ml":
        return estimate_fb_ml(image, alpha, **kwargs)
    else:
        raise Exception("Invalid method %s: expected either cf or sampling"%method)
    
