from .util import make_windows, pad
import numpy as np
import scipy.sparse

def estimate_foreground_background_sampling(
    image,
    alpha,
    num_iterations=20,
    print_info=False
):
    is_fg = alpha > 0.9
    is_bg = alpha < 0.1

    true_foreground = image[is_fg]
    true_background = image[is_bg]

    alpha = alpha[:, :, np.newaxis]

    h,w,d = image.shape

    cost = np.full((h, w), np.inf)
    
    foreground = np.zeros((h, w, 3))
    background = np.zeros((h, w, 3))

    def improve(new_foreground, new_background):
        difference = \
                 alpha *new_foreground + \
            (1 - alpha)*new_background - image
        
        new_cost = np.sum(difference*difference, axis=2)
        
        improved = new_cost < cost
        
        cost      [improved] = new_cost      [improved]
        foreground[improved] = new_foreground[improved]
        background[improved] = new_background[improved]

    for iteration in range(num_iterations):
        # improve randomly
        F_indices = np.random.randint(len(true_foreground), size=(h, w))
        B_indices = np.random.randint(len(true_background), size=(h, w))
        
        new_foreground = true_foreground[F_indices]
        new_background = true_background[B_indices]
        
        new_foreground[is_fg] = image[is_fg]
        new_background[is_bg] = image[is_bg]
        
        improve(new_foreground, new_background)

        # improve with neighbor values
        foreground_neighbors = make_windows(pad(foreground))
        background_neighbors = make_windows(pad(background))
        
        for i in range(9):
            new_foreground = foreground_neighbors[:, :, i, :]
            new_background = background_neighbors[:, :, i, :]
            
            new_foreground[is_fg] = image[is_fg]
            new_background[is_bg] = image[is_bg]
            
            improve(new_foreground, new_background)

        if print_info:
            print("iteration %2d/%2d - error %f"%(
                iteration + 1, num_iterations, cost.mean()))

    foreground = np.clip(foreground, 0, 1)
    background = np.clip(background, 0, 1)

    return foreground, background

def lstsq(A, b, num_iterations=1000, tolerance=1e-10, print_info=False):
    x = np.zeros(A.shape[1])

    AT = A.T
    b = AT.dot(b)

    residual = b - AT.dot(A.dot(x))
    p = residual
    residual_old = np.sum(residual**2)

    for i in range(num_iterations):
        q = AT.dot(A.dot(p))
        alpha = residual_old / np.sum(p*q)

        x += alpha*p
        residual -= alpha*q
        
        residual_new = np.inner(residual, residual)

        if print_info:
            print("%05d/%05d - %e"%(i, num_iterations, residual_new))

        if residual_new < tolerance:
            if print_info:
                print("break after %d iterations"%i)
            break

        p = residual + residual_new/residual_old * p
        
        residual_old = residual_new
    
    return x

def estimate_foreground_background_cf(
    image,
    alpha,
    print_info=False,
    **kwargs
):
    """
    Based on:
        Levin, Anat, Dani Lischinski, and Yair Weiss.
        "A closed-form solution to natural image matting."
        IEEE transactions on pattern analysis and machine intelligence
        30.2 (2008): 228-242.
    """
    
    height, width, depth = image.shape
    n = width*height

    alpha = alpha[:, :, np.newaxis]
    alpha_flat = alpha.flatten()

    def idx(y, x):
        return x + y*width

    d = np.ones((height, width))
    d[:, 0] = 0
    d[0, :] = 0
    d = d.flatten()

    D = scipy.sparse.spdiags(d, 0, n, n)
    Dx = scipy.sparse.spdiags(-d[-idx(0, -1):], idx(0, -1), n, n)
    Dy = scipy.sparse.spdiags(-d[-idx(-1, 0):], idx(-1, 0), n, n)

    W0 = scipy.sparse.spdiags(np.sqrt(np.abs((D + Dx).dot(alpha_flat))), 0, n, n)
    W1 = scipy.sparse.spdiags(np.sqrt(np.abs((D + Dy).dot(alpha_flat))), 0, n, n)

    foreground = np.zeros(image.shape)
    background = np.zeros(image.shape)

    for i in range(3):
        if print_info:
            print("Solving channel %d of 3"%(i + 1))
        
        color = image[:, :, i].flatten()

        alpha0 = scipy.sparse.spdiags(0 + alpha_flat, 0, n, n)
        alpha1 = scipy.sparse.spdiags(1 - alpha_flat, 0, n, n)

        A = scipy.sparse.bmat([
            [alpha0, alpha1],
            [W0.dot(D) + W0.dot(Dx), None],
            [W1.dot(D) + W1.dot(Dy), None],
            [None, W0.dot(D) + W0.dot(Dx)],
            [None, W1.dot(D) + W1.dot(Dy)],
        ])

        b = np.concatenate([color, np.zeros(n*4)])

        x = lstsq(A, b, print_info=print_info, **kwargs)

        foreground[:, :, i] = x[:n].reshape((height, width))
        background[:, :, i] = x[n:].reshape((height, width))

    foreground = np.clip(foreground, 0, 1)
    background = np.clip(background, 0, 1)

    return foreground, background

def estimate_foreground_background(
    image,
    alpha,
    method="cf",
    **kwargs
):
    if method == "cf":
        return estimate_foreground_background_cf(image, alpha, **kwargs)
    elif method == "sampling":
        return estimate_foreground_background_sampling(image, alpha, **kwargs)
    else:
        raise Exception("Invalid method %s: expected either cf or sampling"%method)
    
