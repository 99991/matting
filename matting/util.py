import PIL
import PIL.Image
import numpy as np
import scipy.sparse

vec_vec_dot   = lambda a,b: np.einsum("...i,...i->...", a, b)
mat_vec_dot   = lambda A,b: np.einsum("...ij,...j->...i", A, b)
vec_vec_outer = lambda a,b: np.einsum("...i,...j", a, b)

def load_image(path, mode=None, interpolation=None, width=None, height=None):
    if interpolation is None:
        interpolation = "BILINEAR"
    
    interpolation = interpolation.upper()
    
    interpolation = {
        "NEAREST": PIL.Image.NEAREST,
        "BILINEAR": PIL.Image.BILINEAR,
    }[interpolation]
    
    image = PIL.Image.open(path)
    
    if mode is not None:
        if mode is "GRAY":
            mode = "L"
        
        image = image.convert(mode)
    
    if width is not None and height is not None:
        image = image.resize((width, height), interpolation)
    elif width is not None and height is None:
        height = int(image.height*1.0*width/image.width)
        image = image.resize((width, height), interpolation)
    elif width is None and height is not None:
        width = int(image.width*1.0*height/image.height)
        image = image.resize((width, height), interpolation)
    
    return np.array(image)/255.0

def save_image(path, image):
    assert(image.dtype in [np.uint8, np.float32, np.float64])
    
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image*255, 0, 255).astype(np.uint8)
    
    image = PIL.Image.fromarray(image)
    image.save(path)

def weights_to_laplacian(W, normalize=True):
    if normalize:
        # normalize row sum to 1
        W_row_sum = np.array(W.sum(axis=1)).flatten()
        W_row_sum[np.abs(W_row_sum) < 1e-10] = 1e-10
        W = scipy.sparse.diags(1/W_row_sum).dot(W)
        L = scipy.sparse.identity(len(W_row_sum)) - W
    else:
        W_row_sum = np.array(W.sum(axis=1)).flatten()
        L = scipy.sparse.diags(W_row_sum) - W
    return L

def pad(image, r=1):
    # pad by repeating border pixels of image
    
    # create padded result image with same shape as input
    if len(image.shape) == 2:
        height, width = image.shape
        padded = np.zeros((height + 2*r, width + 2*r), dtype=image.dtype)
    else:
        height, width, depth = image.shape
        padded = np.zeros((height + 2*r, width + 2*r, depth), dtype=image.dtype)
    
    # do padding
    if r > 0:
        # top left
        padded[:r, :r] = image[0, 0]
        # bottom right
        padded[-r:, -r:] = image[-1, -1]
        # top right
        padded[:r, -r:] = image[0, -1]
        # bottom left
        padded[-r:, :r] = image[-1, 0]
        # left
        padded[r:-r,  :r] = image[:,  :1]
        # right
        padded[r:-r, -r:] = image[:, -1:]
        # top
        padded[ :r, r:-r] = image[ :1, :]
        # bottom
        padded[-r:, r:-r] = image[-1:, :]
    
    # center
    padded[r:r+height, r:r+width] = image
    
    return padded

def make_windows(image, radius=1):
    return np.stack([image[
        y:y+image.shape[0]-2*radius,
        x:x+image.shape[1]-2*radius]
            for x in range(2*radius+1)
            for y in range(2*radius+1)
        ], axis=2)

def trimap_split(trimap, flatten=True):
    if flatten:
        trimap = trimap.flatten()
    
    is_fg = (trimap == 1.0)
    is_bg = (trimap == 0.0)
    
    is_known = is_fg | is_bg
    is_unknown = ~is_known
    
    if is_fg.sum() == 0:
        raise ValueError("Trimap did not contain background values (values = 0)")
    if is_bg.sum() == 0:
        raise ValueError("Trimap did not contain foreground values (values = 1)")
    
    return is_fg, is_bg, is_known, is_unknown

def make_system(L, trimap, lambd):
    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
    
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)
    
    A = lambd*D + L
    b = lambd*is_fg.astype(np.float64)
    
    return A, b

def blend(foreground, background, alpha):
    alpha = alpha[:, :, np.newaxis]
    return foreground*alpha + (1 - alpha)*background

def stack_images(*images):
    return np.concatenate([
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ], axis=2)

def pixel_coordinates(w, h, flat=False):
    x = np.arange(w)
    y = np.arange(h)
    x,y = np.meshgrid(x, y)
    
    if flat:
        x = x.flatten()
        y = y.flatten()
    
    return x, y

def sparse_conv_matrix(w, h, dx, dy, weights):
    count = len(weights)
    n = w*h
    
    i_inds = np.zeros(n*count, dtype=np.int32)
    j_inds = np.zeros(n*count, dtype=np.int32)
    values = np.zeros(n*count, dtype=np.float64)
    
    k = 0
    x,y = pixel_coordinates(w, h, flat=True)
    for dx2, dy2,weight in zip(dx, dy, weights):
        x2 = np.clip(x + dx2, 0, w - 1)
        y2 = np.clip(y + dy2, 0, h - 1)
        i_inds[k:k+n] = x + y*w
        j_inds[k:k+n] = x2 + y2*w
        values[k:k+n] = weight
        k += n
    
    A = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    
    return A

def uniform_laplacian(w, h, r):
    size = 2*r + 1
    x,y = pixel_coordinates(size, size, flat=True)
    
    W = sparse_conv_matrix(w, h, x - r, y - r, np.ones(size*size))

    d = np.array(W.sum(axis=1)).flatten()
    D = scipy.sparse.diags(d)
    
    L = D - W
    
    return L

def resize_nearest(image, new_width, new_height):
    old_height, old_width = image.shape[:2]
    
    x = np.arange(new_width )[np.newaxis, :]
    y = np.arange(new_height)[:, np.newaxis]
    x = x*old_width /new_width
    y = y*old_height/new_height
    x = np.clip(x.astype(np.int32), 0, old_width  - 1)
    y = np.clip(y.astype(np.int32), 0, old_height - 1)
    
    if len(image.shape) == 3:
        image = image.reshape(-1, image.shape[2])
    else:
        image = image.ravel()
    
    return image[x + y*old_width]

def resize_bilinear(image, new_width, new_height):
    old_height, old_width = image.shape[:2]
    
    x2 = old_width /new_width *(np.arange(new_width ) + 0.5) - 0.5
    y2 = old_height/new_height*(np.arange(new_height) + 0.5) - 0.5
    
    x2 = x2[np.newaxis, :]
    y2 = y2[:, np.newaxis]
    
    x0 = np.floor(x2)
    y0 = np.floor(y2)
    
    ux = x2 - x0
    uy = y2 - y0
    
    x0 = x0.astype(np.int32)
    y0 = y0.astype(np.int32)
    
    x1 = x0 + 1
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, old_width - 1)
    x1 = np.clip(x1, 0, old_width - 1)
    
    y0 = np.clip(y0, 0, old_height - 1)
    y1 = np.clip(y1, 0, old_height - 1)
    
    if len(image.shape) == 3:
        pix = image.reshape(-1, image.shape[2])
        ux = ux[..., np.newaxis]
        uy = uy[..., np.newaxis]
    else:
        pix = image.ravel()
    
    a = (1 - ux)*pix[y0*old_width + x0] + ux*pix[y0*old_width + x1]
    b = (1 - ux)*pix[y1*old_width + x0] + ux*pix[y1*old_width + x1]
    
    return (1 - uy)*a + uy*b

def inv2(mat):
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]
    
    inv_det = 1 / (a*d - b*c)
    
    inv = np.empty(mat.shape)

    inv[..., 0, 0] = inv_det*d
    inv[..., 0, 1] = inv_det*-b
    inv[..., 1, 0] = inv_det*-c
    inv[..., 1, 1] = inv_det*a

    return inv

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

def solve_cg(
    A,
    b,
    rtol,
    max_iter,
    atol=0.0,
    x0=None,
    precondition=None,
    callback=None,
    print_info=False,
):
    """
    Solve the linear system Ax = b for x using preconditioned conjugate
    gradient descent.
    
    A: np.ndarray of dtype np.float64
        Must be a square symmetric matrix
    b: np.ndarray of dtype np.float64
        Right-hand side of linear system
    rtol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < relative_tolerance norm(b)
    max_iter: int
        Maximum number of iterations
    atol: float64
        Conjugate gradient descent will stop when
        norm(A x - b) < absolute_tolerance
    x0: np.ndarray of dtype np.float64
        Initial guess of solution x
    precondition: func(r) -> r
        Improve solution of residual r, for example solve(M, r)
        where M is an easy-to-invert approximation of A.
    callback: func(A, x, b)
        callback to inspect temporary result after each iteration.
    print_info: bool
        If to print convergence information.
    
    Returns
    -------
    
    x: np.ndarray of dtype np.float64
        Solution to the linear system Ax = b.
    
    """
    
    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
    
    if callback is not None:
        callback(A, x, b)
    
    if precondition is None:
        def precondition(r):
            return r
    
    norm_b = np.linalg.norm(b)
    
    r = b - A.dot(x)
    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)
    for iteration in range(max_iter):
        Ap = A.dot(p)
        alpha = rz/np.inner(p, Ap)
        x += alpha*p
        r -= alpha*Ap

        residual_error = np.linalg.norm(r)
        
        if print_info:
            print("iteration %05d - residual error %e"%(
                iteration,
                residual_error))
        
        if callback is not None:
            callback(A, x, b)
        
        if residual_error < atol or residual_error < rtol*norm_b:
            break

        z = precondition(r)
        beta = 1/rz
        rz = np.inner(r, z)
        beta *= rz
        p *= beta
        p += z

    return x
