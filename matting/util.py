import PIL
import PIL.Image
import numpy as np
import scipy.sparse

def load_image(path, mode=None, interpolation=None, width=None, height=None):
    assert(interpolation is not None)
    
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

def trimap_split(trimap, flatten=True):
    if flatten:
        trimap = trimap.flatten()
    
    is_fg = (trimap == 1.0)
    is_bg = (trimap == 0.0)
    is_known = np.logical_or(is_fg, is_bg)
    is_unknown = np.logical_not(is_known)
    
    return is_fg, is_bg, is_known, is_unknown

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

def make_system(L, trimap, lambd):
    is_fg = (trimap == 1.0).flatten()
    is_bg = (trimap == 0.0).flatten()
    is_known = is_fg | is_bg
    is_unknown = ~is_known
    
    if is_fg.sum() == 0:
        raise ValueError("Trimap did not contain background values (values = 0)")
    if is_bg.sum() == 0:
        raise ValueError("Trimap did not contain foreground values (values = 1)")
    
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)
    
    A = lambd*D + L
    b = lambd*is_fg.astype(np.float64)
    
    return A, b

def blend(foreground, background, alpha):
    alpha = alpha[:, :, np.newaxis]
    return foreground*alpha + (1 - alpha)*background

def solve_cg(
    A,
    b,
    rtol,
    max_iter,
    x0=None,
    precondition=None,
    callback=None
):
    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
    
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
        
        if callback is not None:
            callback(x)
        
        if residual_error < rtol*norm_b:
            break

        z = precondition(r)
        beta = 1/rz
        rz = np.inner(r, z)
        beta *= rz
        p *= beta
        p += z

    return x, 0
