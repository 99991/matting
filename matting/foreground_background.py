from .util import make_windows, pad
import numpy as np

def estimate_foreground_background(
    image,
    trimap,
    alpha,
    num_iterations=20,
    print_info=False
):
    is_fg = trimap > 0.9
    is_bg = trimap < 0.1

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
        
        improve(
            true_foreground[F_indices],
            true_background[B_indices])

        # improve with neighbor values
        foreground_neighbors = make_windows(pad(foreground))
        background_neighbors = make_windows(pad(background))
        
        for i in range(9):
            improve(
                foreground_neighbors[:, :, i, :],
                background_neighbors[:, :, i, :])

        if print_info:
            print("iteration %2d/%2d - error %f"%(
                iteration + 1, num_iterations, cost.mean()))

    return foreground, background
