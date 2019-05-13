from matting import alpha_matting, load_image, save_image, stack_images,\
    estimate_foreground_background, METHODS, PRECONDITIONERS
import os

# Input paths
image_path = "plant_image.jpg"
trimap_path = "plant_trimap.png"
new_background_path = "background.png"
# Output paths
alpha_path = "out/plant_alpha_%s_%s.png"
cutout_path = "out/plant_cutout_%s_%s.png"
os.makedirs("out", exist_ok=True)

# Limit image size to make demo run faster
height = 128

# Load input images
# shape (height, width, 3) of data type numpy.float64 in range [0, 1]
image = load_image(image_path, "RGB", "BILINEAR", height=height)
# shape (height, width) of data type numpy.float64 in range [0, 1]
trimap = load_image(trimap_path, "GRAY", "NEAREST", height=height)

# Calculate alpha with various alpha matting methods and preconditioners
for method in METHODS:
    for preconditioner in PRECONDITIONERS[method]:
        alpha = alpha_matting(
            image, trimap,
            method, preconditioner,
            print_info=True)

        # Save alpha
        save_image(alpha_path % (method, preconditioner), alpha)

        foreground, background = estimate_foreground_background(
            image, alpha, print_info=True)

        # Make new image from foreground and alpha
        cutout = stack_images(foreground, alpha)

        # Save cutout
        save_image(cutout_path % (method, preconditioner), cutout)
