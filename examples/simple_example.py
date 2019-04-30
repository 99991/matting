from matting import alpha_matting, load_image, save_image, estimate_foreground_background, stack_images

image  = load_image("plant_image.jpg", "RGB")
trimap = load_image("plant_trimap.png", "GRAY")

alpha = alpha_matting(image, trimap, method="ifm", preconditioner="jacobi", print_info=True)

foreground, background = estimate_foreground_background(image, alpha, print_info=True)

cutout = stack_images(foreground, alpha)

save_image("out/plant_cutout.png", cutout)
