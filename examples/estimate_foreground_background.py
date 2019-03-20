from matting import alpha_matting, blend, estimate_foreground_background
from matting import load_image, save_image

image_path  = "plant_image.jpg"
trimap_path = "plant_trimap.png"
new_background_path = "background.png"
plant_new_background = "out/plant_new_background.png"

image  = load_image( image_path, "RGB")
trimap = load_image(trimap_path, "GRAY")

new_background = load_image(
    path=new_background_path,
    mode="RGB",
    interpolation="BILINEAR",
    width=image.shape[1],
    height=image.shape[0])

alpha = alpha_matting(image, trimap, print_info=True)

method = "cf"
method = "sampling"

foreground, background = estimate_foreground_background(
    image, alpha, method=method, print_info=True)

image_on_new_background = blend(foreground, new_background, alpha)

save_image(plant_new_background, image_on_new_background)


