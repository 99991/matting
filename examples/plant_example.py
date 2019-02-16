from matting import matting, load_image, save_image, blend
import os

# Input paths
image_path  = "plant_image.jpg"
trimap_path = "plant_trimap.png"
new_background_path = "background.png"
# Output paths
alpha_path  = "out/plant_alpha_method.png"
blended_path = "out/plant_new_background_method.png"
os.makedirs("out", exist_ok=True)

# Limit image size to make demo run faster
height = 128

# Load input images
image  = load_image( image_path, "RGB" , "BILINEAR", height=height)
trimap = load_image(trimap_path, "GRAY", "NEAREST" , height=height)

# Load new background image
new_background = load_image(
    path=new_background_path,
    mode="RGB",
    interpolation="BILINEAR",
    width=image.shape[1],
    height=image.shape[0])

# Calculate alpha with various alpha matting methods
for method in [
    "cf",
    "knn",
    "lkm",
    "ifm",
]:
    print("Calculating alpha matte with %s method"%method)
    
    alpha = matting(image, trimap, method)

    # Save alpha
    save_image(alpha_path.replace("method", method), alpha)

    # Compose image onto new background
    image_on_new_background = blend(image, new_background, alpha)

    # Save image on new background
    save_image(blended_path.replace("method", method), image_on_new_background)
