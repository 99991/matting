# Install

Currently, only Linux is supported.

```
git clone https://github.com/99991/matting.git
cd matting
python setup.py install
```

# Example

![image of plant, trimap and plant alpha](https://raw.githubusercontent.com/99991/matting/master/examples/plant_result.jpg)

This example and the corresponding images can be found at `examples/plant_example.py`.

```python
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
height = 512

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
    "vcycle",
]:
    print("Calculating alpha matte with %s method"%method)
    
    alpha = matting(image, trimap, method, print_info=True)

    # Save alpha
    save_image(alpha_path.replace("method", method), alpha)

    # Compose image onto new background
    image_on_new_background = blend(image, new_background, alpha)

    # Save image on new background
    save_image(blended_path.replace("method", method), image_on_new_background)

```

# Citations

## Closed form alpha matting based on
```
Levin, Anat, Dani Lischinski, and Yair Weiss.
"A closed-form solution to natural image matting."
IEEE transactions on pattern analysis and machine intelligence 30.2 (2008): 228-242.
```

## Knn matting based on
```
Q. Chen, D. Li, C.-K. Tang.
"KNN Matting."
Conference on Computer Vision and Pattern Recognition (CVPR), June 2012.
```

## Lkm matting based on
```
He, Kaiming, Jian Sun, and Xiaoou Tang.
"Fast matting using large kernel matting laplacian matrices."
Computer Vision and Pattern Recognition (CVPR),
2010 IEEE Conference on. IEEE, 2010.
```

## Ifm matting based on
```
Aksoy, Yagiz, Tunc Ozan Aydin, and Marc Pollefeys.
"Designing effective inter-pixel information flow for natural image matting."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
```

## Vcycle matting based on
```
Lee, Philip G., and Ying Wu.
"Scalable matting: A sub-linear approach."
arXiv preprint arXiv:1404.3933 (2014).
```
