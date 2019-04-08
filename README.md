# Output

![image of trimap](https://raw.githubusercontent.com/99991/matting/master/examples/out/plant_new_background.png)

# Input

![image of plant](https://raw.githubusercontent.com/99991/matting/master/examples/plant_image.jpg)
![image of trimap](https://raw.githubusercontent.com/99991/matting/master/examples/plant_trimap.png)

# Install

Install a C compiler (if you want to use alpha matting methods other than vcycle) and then run:

```
git clone https://github.com/99991/matting.git
cd matting
python setup.py install
cd examples
python plant_example.py
```

Replace `python` with `python3` if that is not your default.

## Installing a C compiler

### Linux

```
sudo apt-get install build-essential
```

### Windows

[Install Python and gcc on Windows.](https://github.com/99991/matting/blob/master/docs/INSTALL_WINDOWS.md)

### Mac

Install XCode through the App store.

# Minimal Example

```python
from matting import alpha_matting, load_image, save_image

image = load_image("image.png", mode="RGB")
trimap = load_image("trimap.png", mode="GRAY")

alpha = alpha_matting(image, trimap)

save_image("alpha.png", alpha)
```

# Extended Example

This example and the corresponding images can be found at `examples/plant_example.py`.

```python
from matting import alpha_matting, load_image, save_image, blend
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
# shape (height, width, 3) of data type numpy.float64 in range [0, 1]
image  = load_image( image_path, "RGB" , "BILINEAR", height=height)
# shape (height, width) of data type numpy.float64 in range [0, 1]
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
    
    alpha = alpha_matting(image, trimap, method, print_info=True)

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

The information flow alpha matting method is provided for academic use only.
If you use the information flow alpha matting method for an academic
publication, please cite corresponding publications referenced in the
description of each function:

```
@INPROCEEDINGS{ifm,
author={Aksoy, Ya\u{g}{\i}z and Ayd{\i}n, Tun\c{c} Ozan and Pollefeys, Marc}, 
booktitle={Proc. CVPR}, 
title={Designing Effective Inter-Pixel Information Flow for Natural Image Matting}, 
year={2017}, 
}
```
