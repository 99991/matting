# Remove the background of an image 

<table>
    <tr>
        <td align="center"><b>Input Image</b></td>
        <td align="center"><b>Input mask</b></td>
        <td align="center"><b>Extracted foreground</b></td>
    </tr>
    <tr>
        <td>
<img src="https://raw.githubusercontent.com/99991/matting/master/examples/plant_image.jpg" width="256" height="256">
        </td>
        <td>
<img src="https://raw.githubusercontent.com/99991/matting/master/examples/plant_trimap.png" width="256" height="256">
        </td>
        <td>
<img src="https://raw.githubusercontent.com/99991/matting/master/examples/out/plant_cutout.png" width="256" height="256">
        </td>
    </tr>
</table>

# Install

Install a C compiler (if you want to use alpha matting methods other than vcycle) and then run:

```
git clone https://github.com/99991/matting.git
cd matting
python setup.py install
cd examples
python simple_example.py
python advanced_example.py
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

# Simple Example

The example files can also be found at  can be found at [https://github.com/99991/matting/tree/master/examples](https://github.com/99991/matting/tree/master/examples).

```python
from matting import alpha_matting, load_image, save_image, estimate_foreground_background, stack_images

image  = load_image("plant_image.jpg", "RGB")
trimap = load_image("plant_trimap.png", "GRAY")

alpha = alpha_matting(image, trimap, method="ifm", preconditioner="jacobi", print_info=True)

foreground, background = estimate_foreground_background(image, alpha, print_info=True)

cutout = stack_images(foreground, alpha)

save_image("out/plant_cutout.png", cutout)
```

# Advanced Example

```python
from matting import alpha_matting, load_image, save_image, stack_images\
    ,estimate_foreground_background, METHODS, PRECONDITIONERS
import os

# Input paths
image_path  = "plant_image.jpg"
trimap_path = "plant_trimap.png"
new_background_path = "background.png"
# Output paths
alpha_path  = "out/plant_alpha_%s_%s.png"
cutout_path  = "out/plant_cutout_%s_%s.png"
os.makedirs("out", exist_ok=True)

# Limit image size to make demo run faster
height = 128

# Load input images
# shape (height, width, 3) of data type numpy.float64 in range [0, 1]
image  = load_image( image_path, "RGB" , "BILINEAR", height=height)
# shape (height, width) of data type numpy.float64 in range [0, 1]
trimap = load_image(trimap_path, "GRAY", "NEAREST" , height=height)

# Calculate alpha with various alpha matting methods and preconditioners
for method in METHODS:
    for preconditioner in PRECONDITIONERS[method]:
        alpha = alpha_matting(image, trimap, method, preconditioner,
            print_info=True)

        # Save alpha
        save_image(alpha_path%(method, preconditioner), alpha)
        
        foreground, background = estimate_foreground_background(
            image, alpha, print_info=True)

        # Make new image from foreground and alpha
        cutout = stack_images(foreground, alpha)

        # Save cutout
        save_image(cutout_path%(method, preconditioner), cutout)
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
