from PIL import Image
from array import array
import numpy as np


def boundary_tree_to_image(boundary_tree, size, image_mesh):
    arr = array('B')
    np.apply_along_axis(lambda c: arr.extend(boundary_tree.query(c)), 1, image_mesh)
    return Image.frombytes("RGB", size, arr)


def boundary_tree_accuracy(boundary_tree, gt_image, img_mesh):
    im = boundary_tree_to_image(boundary_tree, gt_image.size, img_mesh)
    total = float(len(gt_image.getdata()))
    return ([c == v for (c, v) in zip(gt_image.getdata(), im.getdata())].count(True)/total)*100.0


def boundary_tree_fullness(boundary_tree):
    summ = 0
    count = 0
    for v in boundary_tree.children.values():
        if len(v) > 0:
            summ += float(len(v))/(boundary_tree.max_children-1)
            count += 1
    return summ / count if count != 0 else 1


def boundary_tree_to_texture(boundary_tree, texture, size, img_mesh):
    sz = texture.size
    arr = array('B')
    img = boundary_tree_to_image(boundary_tree, size, img_mesh)
    img_resize = img.resize(size=sz, resample=Image.NEAREST)
    [arr.extend(list(c)) for c in img_resize.getdata()]
    texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')
    return texture
