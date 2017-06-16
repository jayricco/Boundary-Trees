"""
    Jay Ricco - BoundaryTree
    Introduction to AI, Section: 03
    6/3/17
"""
import numpy as np
from PIL import Image
from kivy.core.image import Texture
from array import array
import copy


class BoundaryTree:

    @staticmethod
    def l2_distance(xy1, xy2):
        return np.sum(np.square(np.subtract(np.asarray(xy1, dtype=np.float64), xy2.transpose())))

    def __init__(self, k, root_data, img_mesh):
        self.max_children = k
        self.data = {0: root_data}
        self.relations = {0: []}
        self.root = 0
        self.count = 1
        self.img_mesh = np.asarray(zip(img_mesh[0].flatten(), img_mesh[1].flatten()))


    def query(self, test_x, internal=False):
        node = self.root
        while True:
            children = self.relations[node]
            if len(children) < self.max_children:
                children = copy.copy(children)
                children.append(node)

            closest_node = min(children,
                               key=lambda child_node: BoundaryTree.l2_distance(self.data[child_node][0], test_x))
            if closest_node == node:
                break
            node = closest_node
        if internal:
            return node
        else:
            return self.data[node][1]

    def train(self, new_x, new_y):
        closest_node = self.query(new_x, internal=True)
        if not np.array_equal(self.data[closest_node][1], new_y):
            self.data[self.count] = (new_x, new_y)
            self.relations[self.count] = []
            self.relations[closest_node].append(self.count)
            self.count += 1

    def toImage(self, width, height):
        arr = array('B')
        map(lambda c: arr.extend(self.query(c)), self.img_mesh)
        img = Image.frombytes("RGB", (width, height), arr)
        return img

    def blit_to_texture(self, texture, img_size, resample_method):
        sz = texture.size
        arr = array('B')
        img = self.toImage(img_size[0], img_size[1])
        img_resize = img.resize(size=sz, resample=resample_method)
        [arr.extend(list(c)) for c in img_resize.getdata()]
        texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')
        return texture
