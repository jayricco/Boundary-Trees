"""
    Jay Ricco - BoundaryForest
    Introduction to AI, Section: 03
    6/4/17
"""
import numpy as np
from BoundaryTree import BoundaryTree


class BoundaryForest(object):

    def __init__(self, k, init_data):
        self.nt = len(init_data)
        self.trees = []
        for ex_i in init_data:
            tmp = BoundaryTree(k, ex_i)
            for (j_x, j_y) in init_data:
                tmp.train(j_x, j_y)
            self.trees.append(tmp)

    def query(self, test_x):
        node_list = []
        for t in self.trees:
            node = t.query(test_x)
            node_list.append((t.euclidean(t.data[node][0], test_x), t.data[node][1]))
        return min(node_list, key=lambda x: x[0])


    def train(self, new_x, new_y):
        for t in self.trees:
            t.train(new_x, new_y)

if __name__ == "__main__":
    from image_test import generateImage
    from PIL import Image
    import pickle
    import time
    from pylab import *

    img = generateImage(60, 60, 10)
    img.show()
    dataset = []
    print(img.size[1])
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            print((np.asarray([x, y], dtype=np.int32), np.asarray(img.getpixel((x, y)), dtype=np.float32)))
            dataset.append((np.asarray([x, y], dtype=np.int32), np.asarray(img.getpixel((x, y)), dtype=np.float32)))
    dataset = np.asarray(dataset)
    k = np.inf
    n_t = 4

    num_examples = len(dataset)
    train_batch = dataset[0:n_t]
    forest = BoundaryForest(k, train_batch)

    print("Forest is photosynthesizing...")
    for ex in xrange(num_examples):
        forest.train(dataset[ex][0], dataset[ex][1])
        if ex % 1000 == 0:
            print(str(ex) + " / " + str(num_examples))
    print("All done!")

    fig = plt.figure()
    a1 = fig.add_subplot(1, 2, 1)
    a1.set_title("Original")
    plt.show(img)
    a2 = fig.add_subplot(1, 2, 2)
