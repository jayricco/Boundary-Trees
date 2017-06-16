"""
    Jay Ricco - DifferentiableBoundaryTree
    Introduction to AI, Section: 03
    6/3/17
"""
import numpy as np
from collections import OrderedDict


class DifferentiableBoundaryTree:

    def euclidean(self, x1, x2):
        return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

    def softmax(self, z):
        print(np.exp(z))
        print(np.sum(np.exp(z)))
        return np.divide(np.exp(z), np.sum(np.exp(z)))

    def trans_to_child(self, current_node, child, y):
            dists = []
            for child in (self.relations[current_node] + [current_node]):
                dists.append(-self.euclidean(child, y))
            s = np.sum(dists)
            return np.sum([np.exp(d)/s for d in dists])

    def greedy_path(self, path, y):
        mul = 1.0
        for i in xrange(len(path)-1):
            mul *= self.trans_to_child(path[i], path[i + 1], y)
        return mul

    def p(self, path, c, y):
        s1 = 0
        s2 = 0
        for i in xrange(len(path)-1):
            s1 += np.log(self.trans_to_child(path[i], path[i+1], y))
        for ss in self.relations[path[-1]]:
            s2 = np.add(s2, np.multiply(self.trans_to_child(ss, path[-1], y), c))
        s2 = np.log(s2)
        return s1 + s2

    def __init__(self, k, root_data):
        self.data = OrderedDict()
        self.data[0] = root_data
        self.relations = OrderedDict()
        self.parents = OrderedDict()
        self.count = 0
        self.root = 0
        self.parents[0] = None
        self.relations[0] = []
        self.max_children = k

    def query(self, test_x):
        path = []
        node = self.root

        while True:
            path.append(node)
            children = set(self.relations[node])
            if len(children) < self.max_children:
                children.add(node)
            closest_node = min(children, key=lambda c_i: self.euclidean(self.data[c_i][0], test_x))
            if closest_node == node:
                break
            node = closest_node
        print(self.p(path, np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0]), test_x))
        return node

    def train(self, new_x, new_y):
        closest_node = self.query(new_x)
        if not np.array_equal(self.data[closest_node], new_y):
            self.count += 1
            self.data[self.count] = (new_x, new_y)
            self.relations[self.count] = []
            self.relations[closest_node].append(self.count)
            self.parents[self.count] = closest_node

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    import pickle
    import time
    from pylab import *

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    k = 30
    n_t = 1

    num_examples = dataset.train.num_examples
    train_batch = dataset.train.next_batch(n_t)
    time_data = []
    forest = DifferentiableBoundaryTree(k, (train_batch[0], train_batch[1]))

    print("Forest is photosynthesizing...")
    for ex in xrange(1000):
        forest.train(dataset.train.images[ex], dataset.train.labels[ex])
        if ex % 100 == 0:
            print(str(ex) + " / " + str(num_examples))
            t_start = time.time()
            forest.query(dataset.train.images[ex])
            t_end = time.time()
            time_data.append(t_end - t_start)