"""
    Jay Ricco - BoundaryTree
    Introduction to AI, Section: 03
    6/3/17
"""
import numpy as np
from collections import OrderedDict


class BoundaryTree:

    @staticmethod
    def euclidean(x1, x2):
        return np.sum(np.square(np.subtract(x1, x2)))

    def __init__(self, k, root_data):
        self.data = OrderedDict()
        self.data[0] = root_data
        self.relations = OrderedDict()
        self.count = 0
        self.root = 0
        self.relations[0] = []
        self.max_children = k

    def query(self, test_x):
        node = self.root
        while True:
            children = set(self.relations[node])
            if len(children) < self.max_children:
                children.add(node)
            closest_node = min(children, key=lambda c_i: BoundaryTree.euclidean(self.data[c_i][0], test_x))
            if closest_node == node:
                break
            node = closest_node
        return node

    def train(self, new_x, new_y):
        closest_node = self.query(new_x)
        if not np.array_equal(self.data[closest_node], new_y):
            self.count += 1
            self.data[self.count] = (new_x, new_y)
            self.relations[self.count] = []
            self.relations[closest_node].append(self.count)



