"""
    Jay Ricco - BoundaryTree
    Introduction to AI, Section: 03
    6/3/17
"""
import numpy as np
import copy
import time


class BoundaryTree:

    @staticmethod
    def l2_distance(xy1, xy2):
        return np.sum(np.square(np.subtract(np.asarray(xy1, dtype=np.float64), xy2.transpose())))

    def __init__(self, k, root_data):
        self.max_children = k
        self.data = {0: root_data}
        self.children = {0: []}
        self.root = 0
        self.count = 1
        self.avg_depth = 0

    def query(self, test_x, internal=False):
        node = self.root
        depth_meter = 0
        while True:
            children = self.children[node]
            if len(children) < self.max_children:
                children = copy.copy(children)
                children.append(node)

            closest_node = min(children,
                               key=lambda child_node: BoundaryTree.l2_distance(self.data[child_node][0], test_x))
            depth_meter += 1
            if closest_node == node:
                break
            node = closest_node
        if internal:
            return (node, depth_meter)
        else:
            return self.data[node][1]

    def train(self, new_x, new_y):
        start = time.clock()
        (closest_node, depth) = self.query(new_x, internal=True)
        stop = time.clock()
        if not np.array_equal(self.data[closest_node][1], new_y):
            self.data[self.count] = (new_x, new_y)
            self.children[self.count] = []
            self.children[closest_node].append(self.count)
            self.count += 1
        return (stop - start), depth







