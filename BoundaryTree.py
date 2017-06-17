import numpy as np
import copy
import time


class BoundaryTree(object):

    @staticmethod
    def distance_function(fv1, fv2):
        """An appropriate distance function. "fv1" and "fv2" are feature vectors,
           between which the distance is computed"""
        # This is just L2 distance, leveraging the speed of numpy - you can play with it if you'd like.
        fv1 = np.asarray(fv1)
        fv2 = np.asarray(fv2)
        return np.sqrt(np.sum(np.square(np.subtract(fv1, fv2))))

    def __init__(self, k, root_data):
        """A function to initialize the tree.
        Remember that the tree needs to keep track of each "node"'s feature vector,
        class label, and all of the refere
        nces to it's children."""

        self.data = {0: root_data}
        self.children = {0:[]}
        self.node_id = 1 # Represents the next UID, just increments.

    def query(self, test_fv, internal=False):
        """ This is where the majority of the work will be for you.
            test_fv: is a feature vector.
            internal: is a flag for knowing when to return the node_id, and when to return the class label....
            The train function expects an id to be returned NOT the class. You can also use depth_meter to implement
            average depth profiling. """

        depth_meter = 0.0

        """*Your Code Here*"""

        if internal:
            return """The node key""", depth_meter
        else:
            return """The node's class"""

    def train(self, test_fv, test_true_class):
        """This is the train function, it has been implemented for you so you can have a better understanding
           of how to implement the query function."""

        start = time.clock()
        # Get the key for the closest node the tree has
        closest_node = self.query(test_fv, internal=True)
        stop = time.clock()

        # Test it against the true class
        if not np.array_equal(self.data[closest_node][1], test_true_class):

            # If they aren't equal, create a new node in the tree
            self.data[self.node_id] = (test_fv, test_true_class)

            # Initialize it's list of children.
            self.children[self.node_id] = []

            # Add it as a child to the closest node
            self.children[closest_node].append(self.node_id)

            # For the purposes of returning the correct class
            closest_node = self.node_id

            # Increase the UID variable.
            self.node_id +=1

        return self.data[closest_node][1], (stop - start)