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

        return min(node_list, key=lambda x: x[0])[1]

    def train(self, new_x, new_y):
        for t in self.trees:
            t.train(new_x, new_y)

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    import pickle
    import time
    from pylab import *

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    try:
        fi = open("forest.pickle", 'r+b')
        print("Found a forest! Loading it!")
        t_start = time.time()
        forest = pickle.load(fi)
        t_end = time.time()
        fi.close()
        print("All done!, Time taken: %f" % (t_start - t_end))
    except Exception:
        print("No forest found, I need to plant a new one!")
        k = 30
        n_t = 4

        num_examples = dataset.train.num_examples
        train_batch = dataset.train.next_batch(n_t)
        batch = zip(train_batch[0], train_batch[1])
        time_data = []
        forest = BoundaryForest(k, batch)

        print("Forest is photosynthesizing...")
        for ex in xrange(num_examples):
            forest.train(dataset.train.images[ex], dataset.train.labels[ex])
            if ex % 1000 == 0:
                print(str(ex) + " / " + str(num_examples))
                t_start = time.time()
                forest.query(dataset.train.images[ex])
                t_end = time.time()
                time_data.append(t_end - t_start)
        print("Dumping the pickle...")
        pickle.dump(forest, open("forest.pickle", 'w+b'))
        print("All done!")
        plt.plot(range(len(time_data)), time_data)
        plt.show()


    print("Beginning Testing...")
    num_correct = 1
    num_total = 1
    qtime = 0.0
    for (t_x, t_y) in zip(dataset.test.images, dataset.test.labels):
        print("Image: %d, Acc: %f" % (num_total, (float(num_correct) / num_total) * 100.0))
        t_start = time.time()
        y_p = forest.query(t_x)
        t_end = time.time()
        qtime += (t_end - t_start)

        if np.array_equal(t_y, y_p):
            num_correct += 1
        num_total += 1
    print("Total accuracy: %f" % ((float(num_correct) / num_total) * 100.0))
    print("Avg query time: %f" % (qtime/num_total))