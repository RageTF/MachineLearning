import numpy as np
import random
from matplotlib import pyplot as plt
from math import sqrt
from enum import Enum

test_data = []
for i in range(1000):
    test_data.append([random.randint(0, 100), random.randint(0, 100)])
test_data = np.array(test_data)


class DistType(Enum):
    EUCLIDEAN = 1
    PLANE = 2
    SQUARE = 3
    MAX = 4


class KMeans:

    def __init__(self, data_set, n, dist=DistType.EUCLIDEAN):
        self.set = data_set
        self.n = n
        self.e = 1000
        self.tolerance = .01
        self.fitted = False
        self.dist = dist
        self.labels = np.array([])
        self.centroids = self.choose_centroids()

    def choose_centroids(self):
        # return np.array([self.dataset[k] for k in random.sample(range(len(self.dataset)), self.n_clusters)])

        point_range = range(len(self.set))

        start = random.randint(0, len(point_range) - 1)

        point_range_without_centroids = [k for k in point_range if k != start]

        centroids = [self.set[start]]

        for c in range(self.n - 1):
            max_dist = 0
            indx = -1

            for i in range(len(centroids)):
                for d in range(len(point_range_without_centroids)):
                    dist = self.get_dist(centroids[i], self.set[point_range_without_centroids[d]])
                    if dist > max_dist:
                        max_dist = dist
                        indx = d

            if indx != -1:
                centroids.append(self.set[point_range_without_centroids[indx]])
                point_range_without_centroids.pop(indx)
            else:
                raise ValueError('Count clusters more then points inside dataset')

        print("Centroids:\n------------")
        for i in range(len(centroids)):
            print(centroids[i])
        print("------------")

        return np.array(centroids)

    def get_dist(self, list1, list2):
        if self.dist == DistType.EUCLIDEAN:
            return self.get_euclidian_dist(list1, list2)
        elif self.dist == DistType.PLANE:
            return self.get_abs_dist(list1, list2)
        elif self.dist == DistType.SQUARE:
            return self.get_dist_square(list1, list2)
        elif self.dist == DistType.MAX:
            return self.get_max_dist(list1, list2)
        else:
            raise ValueError('Incorrect distance')

    def get_dist_square(self, list1, list2):
        return sum((i - j) ** 2 for i, j in zip(list1, list2))

    def get_euclidian_dist(self, list1, list2):
        return sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))

    def get_abs_dist(self, list1, list2):
        return sum(abs(i - j) for i, j in zip(list1, list2))

    def get_max_dist(self, list1, list2):
        return max(abs(i - j) for i, j in zip(list1, list2))

    def prepare(self):
        self.labels = np.array([])
        for elem in self.set:
            dist2 = [self.get_dist(elem, center) for center in self.centroids]
            idx = dist2.index(min(dist2))
            self.labels = np.append(list(self.labels), idx).astype(int)

    def calc_centroids(self):
        for j in range(self.n):
            num = 0
            temp = np.zeros(self.set[0].shape)
            for k, label in enumerate(self.labels):
                if label == j:
                    temp = temp + self.set[k]
                    num += 1
            if num == 0:
                num = 1
            self.centroids[j] = temp / num

    def start(self):
        iter = 0
        while iter < self.e:
            prev_centroids = np.copy(self.centroids)
            self.prepare()
            self.calc_centroids()
            if max([self.get_dist(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1
        self.fitted = True

    def show(self):
        if self.set.ndim != 2:
            return

        cmap = plt.cm.get_cmap('hsv', len(self.labels))
        plt.figure()

        for i, (X, Y) in zip(self.labels, self.set):
            plt.scatter(X, Y, c=cmap(i))

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=100, c='black')
        plt.show()


test = KMeans(test_data, 100)
test.start()
test.show()
