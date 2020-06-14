from micro_cluster import MicroCluster
from sklearn import metrics
import numpy as np
import sklearn
import operator
import time
import math


class SOStream:

    def __init__(self, data, cluster_object, alpha, minPts, merge_threshold, decay_rate, fade_threshold):
        self.data = data
        self.cluster_object = cluster_object
        self.alpha = alpha
        self.minPts = minPts
        self.merge_threshold = merge_threshold
        self.decay_rate = decay_rate
        self.fade_threshold = fade_threshold
        self.number_of_merged_clusters = 0
        self.number_of_faded_clusters = 0

    def min_distance(self, sample):
        clusters = self.cluster_object.get_clusters()
        distances = {}
        for cluster in clusters:
            distances[clusters.index(cluster)] = metrics.pairwise.euclidean_distances(sample, cluster.centroid)
        sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
        return clusters[sorted_distances[0][0]]

    def find_neighbors(self, win):
        if len(self.cluster_object.clusters) >= self.minPts:
            win_distances = {}
            clusters = self.cluster_object.get_clusters()
            for cluster in clusters:
                if clusters.index(cluster) != clusters.index(win):
                    win_distances[clusters.index(cluster)] = \
                        sklearn.metrics.pairwise.euclidean_distances(win.centroid,
                                                                     cluster.centroid)
            sorted_win_distances = sorted(win_distances.items(), key=operator.itemgetter(1))
            if len(sorted_win_distances) <= 2:
                radius = sorted_win_distances[self.minPts - 2][1]
            else:
                radius = sorted_win_distances[self.minPts - 1][1]
            win.set_radius(radius)
            win_neighbors = []
            for d in sorted_win_distances:
                if d[1] <= radius:
                    win_neighbors.append(clusters[d[0]])
            return win_neighbors
        else:
            return 0

    def update_cluster(self, win, input_vector, win_neighbors):
        win.number_data_points += 1
        win.insert(input_vector.index[0])
        win.update_last_edited_time(time.time())
        for n in win_neighbors:
            last_centroid = n.centroid
            power = -(sklearn.metrics.pairwise.euclidean_distances(last_centroid, win.centroid)) / (
                    2 * (win.radius ** 2))
            beta = math.exp(power)
            n.centroid = np.add(last_centroid, self.alpha * beta * np.subtract(win.centroid, last_centroid))

        return win_neighbors

    def find_overlap(self, win, win_neighbor):
        overlap = []
        clusters = self.cluster_object.get_clusters()

        for n in win_neighbor:
            if clusters.index(win) != clusters.index(n):
                d = sklearn.metrics.pairwise.euclidean_distances(win.centroid, n.centroid)
                if d - (win.radius + n.radius) < 0:
                    overlap.append(n)
        return overlap

    def merge_clusters(self, win, overlap):
        for n in overlap:
            distance = sklearn.metrics.pairwise.euclidean_distances(win.centroid, n.centroid)
            if distance < self.merge_threshold:
                self.number_of_merged_clusters += 1

                new_centroid = np.add(win.number_data_points * win.centroid, n.number_data_points *
                                      n.centroid) / (
                                       win.number_data_points + n.number_data_points)

                d1 = sklearn.metrics.pairwise.euclidean_distances(new_centroid, win.centroid) + win.radius
                d2 = sklearn.metrics.pairwise.euclidean_distances(new_centroid, n.centroid) + n.radius
                new_radius = max(d1, d2)

                merged_cluster = MicroCluster(number_data_points=win.number_data_points + n.number_data_points,
                                              radius=new_radius,
                                              centroid=new_centroid,
                                              current_timestamp=time.time())

                merged_cluster.merge_data_points(win.data_points, n.data_points)
                self.cluster_object.insert(merged_cluster)

                self.cluster_object.get_clusters().remove(n)
                if win in self.cluster_object.get_clusters():
                    self.cluster_object.get_clusters().remove(win)

    def fading_all(self):
        for n in self.cluster_object.get_clusters():
            fade_value = n.calculate_fade(time.time(), self.decay_rate)
            if fade_value < self.fade_threshold:
                self.number_of_faded_clusters += 1
                self.cluster_object.get_clusters().remove(n)

    def get_clusters(self):
        return self.cluster_object.clusters

    def get_cluster_obj(self):
        return self.cluster_object

    def get_number_of_merged_clusters(self):
        return self.number_of_merged_clusters

    def get_number_of_faded_clusters(self):
        return self.number_of_faded_clusters
