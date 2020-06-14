import operator
import time
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SOStream import SOStream
from cluster import Cluster
from micro_cluster import MicroCluster


def cal_purity(clusters, data):
    sum = 0
    for cluster in clusters:
        points = cluster.data_points
        labels = np.array([])
        for point in points:
            labels = np.append(labels, data.iloc[point:point + 1, 2])
        unq_label, counts = np.unique(labels, return_counts=True)
        d = dict(zip(unique_label, counts))
        dominant_label_count = max(d.items(), key=operator.itemgetter(1))[1]
        sum += float(dominant_label_count) / cluster.number_data_points
    return (sum / len(clusters)) * 100


if __name__ == '__main__':
    data = pd.read_csv('Dataset_2.csv', header=None)
    data = data.reindex(np.random.permutation(data.index))
    df_label = np.array(data.iloc[:, 2])
    unique_label = np.unique(df_label)
    data_frame = data.drop([2], axis=1)
    print(data_frame.shape)

    sos = SOStream(data=data_frame, cluster_object=Cluster(), alpha=0.1, minPts=3, merge_threshold=2,
                   decay_rate=0.1, fade_threshold=2)

    # iteration on the data frame
    purity_list = dict()
    cluster_list = dict()
    start_time = time.time()
    for i in range(len(data_frame)):
        input_vector = data_frame[i:i + 1]
        t = time.time()
        if len(sos.get_cluster_obj().get_clusters()) - 1 >= sos.minPts:

            win = sos.min_distance(input_vector)
            win_neighbors = sos.find_neighbors(win)

            distance = sklearn.metrics.pairwise.euclidean_distances(input_vector, win.centroid)
            if distance <= win.radius:
                win_neighbors = sos.update_cluster(win, input_vector, win_neighbors)

            else:
                new_micro_cluster = MicroCluster(number_data_points=1, centroid=input_vector, radius=0,
                                                 current_timestamp=t)
                new_micro_cluster.insert(input_vector.index[0])
                sos.get_cluster_obj().insert(new_micro_cluster)

            overlap = sos.find_overlap(win, win_neighbors)
            if len(overlap) > 0:
                sos.merge_clusters(win, overlap)

        else:
            new_micro_cluster = MicroCluster(number_data_points=1, centroid=input_vector, radius=0,
                                             current_timestamp=t)
            new_micro_cluster.insert(input_vector.index[0])
            sos.get_cluster_obj().insert(new_micro_cluster)

        if i % 100 == 0:
            sos.fading_all()
        # time.sleep(0.1)

        # changes of number of clusters over time
        if i % 50 == 0:
            cluster_list[i] = len(sos.get_cluster_obj().get_clusters())
            if i != 0:
                purity = cal_purity(sos.get_clusters(), data)
                purity_list[i] = purity

    cluster_list[len(data_frame)] = len(sos.get_cluster_obj().get_clusters())
    purity_list[len(data_frame)] = cal_purity(sos.get_clusters(), data)

    print("------------------------------------------------------------------------- ")
    print(cluster_list)
    y_pos = list(cluster_list.keys())
    plt.ylabel('number of clusters')
    plt.xlabel('number of received data')
    plt.bar(y_pos, cluster_list.values(), width=10, color='r', align='center', alpha=0.3)
    plt.show()

    print("------------------------------------------------------------------------- ")
    print(purity_list)
    y_pos = list(purity_list.keys())
    plt.ylabel('purity')
    plt.xlabel('number of received data')
    plt.bar(y_pos, purity_list.values(), width=10, color='b', align='center', alpha=0.3)
    plt.show()

    _sum = 0
    count = 0
    for key in purity_list:
        count += 1
        _sum += purity_list[key]

    ave = _sum / count
    print("*************************************************************")
    print("Results : ")
    process_time = time.time() - start_time
    print("process time = ", process_time)
    print("total number of clusters = ", len(sos.get_cluster_obj().get_clusters()))
    print("Number of merged clusters = ", sos.get_number_of_merged_clusters())
    print("Number of faded clusters = ", sos.get_number_of_faded_clusters())
    print("Average Purity = ", ave)

