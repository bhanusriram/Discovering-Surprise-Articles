import csv
import math
from collections import defaultdict
from collections import Counter
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]

if __name__ == '__main__':
    input_p = 'C:/Users/doula/Desktop/KDD Final Project Workspace/topics_distribution_rounded.csv'
    columns = defaultdict(list)
    with open(input_p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)
    if '' in columns: del columns['']

    dict_with_floats = defaultdict(list)
    dict_with_floats = {k: list(map(float, columns[k])) for k in columns}
    dict_no_zeros = defaultdict(list)
    for k, v in dict_with_floats.items():
        temp_v = [item for item in v if item != 0.0]
        dict_no_zeros[k] = temp_v

    output_p = 'C:/Users/doula/Desktop/KDD Final Project Workspace/out.txt'
    f = open(output_p, 'w')

    topic_entro = defaultdict(float)
    for k, v in dict_no_zeros.items():
        setted_list = set(v)
        for f in setted_list:
            p = v.count(f) / len(v)
            topic_entro[k] = topic_entro[k] - p * math.log2(p)

    lowest_3_topic_entropies = defaultdict(float)
    for k, v in topic_entro.items():
        lowest_3_topic_entropies[k] = v

    print(lowest_3_topic_entropies)
    k1, v = min(lowest_3_topic_entropies.items(), key=itemgetter(1))
    lowest_3_topic_entropies.pop(k1)
    k2, v = min(lowest_3_topic_entropies.items(), key=itemgetter(1))
    lowest_3_topic_entropies.pop(k2)
    k3, v = min(lowest_3_topic_entropies.items(), key=itemgetter(1))
    lowest_3_topic_entropies.pop(k3)

    # print(len(topic_entro))
    top_3 = Counter(topic_entro)
    for m in top_3:
        print(m)
    top_3_topic_entropies = top_3.most_common(3)

    doc_top_3_topics = defaultdict(list)
    with open(input_p) as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            # print(row[''])
            for item in top_3_topic_entropies:
                doc_top_3_topics[row['']].append(row[item[0]])
    doc_vec = {k: list(map(float, doc_top_3_topics[k])) for k in doc_top_3_topics}
    print(len(doc_vec))


    output_p = 'C:/Users/doula/Desktop/KDD Final Project Workspace/k_mean.csv'
    list_of_topics = []
    list_of_docs = []
    aggregate_topic_docs = []
    with open(output_p, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(('Topic_18', 'Topic_28', 'Topic_2'))
        for k, v in doc_vec.items():
            if sum(v) >= 0.7:
                writer.writerow(v)
                list_of_docs.append(k)
    df = pd.read_csv(output_p)
    print((list_of_docs))
    X = df
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    fig = plt.figure(1, figsize=(15, 15))
    plt.clf()
    ax = Axes3D(fig, elev=45, azim=134)
    ax.scatter(X["Topic_18"], X["Topic_28"], X["Topic_2"], c=labels.astype(np.float))
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='o', s=169, linewidths=3,
               color='r', zorder=10)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Cluster_1')
    ax.set_ylabel('Cluster_2')
    ax.set_zlabel('Cluster_0')
    plt.show()

    # calculating distances between centroid points
    matrix = [[0 for x in range(3)] for y in range(3)]
    matrix[0][0] = 0
    matrix[1][1] = 0
    matrix[0][2] = 0
    matrix[0][1] = matrix[1][0] = dist(centroids[0], centroids[1], None)
    matrix[0][2] = matrix[2][0] = dist(centroids[2], centroids[1], None)
    matrix[1][2] = matrix[2][1] = dist(centroids[0], centroids[2], None)

    # finding cross distance for each centroid point
    dist_list = []
    for t in range(3):
        dist_list.append(0)
    for i in range(0, 2):
        current_dist = 0
        for j in range(0, 2):
            current_dist = current_dist + matrix[i][j]
        dist_list[i] = current_dist

    # finding centroid point with the maximum cross distance
    max_dist = 0
    for item in range(0, 2):
        if dist_list[item] > max_dist:
            max_dist = item
    print(max_dist)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X.index.values
    cluster_map['cluster'] = kmeans.labels_

    # Getting all points that belong to a cluster _ method 1
    print(cluster_map[cluster_map.cluster == 1])

    # Getting all points that belong to a cluster _ method 2
    print(type(ClusterIndicesNumpy(1, kmeans.labels_)))



    # for i in ClusterIndicesNumpy(0, kmeans.labels_):
    #     print(list_of_docs[i],end=',')
    #

    # Articles that belong to cluster 1 are outliers, articles belong to cluster 0 and 2 are most relevant
    # displaying the set of articles that belongs to clutser 0 and 2
    for i in ClusterIndicesNumpy(2, kmeans.labels_):
        print(list_of_docs[i], end=',')
    print('\n')
    for i in ClusterIndicesNumpy(0, kmeans.labels_):
        print(list_of_docs[i], end=',')




