from collections import defaultdict
from collections import namedtuple 
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster as sk_cluster
from sklearn import metrics as sk_metrics
import tqdm



CLUSTERING_FMS_FIG_FILENAME = 'fmses.png'
CLUSTERING_KMEANS_CLUSTERER_NAME_PREFIX = 'kmeans'
CLUSTERING_KMEANS_PREDICTIONS_NAME_PREFIX = 'kmeans_predictions'
CLUSTERING_KMEANS_PREDICTIONS_SUBDIR = 'kmeans_predictions'
CLUSTERING_KMEANS_SUBDIR = 'kmeans_clusterers'
CLUSTERING_SAVING_SUBDIR = 'clustering'


CCSResult = namedtuple(
    typename='CCSResult',
    field_names=[
        'cluster_nb_to_k_means',
        'cluster_nb_to_predictions',
        'fms_clusters',
        'fmses',
        'cluster_nb_to_fmses',
        'best_clusterings',
        'best_clusterings_fmses',
        'best_clusterers',
    ]
)


def cellohood_cluster_selection(
    x,
    clusters_to_analyze: Optional[List],
    nb_of_tries_per_cluster_nb: int = 5,
    take_every_example_for_training: int = 5,
    first_random_state: int = 42,
):
    cluster_nb_to_k_means = defaultdict(list)

    current_seed_offset = 0
    for nb_of_clusters in tqdm.tqdm(clusters_to_analyze):
        for _ in range(nb_of_tries_per_cluster_nb):
            current_k_means = sk_cluster.MiniBatchKMeans(
                n_clusters=nb_of_clusters,
                n_init='auto',
                random_state=first_random_state + current_seed_offset,
            )
            current_seed_offset += 1
            cluster_nb_to_k_means[nb_of_clusters].append(current_k_means.fit(x[::take_every_example_for_training]))

    cluster_nb_to_predictions = defaultdict(list)
    
    for nb_of_clusters, k_means_clusterers in cluster_nb_to_k_means.items():
        for k_means_clusterer in k_means_clusterers:
            cluster_nb_to_predictions[nb_of_clusters].append(
                 batch_predict(k_means_clusterer, x))

    fms_clusters, fmses, cluster_nb_to_fmses = [], [], defaultdict(list)

    for current_cluster_ind in range(1, len(clusters_to_analyze) - 1):
        current_cluster = clusters_to_analyze[current_cluster_ind]
        previous_cluster = clusters_to_analyze[current_cluster_ind - 1]
        next_cluster = clusters_to_analyze[current_cluster_ind + 1]
        for current_clusters_predictions in cluster_nb_to_predictions[current_cluster]:
            for previous_clusters_predictions in cluster_nb_to_predictions[previous_cluster]:
                fms_clusters.append(current_cluster) 
                current_fms = sk_metrics.fowlkes_mallows_score(current_clusters_predictions, previous_clusters_predictions)
                fmses.append(current_fms)
                cluster_nb_to_fmses[current_cluster].append(current_fms)
            for next_clusters_predictions in cluster_nb_to_predictions[next_cluster]:
                fms_clusters.append(current_cluster) 
                current_fms = sk_metrics.fowlkes_mallows_score(current_clusters_predictions, next_clusters_predictions)
                fmses.append(current_fms)
                cluster_nb_to_fmses[current_cluster].append(current_fms)

    best_clusterings, best_clusterings_fmses = get_best_clusterings_from_cluster_nb_to_fmses(cluster_nb_to_fmses=cluster_nb_to_fmses)

    best_clusterers = {}
    for bc in best_clusterings:
        inertias = np.array([km.inertia_ for km in cluster_nb_to_k_means[bc]])
        best_cl_index = np.argmin(inertias)
        best_clusterers[bc] = cluster_nb_to_k_means[bc][best_cl_index]
    
    return CCSResult(
        cluster_nb_to_k_means,
        cluster_nb_to_predictions,
        fms_clusters,
        fmses,
        cluster_nb_to_fmses,
        best_clusterings,
        best_clusterings_fmses,
        best_clusterers,
    )
    

def plot_cluster_size_to_fmses(
    cluster_nb_to_fmses,
    fms_clusters,
    fmses,
    best_clusters,
    best_clusters_fmses,
    save_fig: bool = True,
    path: str = None,
):
    cluster_nb_to_mean = {cn: np.mean(cluster_nb_to_fmses[cn]) for cn in cluster_nb_to_fmses}
    cluster_nb_to_std_low = {cn: np.mean(cluster_nb_to_fmses[cn]) - np.std(cluster_nb_to_fmses[cn]) for cn in cluster_nb_to_fmses}
    cluster_nb_to_std_high = {cn: np.mean(cluster_nb_to_fmses[cn]) + np.std(cluster_nb_to_fmses[cn]) for cn in cluster_nb_to_fmses}
    plt.scatter(fms_clusters, fmses, alpha=0.1, label='FMS')
    plt.scatter(cluster_nb_to_mean.keys(), cluster_nb_to_mean.values(), label='mean FMS', alpha=1., c='r')
    plt.plot(cluster_nb_to_std_low.keys(), cluster_nb_to_std_low.values(), '--', alpha=1., c='r')
    plt.plot(cluster_nb_to_std_high.keys(), cluster_nb_to_std_high.values(), '--', alpha=1., c='r')
    plt.fill_between(
        list(cluster_nb_to_std_low.keys()),
        list(cluster_nb_to_std_low.values()),
        list(cluster_nb_to_std_high.values()),
        color='r',
        alpha=0.2,
    )
    plt.plot(cluster_nb_to_std_high.keys(), cluster_nb_to_std_high.values(), '--', alpha=1., c='r')
    plt.scatter(best_clusters, best_clusters_fmses, marker='p', label='Best Clusters', c='green', s=100)
    plt.legend()
    if save_fig:
        fig_path = os.path.join(path, CLUSTERING_FMS_FIG_FILENAME)
        plt.savefig(fig_path)


def plot_ccs_result(
    ccs_result: CCSResult,
    save_fig: bool = False,
    path: str = None,
):
    plot_cluster_size_to_fmses(
        cluster_nb_to_fmses=ccs_result.cluster_nb_to_fmses,
        fmses=ccs_result.fmses,
        fms_clusters=ccs_result.fms_clusters,
        best_clusters=ccs_result.best_clusterings,
        best_clusters_fmses=ccs_result.best_clusterings_fmses,
        save_fig=save_fig,
        path=path,
    )

        
def get_best_clusters_from_ccs_result(ccs_result: CCSResult):
    best_clusters = ccs_result[-2]
    best_clusterers = {}
    for bc in best_clusters:
        inertias = np.array([km.inertia_ for km in ccs_result[0][bc]])
        best_cl_index = np.argmin(inertias)
        best_clusterers[bc] = ccs_result[0][bc][best_cl_index]
    return best_clusterers


def get_best_clusterings_from_cluster_nb_to_fmses(cluster_nb_to_fmses):
    cluster_nb_to_mean = {cn: np.mean(cluster_nb_to_fmses[cn]) for cn in cluster_nb_to_fmses}
    means = [cluster_nb_to_mean[cn] for cn in cluster_nb_to_fmses]
    clusters = list(cluster_nb_to_fmses.keys())
    best_clusters, best_clusters_fmses = [], []
    for cluster_ind in range(1, len(clusters) - 1):
        if means[cluster_ind] > max(means[cluster_ind - 1], means[cluster_ind + 1]):
            best_clusters.append(clusters[cluster_ind])
            best_clusters_fmses.append(means[cluster_ind])
    return best_clusters, best_clusters_fmses

    


def batch_predict(model, data, batch_size=1024):
    current_start = 0
    current_predictions = []
    while current_start < len(data):
        current_end = min(current_start + batch_size, len(data))
        current_predictions.append(model.predict(data[current_start:current_end]))
        current_start += batch_size
    return np.concatenate(current_predictions)
