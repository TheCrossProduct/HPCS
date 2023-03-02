import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score, confusion_matrix
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ri
from sklearn.metrics import jaccard_score as iou

from hpcs.utils.arrays import cartesian_product

import itertools


def compute_scores(y_true, y_pred, threshold=0.0, print_info=False, sample_name=None):
    """Auxiliary Function that compute binary prediction scores.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndrarray
        Prediction. It can also be a probability vector

    threshold: float
        if greater than 0.0 this value will be used as threshold value for prediction.

    print_info: bool
        if true program flushes out obtained scores

    sample_name: str
        print out name of sample in flush out
    """

    if threshold > 0.0:
        y_pred = (y_pred > threshold).flatten()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    if print_info:
        print("Scores {}: \n"
              "F1 -> {},\n"
              "Recall -> {},\n"
              "Precision -> {},\n"
              "Accuracy -> {},\n"
              "Jaccard -> {}.".format(sample_name, f1, recall, precision, acc, jaccard))

        print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores


def mat_renorm_rows(M):
    sr = M.astype(float).sum(axis=1)[:, np.newaxis]
    return np.divide(M, sr, where=sr != 0.0)


def get_confusion_matrix(y_true, y_pred, selectedId):
    conf_mat = confusion_matrix(y_true, y_pred, labels=selectedId)
    conf_mat_norm = mat_renorm_rows(conf_mat)
    return conf_mat, conf_mat_norm


def normalize_confusion_matrix(conf_mat):
    conf_mat_norm = np.zeros(conf_mat.shape)
    mySum = conf_mat.astype(np.float).sum(axis=1)
    myLen = mySum.shape[0]

    for i in range(myLen):
        currentSum = mySum[i]
        if currentSum > 0:
            for j in range(myLen):
                conf_mat_norm[i, j] = conf_mat[i, j] / mySum[i]

    conf_mat_norm = np.around(conf_mat_norm, decimals=2)

    return conf_mat_norm


def condense_confusion_matrix(conf_mat, input_labels, condense_list):
    condensed_mat = np.zeros((len(condense_list), len(condense_list)))
    ranges = np.arange(len(condense_list))
    ext_to_condense = []
    cond = []
    for condense in condense_list:
        cond_i = []
        for el in condense:
            cond_i.append(input_labels.index(el))
        cond.append(cond_i)

    for i in ranges:
        for j in ranges:
            vec0, vec1 = np.array(cond[i]), np.array(cond[j])
            c = cartesian_product([vec0, vec1])
            ii = np.repeat(i, len(c))
            jj = np.repeat(j, len(c))
            ext_to_condense.append(np.c_[ii, jj, c])

    ext_to_condense = np.concatenate(ext_to_condense, axis=0)
    for coords in ext_to_condense:
        condensed_mat[coords[0], coords[1]] += conf_mat[coords[2], coords[3]]

    assert condensed_mat.sum() == conf_mat.sum()

    return condensed_mat


############################################################################
# Copyright ESIEE Paris (2019)                                             #
#                                                                          #
# Contributor(s) : Giovanni Chierchia, Benjamin Perret                     #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################
import torch

def remap_labels(y_true):
    y_remap = torch.zeros_like(y_true)
    for i, l in enumerate(torch.unique(y_true)):
        y_remap[y_true==l] = i
    return y_remap

def get_optimal_k(y, linkage_matrix, index):
    best_score = 0.0
    n_clusters = y.max() + 1
    # min_num_clusters = max(n_clusters - 1, 1)
    best_k = 0
    best_pred = None
    y_true = remap_labels(y)
    y_true_clusters = len(torch.unique(y_true))
    for k in range(1, y_true_clusters + 5):
        # print(k)
        y_pred = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        y_pred_clusters = len(torch.unique(torch.Tensor(y_pred)))
        matrix = torch.zeros(y_true_clusters, y_pred_clusters)
        if index == 'ri':
            k_score = ri(y, y_pred)
            if k_score > best_score:
                best_score = k_score
                best_k = k
                best_pred = y_pred
        elif index == 'iou':
            for i in range(y_true_clusters):
                for j in range(y_pred_clusters):
                    matrix[i, j] = iou(y_true==i, y_pred==j)  # matrix.shape = N x M; N true ; M pred;
            out, ind = torch.max(matrix, dim=1) # ind.shape  0 <= N[i] <= (M - 1)
            y_remap = np.zeros_like(y_pred)
            for i in range(y_true_clusters):
                y_remap[y_pred==int(ind[i])] = i + 1

            y_true_cat = np.eye(y_true_clusters+1)[y_true+1]
            y_pred_cat = np.eye(y_true_clusters+1)[y_remap]
            k_score = (np.logical_and(y_true_cat, y_pred_cat).sum()) / (np.logical_or(y_true_cat, y_pred_cat).sum())
            if k_score > best_score:
                best_score = k_score
                best_k = k
                best_pred = y_pred

    return best_pred, best_k, best_score


def accuracy_clustering(y_true, y_pred):
    # Ordering labels
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)

    scores = []

    # Try all the possible permutations
    permutations = list(itertools.permutations(labels))
    for perm in permutations:
        y_permuted = np.zeros_like(y_true)
        for i, k in enumerate(perm):
            y_permuted[y_pred == k] = labels[i]
        score = accuracy_score(y_true, y_permuted)
        scores.append(score)

    return max(scores)


def purity(y_true, y_pred):
    # vector which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def eval_clustering(y_true, Z):
    n_clusters = y_true.max() + 1
    y_pred = fcluster(Z, n_clusters, criterion='maxclust') - 1
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)

    acc_score = accuracy_clustering(y_true, y_pred)
    pu_score = purity(y_true, y_pred)
    nmi_score = nmi(y_true, y_pred, average_method='geometric')  # average_method='arithmetic'
    ri_score = ri(y_true, y_pred)
    iou_score = iou(y_true, y_pred, average='weighted')
    return ri_score, iou_score


