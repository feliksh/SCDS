import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
import scipy.optimize as kuhn


def get_mr(labels, ground_truth):
    le = preprocessing.LabelEncoder().fit(ground_truth)
    ground_truth = le.transform(ground_truth)
    miss = np.float(np.sum(labels != ground_truth))
    return miss / labels.shape[0]


def get_ari(labels, ground_truth):
    le = preprocessing.LabelEncoder().fit(ground_truth)
    return adjusted_rand_score(le.transform(ground_truth), labels)


def get_acp(labels, ground_truth):
    acp = 0.
    for i in range(0, np.max(labels) + 1):
        cluster_elements = ground_truth[labels == i]
        n_i = len(cluster_elements)
        pi_i = 0.
        speaker = []
        for sp in cluster_elements:
            if sp not in speaker:
                n_ij = cluster_elements[cluster_elements == sp].shape[0]
                pi_i += (n_ij * n_ij) / (n_i * n_i)
                speaker.append(sp)
        acp += pi_i * n_i
    acp /= ground_truth.shape[0]
    return acp


def get_asp(labels, ground_truth):
    asp = 0.
    for speaker in np.unique(ground_truth):
        n_j = len(ground_truth[ground_truth == speaker])
        p_j = 0.
        speaker_clusters = np.unique(labels[ground_truth == speaker])
        for i in speaker_clusters:
            cluster_elements = ground_truth[labels == i]
            n_ij = len(cluster_elements[cluster_elements == speaker])
            p_j += (n_ij * n_ij) / (n_j * n_j)
        asp += p_j*n_j
    return asp / ground_truth.shape[0]


def evaluate(labels, ground_truth):
    le = preprocessing.LabelEncoder().fit(ground_truth)
    ground_truth = le.fit(ground_truth)
    return get_mr(labels, ground_truth), get_ari(labels, ground_truth), get_acp(labels, ground_truth)


def get_hungarian(labels, ground_truth):
    le = preprocessing.LabelEncoder().fit(ground_truth)
    ground_truth = le.transform(ground_truth)
    unique_truth = np.unique(ground_truth)
    # Matrix for hungarian people
    oc_mat = np.zeros(shape=(np.max(labels) + 1, unique_truth.shape[0]))

    for i in range(0, np.max(labels) + 1):
        cluster_elements = ground_truth[labels == i]
        if len(cluster_elements) > 0:
            for id in cluster_elements:
                oc_mat[i, id] += 1
    assign = kuhn.linear_sum_assignment(np.max(oc_mat) - oc_mat)

    v = np.zeros_like(ground_truth)
    for i in range(0, assign[0].shape[0]):
        v[labels == assign[0][i]] = unique_truth[assign[1][i]]

    return v


def get_most_partecipating(labels, ground_truth, x):
    le = preprocessing.LabelEncoder().fit(ground_truth)
    result_labels = np.zeros_like(labels)
    for i in range(0, np.max(labels) + 1):
        cluster_elements = ground_truth[labels == i]
        cluster_vals = x[labels == i]
        dominant_element = cluster_elements[cluster_vals == np.max(cluster_vals)][0]
        mask1 = labels == i
        mask2 = ground_truth == dominant_element
        result_labels[mask1 & mask2] = le.transform([dominant_element])[0]
        result_labels[mask1 & np.logical_not(mask2)] = -1

    return result_labels
