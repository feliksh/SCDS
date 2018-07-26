from scipy.io import loadmat
import numpy as np


def extract_data(filename, extract_mean=True):
    mat_dict = loadmat(filename)
    embedding_vectors = mat_dict['A']
    spk_id = (mat_dict['L'])
    spk_id = [x[0] for x in spk_id]
    spk_id = [x[0] for x in spk_id]
    spk_id = np.asarray(spk_id).flatten()
    train_label = np.asarray(([True] * 8 + [False] * 2) * np.unique(spk_id).shape[0])
    if not extract_mean:  # MU
        mean_vectors = embedding_vectors
        cluster_ids = spk_id
    else:
        mean_vectors, cluster_ids = extract_mean_per_speaker(embedding_vectors, spk_id, train_label)
        cluster_ids = cluster_ids.flatten()

    return mean_vectors, cluster_ids


def extract_mean_per_speaker(feature_matrix, ids, train_labels):
    # calculate mean of single utterances
    first = 0
    cluster_ids = []
    unique_ids = np.unique(ids)

    for speaker_id in unique_ids:
        faks0 = ids == speaker_id
        training = train_labels == True

        speaker_train_vector = feature_matrix[np.where(faks0, training, False), :]
        speaker_test_vector = feature_matrix[np.where(faks0, training == False, False), :]

        column_sum = speaker_train_vector.sum(axis=0)
        mean_train_vector = column_sum / speaker_train_vector.shape[0]

        column_sum = speaker_test_vector.sum(axis=0)
        mean_test_vector = column_sum / speaker_test_vector.shape[0]

        if first < 1:
            mean_vectors = np.asarray(mean_train_vector).squeeze()
            mean_vectors = np.vstack([mean_vectors, np.asarray(mean_test_vector).squeeze()])
            first += 1
        else:
            mean_vectors = np.vstack([mean_vectors, np.asarray(mean_train_vector).squeeze()])
            mean_vectors = np.vstack([mean_vectors, np.asarray(mean_test_vector).squeeze()])

        cluster_ids.append(speaker_id)
        cluster_ids.append(speaker_id)
    cluster_ids = np.asarray(cluster_ids)

    return mean_vectors, cluster_ids
