import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing
import evaluation
from sklearn.metrics import pairwise as pw
from scipy.spatial import distance


class DominantSetClustering:
    def __init__(self, feature_vectors, speaker_ids, metric,
                 epsilon=1.0e-6, cutoff=1.0e-6, reassignment='noise',
                 dominant_search=False):
        """
        :param metric:
            'cosine'
            'euclidean'
        :param cutoff
            <0: relative cutoff at '-cutoff * max(x)'
            >0: cutoff value. usual value 1e-6
        :param epsilon
            float: minimum distance after which to stop dynamics (tipicaly 1e-6)
        :param reassignment
            'noise': reassign each point to the closest cluster
            'whole': create a new cluster with all remaining points
            'single': create a singleton cluster for each point left
        :param dominant_search
            bool: decide label of clusters through max method. 
            (if False Hungarian method will be applied)
        """
        self.feature_vectors = feature_vectors
        self.cutoff = cutoff
        self.adj_matrix = None
        self.unique_ids = np.unique(speaker_ids)
        self.speaker_ids = speaker_ids
        self.mutable_speaker_ids = np.array(range(self.speaker_ids.shape[0]))
        self.le = preprocessing.LabelEncoder().fit(self.speaker_ids)
        self.ds_result = np.zeros(shape=self.speaker_ids.shape[0], dtype=int)-1
        self.ds_vals = np.zeros(shape=self.speaker_ids.shape[0])
        self.cluster_counter = 0
        self.metric = metric
        self.reassignment = reassignment
        self.dominant_search = dominant_search
        self.epsilon = epsilon
        self.k = 0

    def update_cluster(self, idx, values):
        # in results vector (self.ds_result) assign cluster number to elements of idx
        # values: are partecipating values of carateristic vector of DS
        self.ds_result[self.mutable_speaker_ids[idx]] = self.cluster_counter
        self.ds_vals[self.mutable_speaker_ids[idx]] = values[idx]
        self.mutable_speaker_ids = self.mutable_speaker_ids[idx == False]
        self.cluster_counter += 1

    def get_n_clusters(self):
        return np.max(self.ds_result)

    # similarity matrix: high value = highly similar
    def get_adj_matrix(self):
        if self.metric == 'euclidean':
            dist_mat = distance.pdist(self.feature_vectors, metric=self.metric)
            dist_mat = distance.squareform(dist_mat)
        else:  # cosine distance
            dist_mat = pw.cosine_similarity(self.feature_vectors)
            dist_mat = np.arccos(dist_mat)
            dist_mat[np.eye(dist_mat.shape[0]) > 0] = 0
            dist_mat /= np.pi

        # the following heuristic is derived from Perona 2005 (Self-tuning spectral clustering)
        # with adaption from Zemene and Pelillo 2016 (Interactive image segmentation using
        # constrained dominant sets)
        sigmas = np.sort(dist_mat, axis=1)[:, 1:8]
        sigmas = np.mean(sigmas, axis=1)
        sigmas = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])
        dist_mat /= -sigmas
        self.adj_matrix = np.exp(dist_mat)

        # zeros in main diagonal needed for dominant sets
        self.adj_matrix = self.adj_matrix * (1. - np.identity(self.adj_matrix.shape[0]))
        return self.adj_matrix

    def reassign(self, A, x):
        if self.reassignment == 'noise':
            for id in range(0, A.shape[0]):
                mask = id == np.arange(0, A.shape[0])
                features = self.feature_vectors[self.mutable_speaker_ids[mask]][0]
                nearest = 0.
                cluster_id = 0
                for i in range(0, np.max(self.ds_result)):
                    cluster_elements = self.feature_vectors[self.ds_result == i]
                    if len(cluster_elements) > 0:
                        cluster_vls = self.ds_vals[self.ds_result == i]
                        dominant_element = cluster_elements[cluster_vls == np.max(cluster_vls)][0]
                        temp_features = self.feature_vectors
                        self.feature_vectors = np.asmatrix([features, dominant_element])
                        # call to get_adj_matrix will give the similarity matrix for the selected 2 elements
                        dista = self.get_adj_matrix()[0, 1]
                        self.feature_vectors = temp_features

                        if dista > nearest:
                            cluster_id = i
                            nearest = dista
                self.ds_result[self.mutable_speaker_ids[mask]] = cluster_id
                self.ds_vals[self.mutable_speaker_ids[mask]] = 0.
        if self.reassignment == 'whole':
            self.update_cluster(np.asarray(x) >= 0., np.zeros(shape=len(x)))
        if self.reassignment == 'single':
            x = np.ones(x.shape[0])
            while np.count_nonzero(x) > 0:
                temp = np.zeros(shape=x.shape[0], dtype=bool)
                temp[0] = True
                self.update_cluster(temp, np.zeros(shape=x.shape[0]))
                x = x[1:]

    def apply_clustering(self):

        self.get_adj_matrix()  # calculate similarity matrix based on metric

        counter = 0
        A = self.adj_matrix
        x = np.ones(A.shape[0]) / float(A.shape[0])  # initialize x (carateristic vector)
        while x.size > 1:  # repeat until all objects have been clustered
            dist = self.epsilon * 2
            while dist > self.epsilon and A.sum() > 0:  # repeat until convergence (dist < epsilon means convergence)
                x_old = x.copy()
                counter += 1

                x = x * A.dot(x)  # apply replicator dynamics
                x = x / x.sum() if x.sum() > 0. else x
                dist = norm(x - x_old)  # calculate distance
            
            temp_cutoff = self.cutoff                
            if self.cutoff < 0.:  # relative cutoff
                temp_cutoff = np.abs(self.cutoff) * np.max(x)

            # in case of elements not belonging to any cluster at the end,
            # we assign each of them based on self.reassignment preference
            if A.sum() == 0 or sum(x >= temp_cutoff) == 0:
                print("leaving out:" + str(x.size))
                self.reassign(A, x)
                return

            counter = 0
            idx = x < temp_cutoff
            # those elements whose value is >= temp_cutoff are the ones belonging to the cluster just found
            # on x are their partecipating values (carateristic vector)
            self.update_cluster(x >= temp_cutoff, x)

            A = A[idx, :][:, idx]  # remove elements of cluster just found, from matrix
            x = np.ones(A.shape[0]) / float(A.shape[0])  # re-initialize x (carateristic vector)
        if x.size > 0:  # in case of 1 remaining element, put him on a single cluster
            self.update_cluster(x >= 0., x)

    def evaluate(self):
        self.k = np.max(self.ds_result)+1

        # Missclassification Rate
        # assignment of clusters label
        if self.dominant_search:
            v = evaluation.get_most_partecipating(labels=self.ds_result, ground_truth=self.speaker_ids, x=self.ds_vals)
        else:
            v = evaluation.get_hungarian(labels=self.ds_result, ground_truth=self.speaker_ids)
        mr = evaluation.get_mr(labels=v, ground_truth=self.speaker_ids)

        # Cluster Purity
        acp = evaluation.get_acp(labels=self.ds_result, ground_truth=self.speaker_ids)

        # Adjusted Random Index
        randi = evaluation.get_ari(labels=self.ds_result, ground_truth=self.speaker_ids)

        return mr, randi, acp
