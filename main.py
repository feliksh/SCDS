import dominantset as ds
import extractor
import numpy as np

np.random.seed(23)

# VGGVox feature vectors extracted on TIMIT
voxTimit_small = "TIMIT_SMALL_VGGVOX.mat"
voxTimit_all = "TIMIT_ALL_VGGVOX.mat"

feature_vectors, cluster_ids = extractor.extract_data(voxTimit_small)

# for SCDS set epsilon=1e-6 and cutoff=-0.1
# for SCDSbest (TIMIT full dataset with VGGVox features [1]) set espsilon=1e-7 and cutoff=-0.67
dos = ds.DominantSetClustering(feature_vectors=feature_vectors, speaker_ids=cluster_ids,
                               metric='cosine', dominant_search=False,
                               epsilon=1e-7, cutoff=-0.67)

dos.apply_clustering()
mr, ari, acp = dos.evaluate()

print("MR\t\tARI\t\tACP")
print("{0:.4f}\t{1:.4f}\t{2:.4f}".format(mr, ari, acp))  # MR - ARI - ACP


"""

[1] F. Hibraj, S. Vascon, T. Stadelmann, M. Pelillo, 
    "Speaker Clustering using Dominant Sets," ICPR 2018

"""