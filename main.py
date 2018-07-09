import dominantset3 as ds

dos = ds.DominantSetClustering(feature_vectors=feature_vectors, speaker_ids=cluster_ids,
                               metric='cosine', dominant_search=False,
                               epsilon=1e-6, cutoff=-0.1)
dos.apply_clustering()
print(dos.evaluate())  # MR - ARI - ACP
