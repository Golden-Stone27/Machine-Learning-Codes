from sklearn import datasets, cluster

import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt



n_samples = 1500
noisy_circle = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05)
noisy_moons = datasets.make_moons(n_samples = n_samples,noise = 0.05)
blobs = datasets.make_blobs(n_samples = n_samples)
no_structers = np.random.rand(n_samples, 2), None


clustering_names = ["MiniBatchKmeans", "SpectralClustering", "Ward", "AgglomerativeClustering", "DBSCAN", "Birch"]

colors = np.array(["b", "g", "r", "c", "m", "y"])
datasets = [noisy_circle, noisy_moons, blobs, no_structers]

plt.figure()
i = 1
for i_datasets, dataset in enumerate(datasets):

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    two_means = cluster.MiniBatchKMeans(n_clusters = 2)
    ward = cluster.AgglomerativeClustering(n_clusters = 2, linkage = "ward")
    spectral = cluster.SpectralClustering(n_clusters = 2)
    dbscan = cluster.DBSCAN(eps = 0.2)
    average_linkage = cluster.AgglomerativeClustering(n_clusters = 2, linkage = "average")
    birch = cluster.Birch(n_clusters=2)

    clustering_alghoritms = [two_means, ward, spectral, dbscan, average_linkage, birch]

    for name, algo in zip(clustering_names, clustering_alghoritms):
        algo.fit(X)
        if hasattr(algo, "labels_"):
            y_pred = algo.labels_.astype(int)
        else:
            y_pred = algo.predict(X)

        plt.subplot(len(datasets), len(clustering_alghoritms), i)
        if i_datasets == 0:
            plt.title(name)
        plt.scatter(X[:, 0], X[:, 1], color = colors[y_pred].tolist(), s = 10)
        i+=1



plt.show()
