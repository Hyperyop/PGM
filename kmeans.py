import cupy as cp
import numpy as np
from cupyx.scipy.spatial import distance_matrix
from pca import PCA
class KMeans:
    def __init__(self, k=2, max_iters=100, init_centroids="kmeans++", epsilon=1e-5, num_seeds = 10):
        self.k = k
        self.max_iters = max_iters
        self.init_centroids = init_centroids
        self.epsilon = epsilon
        self.num_seeds = num_seeds
        if self.init_centroids == "PCA":
          self.num_seeds=1
          self.num_seeds_pca = num_seeds
    def _init_centroids(self, data):
        if self.init_centroids == "kmeans++":
            centroids = cp.empty((self.k, data.shape[1]))
            centroids[0] = data[cp.random.choice(len(data),1)]
            # starting from a vector of infinity distances
            distances = cp.full(len(data), cp.inf)
            for i in range(self.k - 1):
                # compute the minimum of current distance and the distance to the last observed centroid
                distances = cp.minimum(distances, cp.linalg.norm(data - centroids[i], axis=1))
                # sample a new centroid based on the new distance
                # centroids[i+1] = data[cp.random.choice(len(data),1,p=distances/cp.sum(distances))]
                centroids[i+1] = data[cp.argmax(distances)]
                
        elif self.init_centroids == "PCA":
            pca = PCA(n_components=self.k)
            data_projected = pca.fit_transform(data)
            temp_kmeans = KMeans(k=self.k, max_iters=10, init_centroids="kmeans++",num_seeds = self.num_seeds_pca )
            temp_kmeans.fit(data_projected)
            labels = temp_kmeans.labels_
            centroids = cp.empty((self.k, data.shape[1]))
            for i in range(self.k):
                # check if the centroid is empty
                if cp.sum(labels == i) == 0:
                    # reinitialize the centroid
                    centroids[i] = data[cp.random.choice(len(data),1)]
                else:
                    centroids[i] = cp.mean(data[labels == i], axis=0)
        elif self.init_centroids == "mean_distance":
            centroids = self.mean_distance_initialization(data, self.k)
        else:
            centroids = data[cp.random.choice(len(data), self.k)]
        return centroids
    def compute_inertia(self, data, centroid, labels):
        inertia = 0
        for i in range(centroid.shape[0]):
            # Select only data points that belong to this cluster
            cluster_data = data[labels == i]
            # Compute the distance from the data points to the cluster's centroid
            distances = cp.linalg.norm(cluster_data - centroid[i], axis=1)
            # Add the squared distance to the inertia
            inertia += cp.sum(distances**2)
        return inertia
    def mean_distance_initialization(self, X, k):
        n_samples = X.shape[0]
        mean = cp.mean(X, axis=0)
        distances = cp.linalg.norm(X - mean, axis=1)
        ordered_indices = cp.argsort(distances)
        centroids = cp.empty((k, X.shape[1]))
        for i in range(1,k+1):
            centroids[i-1] = X[ordered_indices[int(1 + (i - 1) * (n_samples / k)) - 1]]
        return cp.asarray(centroids)
    def fit(self, data):
        data = cp.asarray(data)
        best_centroids = cp.empty((self.k, data.shape[1]))
        best_inertia = cp.inf
        best_labels = cp.empty((data.shape[0]))
        illegal_indices = cp.zeros(self.k, dtype = cp.bool_)
        new_centroids = cp.empty((self.k, data.shape[1]))

        for _ in range(self.num_seeds):
            self.centroids = self._init_centroids(data)
            for _ in range(self.max_iters):
                illegal_indices = False * illegal_indices
                distance = distance_matrix(data, self.centroids)
                self.labels_ = cp.argmin(distance, axis=1)
                for i in range(self.k):
                    # check if the centroid is empty
                    if cp.sum(self.labels_ == i) == 0:
                        illegal_indices[i] = True
                        # reinitialize the centroid
                        new_centroids[i] = data[cp.random.choice(len(data),1)]
                    else:
                        new_centroids[i] = cp.mean(data[self.labels_ == i], axis=0)

                legal_mask = ~(illegal_indices)
                max_movement = cp.max(cp.linalg.norm(new_centroids - self.centroids, axis=1)[legal_mask])
                if max_movement < self.epsilon:
                    break

                self.centroids = new_centroids
            inertia = self.compute_inertia(data, self.centroids, self.labels_)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids
                best_labels = self.labels_
        self.centroids = best_centroids.get()
        self.labels_ = best_labels.get()
    def predict(self, data):
        # check if data is of correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != self.centroids.shape[1]:
            raise ValueError("Data has incorrect shape")
        distance = distance_matrix(data, self.centroids)
        return cp.argmin(distance, axis=1).get()
    def transform(self, data):
        # check if data is of correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != self.centroids.shape[1]:
            raise ValueError("Data has incorrect shape")
        return distance_matrix(data,self.centroids).get()