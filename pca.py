import cupy as cp

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        X = cp.asarray(X)
        self.mean_ = cp.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = cp.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrix)
        sorted_indices = cp.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, sorted_indices[:self.n_components]]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = cp.asarray(X)
        X_centered = X - self.mean_
        return cp.dot(X_centered, self.components_)