import cupy as cp
from cupyx.scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = cp.ones(self.n_components) / self.n_components
        self.means = cp.random.randn(self.n_components, n_features)
        self.covariances = cp.array([cp.eye(n_features) for _ in range(self.n_components)])

        # EM algorithm
        for _ in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._expectation(X)

            # M-step: update parameters
            self._maximization(X, responsibilities)

    def _expectation(self, X):
        n_samples = X.shape[0]
        responsibilities = cp.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])

        responsibilities /= cp.sum(responsibilities, axis=1, keepdims=True)

        return responsibilities

    def _maximization(self, X, responsibilities):
        n_samples = X.shape[0]

        # Update weights
        self.weights = cp.mean(responsibilities, axis=0)

        # Update means
        weighted_sum = cp.dot(responsibilities.T, X)
        self.means = weighted_sum / cp.sum(responsibilities, axis=0, keepdims=True)

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = cp.dot((diff * responsibilities[:, k]).T, diff)
            self.covariances[k] = weighted_diff / cp.sum(responsibilities[:, k])

    def predict(self, X):
        n_samples = X.shape[0]
        log_likelihoods = cp.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_likelihoods[:, k] = cp.log(self.weights[k]) + multivariate_normal.logpdf(X, self.means[k], self.covariances[k])

        return cp.argmax(log_likelihoods, axis=1)
