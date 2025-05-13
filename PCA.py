#PCA.py
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # Select top n_components
        self.components = eigenvectors[:, :self.n_components]  # shape: (n_features, n_components)
        return self

    def transform(self, X):
        # Ensure X has the right shape for mean subtraction
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Input shape {X.shape} doesn't match training shape. Expected second dimension: {self.mean.shape[0]}")
        
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)  # Project data onto principal components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)