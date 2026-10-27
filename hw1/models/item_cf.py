import numpy as np
from scipy.sparse import csr_matrix
from .base import Recommender, matrix


class ItemCF(Recommender):
    def __init__(self, neighbors=100):
        self.neighbors = neighbors

    def fit(self, frame, shape):
        self.x = matrix(frame, shape, (frame.rating >= 4).astype(float))
        norms = np.sqrt(np.asarray(self.x.power(2).sum(axis=0)).ravel())
        gram = (self.x.T @ self.x).toarray()
        gram /= np.maximum(norms[:, None] * norms[None, :], 1e-12)
        np.fill_diagonal(gram, 0)
        # For each target keep its strongest source neighbors.
        indices = np.argsort(-gram, axis=0, kind='stable')[:self.neighbors]
        cols = np.broadcast_to(np.arange(shape[1]), indices.shape)
        self.weights = csr_matrix((gram[indices, cols].ravel(),
                                   (indices.ravel(), cols.ravel())), shape=gram.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
