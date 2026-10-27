import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender, matrix


class EASE(Recommender):
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, frame, shape):
        self.x = matrix(frame, shape, (frame.rating >= 4).astype(float))
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::shape[1]+1] += self.reg
        factor = cho_factor(gram, overwrite_a=True)
        inverse = cho_solve(factor, np.eye(shape[1]))
        diagonal = inverse.diagonal().copy()
        inverse /= -diagonal[None, :]
        np.fill_diagonal(inverse, 0)
        self.weights = inverse
        return self

    def score(self, users):
        return self.x[users] @ self.weights
