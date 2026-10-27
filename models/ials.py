import numpy as np
from scipy.linalg import solve
from .base import Recommender, matrix


class IALS(Recommender):
    def __init__(self, factors=32, reg=0.1, alpha=10, iterations=15, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def target(self, ratings):
        return (ratings >= 4).astype(float)

    def update(self, observed, targets, other):
        base = other.T @ other + self.reg * np.eye(self.factors)
        result = np.zeros((observed.shape[0], self.factors))
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row:row+2]
            ids = observed.indices[start:end]
            vectors = other[ids]
            a = base + self.alpha * (vectors.T @ vectors)
            b = (1 + self.alpha) * (vectors.T @ targets.data[start:end])
            result[row] = solve(a, b, assume_a='pos', check_finite=False)
        return result

    def fit(self, frame, shape):
        observed = matrix(frame, shape)
        targets = matrix(frame, shape, self.target(frame.rating.to_numpy()))
        # Explicit zero targets must remain stored and aligned with observed.
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.01, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.users = self.update(observed, targets, self.items)
            self.items = self.update(observed.T.tocsr(), targets.T.tocsr(), self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
