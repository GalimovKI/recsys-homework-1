import numpy as np
from scipy.sparse import csr_matrix


def matrix(frame, shape, values=None):
    if values is None:
        values = np.ones(len(frame))
    return csr_matrix((values, (frame.user, frame.item)), shape=shape, dtype=np.float64)


def topk(scores, seen, users, k=10):
    scores = np.asarray(scores).copy()
    for row, user in enumerate(users):
        scores[row, seen[user].indices] = -np.inf
    # Stable sorting resolves all ties by ascending item ID.
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    return order


class Recommender:
    def recommend(self, users, seen, k=10):
        return topk(self.score(users), seen, users, k)
