import numpy as np
from .ials import IALS


class CustomALS(IALS):
    def target(self, ratings):
        return np.maximum(ratings - 3, 0) / 2
