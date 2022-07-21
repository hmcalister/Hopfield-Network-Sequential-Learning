import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class HebbianTerm(AbstractEWCTerm):
    def __init__(self, taskWeights, taskPatterns, network=None):
        super().__init__(taskWeights, taskPatterns, network)
        self.importance = np.zeros_like(self.taskWeights)
        for pattern in self.taskPatterns:
            self.importance = self.importance+np.outer(pattern, pattern)
        print(self.importance)
        print(np.sum(self.importance))
    @classmethod
    def __str__(cls):
        return "HebbianTermGenerator"

    @classmethod
    def toString(cls):
        return "HebbianTermGenerator"