import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class WeightDecayTerm(AbstractEWCTerm):
    def __init__(self, taskWeights, taskPatterns, network=None):
        super().__init__(taskWeights, taskPatterns, network)
        self.importance = np.ones_like(self.taskWeights)

    @classmethod
    def __str__(cls):
        return "WeightDecayTermGenerator"

    @classmethod
    def toString(cls):
        return "WeightDecayTermGenerator"