import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class WeightDecayTerm(AbstractEWCTerm):
    def __init__(self, task_weights, network=None):
        super().__init__(task_weights, network)
        self.importance = np.ones_like(self.taskWeights)

    @classmethod
    def __str__(cls):
        return "WeightDecayTermGenerator"

    @classmethod
    def toString(cls):
        return "WeightDecayTermGenerator"