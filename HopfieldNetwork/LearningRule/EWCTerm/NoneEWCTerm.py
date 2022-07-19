import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class NoneEWCTerm(AbstractEWCTerm):
    def __init__(self, task_weights, network=None):
        super().__init__(task_weights, network)
        self.importance = np.zeros_like(self.taskWeights)

    @classmethod
    def __str__(cls):
        return "NoneEWCTermGenerator"

    @classmethod
    def toString(cls):
        return "NoneEWCTermGenerator"
