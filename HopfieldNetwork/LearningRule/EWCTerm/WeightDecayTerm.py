import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class WeightDecayTerm(AbstractEWCTerm):
    def __init__(self):
        pass

    def generateTerm(self, taskWeights, taskPatterns):
        importance = np.ones_like(taskWeights)
        return self.EWCTerm(importance, taskWeights, len(taskPatterns))

    
    def __str__(self):
        return "WeightDecayTermGenerator"

    
    def toString(self):
        return "WeightDecayTermGenerator"