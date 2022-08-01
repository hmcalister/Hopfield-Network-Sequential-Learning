import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class HebbianTerm(AbstractEWCTerm):
    def __init__(self):
        super().__init__()

    def generateTerm(self, taskWeights, taskPatterns):
        importance = np.zeros_like(taskWeights)
        for pattern in taskPatterns:
            importance = importance+np.outer(0.5*pattern+0.5, 0.5*pattern+0.5)
        np.fill_diagonal(importance, 0)
        importance /= np.max(importance)
        importance = np.abs(importance)

        return self.EWCTerm(importance, taskWeights)
        
    
    def __str__(self):
        return "HebbianTermGenerator"

    
    def toString(self):
        return "HebbianTermGenerator"