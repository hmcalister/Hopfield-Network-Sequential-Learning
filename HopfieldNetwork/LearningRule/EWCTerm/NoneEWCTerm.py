import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class NoneEWCTerm(AbstractEWCTerm):

    def __init__(self):
        pass

    def generateTerm(self, taskWeights, taskPatterns):
        importance = np.zeros_like(self.taskWeights)
        return self.EWCTerm(importance, taskWeights)

    
    def __str__(self):
        return "NoneEWCTermGenerator"

    
    def toString(self):
        return "NoneEWCTermGenerator"
