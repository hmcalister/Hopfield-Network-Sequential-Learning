import numpy as np
from .AbstractEWCTerm import AbstractEWCTerm

class SignCounterTerm(AbstractEWCTerm):
    def __init__(self):
        self.totalSignChanges = 1
    
    def generateTerm(self, taskWeights, taskPatterns, **kwargs):
        importance = 1/self.totalSignChanges
        np.fill_diagonal(importance, 0)
        importance /= np.max(importance)
        return self.EWCTerm(importance, taskWeights)

    def startTask(self, **kwargs):
        self.totalSignChanges = 1
        

    def epochCalculation(self, **kwargs):
        newWeight = kwargs.get("weight")
        networkWeights = self.network.weights
        signChanges = np.sign(networkWeights) != np.sign(newWeight)
        self.totalSignChanges += signChanges
        

    def finishTask(self, **kwargs):
        pass

    
    def __str__(self):
        return "SignCounterTerm"

    
    def toString(self):
        return "SignCounterTerm"