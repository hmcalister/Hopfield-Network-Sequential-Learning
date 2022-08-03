import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AbstractEWCTerm(ABC):

    @abstractmethod
    def __init__(self):
        """Create a new generator for terms"""

    def setNetworkReference(self, network):
        self.network = network

    @abstractmethod
    def generateTerm(self, taskWeights, taskPatterns, **kwargs):
        """
        Init any variables for this importance calculation

        Args:
            taskWeights (np.array): The weights found at the end of this task
            taskPatterns (np.ndarray): The patterns for this task
            network (HopfieldNetwork, optional): A reference to the Hopfield network, defaults to None
        """
        pass

    def startTask(self, **kwargs):
        """
        Do some tidy up before the epoch, reset vars etc
        """
        pass

    def epochCalculation(self, **kwargs):
        """
        Do some calculations for this term at the epoch level
        """
        pass

    def finishTask(self, **kwargs):
        """
        Finish/reset any epoch level calculations
        """
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractEWCTermGenerator"

    @abstractmethod
    def toString(self):
        return "AbstractEWCTermGenerator"

    @dataclass
    class EWCTerm:
        importance: np.ndarray
        taskWeights: np.ndarray
        numPatterns: np.ndarray

        def getImportance(self):
            return self.importance.copy()

        def getTaskWeights(self):
            return self.taskWeights.copy()