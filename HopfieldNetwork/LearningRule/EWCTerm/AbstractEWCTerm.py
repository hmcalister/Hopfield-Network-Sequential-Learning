import numpy as np
from abc import ABC, abstractmethod

class AbstractEWCTerm(ABC):
    @abstractmethod
    def __init__(self, taskWeights, taskPatterns, network=None):
        """
        Init any variables for this importance calculation

        Args:
            taskWeights (np.array): The weights found at the end of this task
            taskPatterns (np.ndarray): The patterns for this task
            network (HopfieldNetwork, optional): A reference to the Hopfield network, defaults to None
        """
        self.taskWeights = taskWeights.copy()
        self.network = network
        self.importance = np.zeros_like(self.taskWeights)
        pass

    def getImportance(self):
        return self.importance.copy()

    def getTaskWeights(self):
        return self.taskWeights.copy()

    @classmethod
    @abstractmethod
    def __str__(cls):
        return "AbstractEWCTermGenerator"

    @classmethod
    @abstractmethod
    def toString(cls):
        return "AbstractEWCTermGenerator"