from abc import ABC, abstractmethod
from typing import List
import numpy as np

class AbstractLearningRule(ABC):

    @abstractmethod
    def __init__(self):
        """
        Create a new Abstract Learning Rule
        """

        # Define how many update steps are needed for each pattern
        self.updateSteps = 0
        # Define the maximum number of epochs to learn for
        self.maxEpoches = 0
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractLearningRule"

    @abstractmethod
    def __call__(self, patterns:List[np.ndarray], resultStates:List[np.ndarray], weights:np.ndarray)->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn
            resultStates (List[np.ndarray]): The relaxed states of the network for each pattern
            weights (np.ndarray): The current weights of the network

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        pass
