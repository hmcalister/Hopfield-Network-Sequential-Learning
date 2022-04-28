from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class BinaryHebbian(AbstractLearningRule):

    def __init__(self):
        """
        Create a new Hebbian Learning Rule
        """

        self.updateSteps = 0
        self.maxEpoches = 1

    def __str__(self):
        return "BinaryHebbian"

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

        weightChanges = np.zeros_like(weights)
        for pattern in patterns:
            weightChanges = weightChanges+np.outer(2*pattern-1, 2*pattern-1)

        return weights+(1/len(patterns))*weightChanges