from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class Delta(AbstractLearningRule):

    def __init__(self, epochs:int=10):
        """
        Create a new Delta Learning Rule

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 10
        """

        self.updateSteps = 1
        self.epochs = epochs

        self.numStatesLearned = 0

    def __str__(self):
        return "Delta"

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
        for i in range(len(patterns)):
            pattern = patterns[i]
            resultState = resultStates[i]

            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)

        # weights *=np.max(weights)
        # weights += weightChanges
        # weights /= np.max(weights)
        return weights + weightChanges