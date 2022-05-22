from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class Delta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, trainUntilStable:bool=True):
        """
        Create a new Delta Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 10
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to True, train the current pattern until stable.
        """

        # Delta rule requires a single update step 
        self.updateSteps = 1

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs

        # Currently, numStatesLearned unused by Delta
        self.numStatesLearned = 0

        # Flag to determine if we should train until stable, defaults to True
        self.trainUntilStable = trainUntilStable

    def __str__(self):
        return f"Delta-{self.maxEpochs} Max Epochs"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        # The weights changes start as a zero matrix
        weightChanges = np.zeros_like(self.network.weights)
        for i in range(len(patterns)):
            # Get pattern i and the resultState i
            pattern = patterns[i].copy()
            resultState = self.findRelaxedState(pattern)

            # The weight changes of this pattern is the outer product of
            # The difference of pattern and result state and this pattern
            # Scale factor is 0.5 so updates are -1 and 1 (one from both sides)
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)

        # This section is an attempt at scaling and unscaling weights between tasks
        # This is tricky with Delta due to many epochs...
        # weights /= np.max(weights)
        # weights += weightChanges
        # weights *= np.max(weights)
        
        return self.network.weights + weightChanges