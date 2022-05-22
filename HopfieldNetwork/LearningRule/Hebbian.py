from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class Hebbian(AbstractLearningRule):

    def __init__(self, trainUntilStable:bool=True):
        """
        Create a new Hebbian Learning Rule
        """

        # Hebbian is a single calculation, no need for updates or multiple epochs
        self.updateSteps = 0
        self.maxEpochs = 1

        # Hebbian does use numStatesLearned
        self.numStatesLearned = 0
        # Flag to determine if we should train until stable, defaults to True
        self.trainUntilStable = trainUntilStable

    def __str__(self):
        return "Hebbian"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        # Weight changes start as zero matrix
        weights = self.network.weights
        weightChanges = np.zeros_like(self.network.weights)

        # For every pattern calculate a weight change
        for pattern in patterns:
            # The weight change of this pattern is outer product of pattern with itself
            weightChanges = weightChanges+np.outer(pattern, pattern)

        # If numStatesLearned is zero we are learning the first task
        if self.numStatesLearned==0:
            # Update the number of states learned
            self.numStatesLearned+=len(patterns)
            # And return the scaled weight changes
            return weights+(1/self.numStatesLearned)*weightChanges
        # Otherwise, we are on a sequential task
        else:
            # First, scale the weights back up by number of patterns learned (before this task)
            weights = weights*self.numStatesLearned
            # Add the patterns we just learned
            self.numStatesLearned+=len(patterns)
            # Add the weight changes
            weights += weightChanges
            # And scale back down
            weights /= self.numStatesLearned
            return weights