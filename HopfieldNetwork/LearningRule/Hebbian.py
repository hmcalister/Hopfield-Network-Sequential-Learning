from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class Hebbian(AbstractLearningRule):

    def __init__(self):
        """
        Create a new Hebbian Learning Rule
        """

        # Hebbian is a single calculation, no need for updates or multiple epochs
        self.updateSteps = 0
        self.epochs = 1

        # Hebbian does actually use numStatesLearned
        self.numStatesLearned = 0

    def __str__(self):
        return "Hebbian"

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

        # Weight changes start as zero matrix
        weightChanges = np.zeros_like(weights)

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