from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class MappedBinaryHebbian(AbstractLearningRule):

    def __init__(self):
        """
        Create a new MappedBinaryHebbian Learning Rule
        This maps the binary field to the bipolar field, allowing for negative weight updates
        """

        # The hebbian does not use result states or multiple epochs
        self.updateSteps = 0
        self.epochs = 1

        # The hebbian does use the states used tracker
        self.numStatesLearned = 0

    def __str__(self):
        return "MappedBinaryHebbian"

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

        # weight changes start off as zero matrix
        weightChanges = np.zeros_like(weights)
        # Calculate an update for each pattern
        for pattern in patterns:
            # Mapping Binary to Bipolar requires 2*x-1, maps 1 to 1 and 0 to -1
            weightChanges = weightChanges+np.outer(2*pattern-1,2*pattern-1)


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