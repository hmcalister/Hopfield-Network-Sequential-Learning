from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class MappedBinaryHebbian(AbstractLearningRule):

    def __init__(self, trainUntilStable:bool=False):
        """
        Create a new MappedBinaryHebbian Learning Rule
        This maps the binary field to the bipolar field, allowing for negative weight updates
        """

        # The hebbian does not use result states or multiple epochs
        self.updateSteps = 0
        self.maxEpochs = 1

        # The hebbian does use the states used tracker
        self.numStatesLearned = 0
        # Flag to determine if we should train until stable, defaults to True
        self.trainUntilStable = trainUntilStable

    def __str__(self):
        return "MappedBinaryHebbian"

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

        # Calculate an update for each pattern
        for pattern in patterns:
            # Mapping Binary to Bipolar requires 2*x-1, maps 1 to 1 and 0 to -1
            weightChanges = weightChanges+np.outer(2*pattern-1,2*pattern-1)


        # If numStatesLearned is zero we are learning the first task
        if self.numStatesLearned==0:
            # And return the scaled weight changes
            return weights+(1/len(patterns))*weightChanges
        # Otherwise, we are on a sequential task
        else:
            # First, scale the weights back up by number of patterns learned (before this task)
            weights = weights*self.numStatesLearned
            # Add the weight changes
            weights += weightChanges
            # And scale back down
            weights /= (self.numStatesLearned+len(patterns))
            return weights

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.numStatesLearned+=len(taskPatterns)