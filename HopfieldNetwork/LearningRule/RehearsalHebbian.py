from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class RehearsalHebbian(AbstractLearningRule):

    def __init__(self, maxEpochs:int=1, fracRehearse:np.float64=1, keepPreviousWeights:bool=True, trainUntilStable:bool=True):
        """
        Create a new RehearsalHebbian Learning Rule

        Notice that fracRehearse=1, keepPreviousWeights=False is a "best case scenario" for sequential learning

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 1
            fracRehears (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 1. (all patterns)
            keepPreviousWeights (bool, optional): Flag to keep the previous weights of the network, or throw these away
                Defaults to True.
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to True, train the current pattern until stable.
        """

        # Hebbian is a single calculation, no need for updates or multiple epochs
        self.updateSteps = 0
        self.maxEpochs = maxEpochs
        self.trainUntilStable = trainUntilStable

        # Hebbian does actually use numStatesLearned
        self.numStatesLearned = 0

        # Rehearsal requires knowledge of what we have learned so far
        self.learnedPatterns = []
        self.fracRehearse = fracRehearse
        self.keepPreviousWeights = keepPreviousWeights

    def __str__(self):
        return f"RehearsalHebbian-{self.fracRehearse} FracRehearse"

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

        # rehearse previous states
        randomGen = np.random.default_rng()
        rehearsalPatterns = randomGen.choice(self.learnedPatterns, int(self.fracRehearse*len(self.learnedPatterns)), replace=False)
        for pattern in rehearsalPatterns:
            weightChanges = weightChanges+np.outer(pattern, pattern)

        # Put new states into our array to track them
        self.learnedPatterns.extend(patterns)

        if not self.keepPreviousWeights:
            self.numStatesLearned+=len(patterns)
            return (1/self.numStatesLearned)*weightChanges

        # If numStatesLearned is zero we are learning the first task
        if self.numStatesLearned==0:
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