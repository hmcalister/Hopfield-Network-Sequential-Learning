from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class RehearsalDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, fracRehearse:np.float64=1, keepPreviousWeights:bool=True, trainUntilStable:bool=True):
        """
        Create a new RehearsalDelta Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 100
            fracRehears (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 1. (all patterns)
            keepPreviousWeights (bool, optional): Flag to keep the previous weights of the network, or throw these away
                Defaults to True.
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to True, train the current pattern until stable.
        """

        # Delta rule requires a single update step 
        self.updateSteps = 1
        self.trainUntilStable = trainUntilStable

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs

        # Currently, numStatesLearned unused by Delta
        self.numStatesLearned = 0

        # Rehearsal requires knowledge of what we have learned so far
        self.learnedPatterns = []
        self.fracRehearse = fracRehearse
        self.keepPreviousWeights = keepPreviousWeights

    def __str__(self):
        return f"RehearsalDelta-{self.maxEpochs} Max Epochs {self.fracRehearse} FracRehearse"

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

        # The weights changes start as a zero matrix
        weightChanges = np.zeros_like(weights)
        for i in range(len(patterns)):
            # Get pattern i and the resultState i
            pattern = patterns[i]
            resultState = resultStates[i]

            # The weight changes of this pattern is the outer product of
            # The difference of pattern and result state and this pattern
            # Scale factor is 0.5 so updates are -1 and 1 (one from both sides)
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)
        
        # rehearse previous states
        randomGen = np.random.default_rng()
        rehearsalPatterns = randomGen.choice(self.learnedPatterns, int(self.fracRehearse*len(self.learnedPatterns)), replace=False)
        for pattern in rehearsalPatterns:
            weightChanges = weightChanges+np.outer(pattern, pattern)
        
        return weights + weightChanges