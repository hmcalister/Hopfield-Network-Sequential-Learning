from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class EnergyDirectedDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, trainUntilStable:bool=False, alpha:np.float64=0):
        """
        Create a new EnergyDirectedDelta Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 10
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False
            alpha (np.float64, optional): The learning rate hyperparameter for energy directed learning
                Defaults to 0.
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
        
        self.alpha = alpha
        # self.temperatureDecay = temperatureDecay
        # self.initTemperature = temperature
        # self.temperature = temperature

    def __str__(self):
            
        return f"EnergyDirectedDelta"

    def infoString(self):
            
        return f"EnergyDirectedDelta-{self.maxEpochs} MaxEpochs"

    def phi(self, x: np.ndarray):
        return 1/(1+np.exp(-x))

    def derivative_phi(self, x: np.ndarray):
        # return 1
        return self.phi(x) * (1-self.phi(x))

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
        for pattern in patterns:
            patternUpdates = np.zeros_like(weightChanges)
            resultState = self.findRelaxedState(pattern.copy())
            patternUpdates = np.outer(pattern-resultState, pattern)
            hebbianMatrix = np.outer(pattern, pattern)
            energyDirectedTerms = self.alpha * hebbianMatrix * self.derivative_phi(-self.network.weights * hebbianMatrix)
            patternUpdates[patternUpdates==0] = energyDirectedTerms[patternUpdates==0]
            weightChanges = weightChanges+patternUpdates


        return self.network.weights + weightChanges

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.numStatesLearned+=len(taskPatterns)