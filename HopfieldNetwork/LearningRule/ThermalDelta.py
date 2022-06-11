from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class ThermalDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, trainUntilStable:bool=False, temperature:np.float64 = 1, temperatureDecay:np.float64=0):
        """
        Create a new ThermalDelta Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 10
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False
            temperature (np.float64, optional): The temperature of the learning rule
                Defaults to 1
            temperatureDecay (np.float64, optional): Value to decay temperature by linearly each epoch
                Defaults to 0
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

        self.temperatureDecay = temperatureDecay
        self.initTemperature = temperature
        self.temperature = temperature

    def __str__(self):
            
        return f"ThermalDelta-{self.maxEpochs}MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay"

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
            resultState = self.findRelaxedState(pattern.copy())
            phi = (np.dot(self.network.weights, pattern))
            weightChanges = weightChanges+np.outer(pattern-resultState, pattern)*np.exp(-1*np.linalg.norm(phi) / self.temperature)

        self.temperature -= self.temperatureDecay

        return self.network.weights + weightChanges

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.temperature = self.initTemperature
        self.numStatesLearned+=len(taskPatterns)