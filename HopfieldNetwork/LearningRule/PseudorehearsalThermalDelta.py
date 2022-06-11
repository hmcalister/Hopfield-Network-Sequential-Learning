from .AbstractPseudorehearsalLearningRule import AbstractPseudorehearsalLearningRule
from typing import List
import numpy as np


class PseudorehearsalThermalDelta(AbstractPseudorehearsalLearningRule):

    def __init__(self, maxEpochs:int=100, temperature:np.float64=1, temperatureDecay:np.float64=0,
        numRehearse:np.int64=0, fracRehearse:np.float64=0, numPseudorehearsalSamples:np.int64=0,
        updateRehearsalStatesFreq:str="Epoch", keepFirstTaskPseudoitems:bool=False, rejectLearnedStatesAsPseudoitems:bool=False,
        requireUniquePseudoitems:bool=True, trainUntilStable:bool=False):
        """
        Create a new RehearsalThermalDelta Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 100
            temperature (np.float64, optional): The temperature of the learning rule
                Defaults to 1
            temperatureDecay (np.float64, optional): Value to decay temperature by linearly each epoch
                Defaults to 0
            numRehearse (np.int64, optional): The number of tasks to rehearse. Absolute, not relative.
                Defaults to 0. Still randomly selected. If also present with fracRehearse, fracRehearse takes priority
            fracRehearse (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 0. (all patterns)
            updateRehearsalStatesFreq (str, optional): When to update the rehearsal states. 'Epoch' chooses new states every call/epoch
                'Task' chooses new states every task
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False.
        """

        super().__init__(
            numRehearse = numRehearse, 
            fracRehearse = fracRehearse, 
            numPseudorehearsalSamples = numPseudorehearsalSamples,
            updateRehearsalStatesFreq = updateRehearsalStatesFreq, 
            keepFirstTaskPseudoitems = keepFirstTaskPseudoitems, 
            rejectLearnedStatesAsPseudoitems = rejectLearnedStatesAsPseudoitems,
            requireUniquePseudoitems = requireUniquePseudoitems
        )

        self.updateSteps = 1
        self.trainUntilStable = trainUntilStable

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs

        self.temperatureDecay = temperatureDecay
        self.initTemperature = temperature
        self.temperature = temperature
      

    def __str__(self):
        rejectStr = ""
        if self.rejectLearnedStatesAsPseudoitems:
            rejectStr = "RejectLearnedStates"

        if self.numRehearse!=0:
            return f"PseudoehearsalThermalDelta-{self.maxEpochs} MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay {self.numRehearse}NumRehearse {self.updateRehearsalStatesFreq}UpdateFreq {rejectStr}"
        return f"PseudoehearsalThermalDelta-{self.maxEpochs} MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay {self.fracRehearse}FracRehearse {self.updateRehearsalStatesFreq}UpdateFreq {rejectStr}"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """
        weightChanges = np.zeros_like(self.network.weights)
        for pattern in patterns:
            resultState = self.findRelaxedState(pattern.copy())
            phi = (np.dot(self.network.weights, pattern))
            weightChanges = weightChanges+np.outer(pattern-resultState, pattern)*np.exp(-1*np.linalg.norm(phi) / self.temperature)

        if self.updateRehearsalStatesFreq=="Epoch":
            self.updateRehearsalPatterns()


        heteroassociativeNoiseRatioCopy = self.heteroassociativeNoiseRatio
        self.heteroassociativeNoiseRatio = 0
        for pattern in self.rehearsalPatterns:
            resultState = self.findRelaxedState(pattern)
            phi = (np.dot(self.network.weights, pattern))
            weightChanges = weightChanges+np.outer(pattern-resultState, pattern)*np.exp(-1*np.linalg.norm(phi) / self.temperature)
        self.heteroassociativeNoiseRatio = heteroassociativeNoiseRatioCopy
        
        self.temperature -= self.temperatureDecay

        return self.network.weights + weightChanges

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        super().finishTask(taskPatterns)

        self.temperature = self.initTemperature