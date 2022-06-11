from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class RehearsalThermalDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, temperature:np.float64=1, temperatureDecay:np.float64=0,
        numRehearse:np.int64=0, fracRehearse:np.float64=0, rehearseFirstTaskOnly:bool = False,
        updateRehearsalStatesFreq:str="Epoch", trainUntilStable:bool=False):
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
            rehearseFirstTaskOnly (bool, option): A flag to rehearse only the first task given.
                Defaults to False, rehearse all tasks.
            updateRehearsalStatesFreq (str, optional): When to update the rehearsal states. 'Epoch' chooses new states every call/epoch
                'Task' chooses new states every task
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False.
        """

        # Delta rule requires a single update step 
        self.updateSteps = 1
        self.trainUntilStable = trainUntilStable

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs

        # Currently, numStatesLearned unused by Delta
        self.numStatesLearned = 0

        self.temperatureDecay = temperatureDecay
        self.initTemperature = temperature
        self.temperature = temperature

        # Rehearsal requires knowledge of what we have learned so far
        self.learnedPatterns = []
        self.rehearsalPatterns = []
        self.numRehearse = numRehearse
        self.fracRehearse = fracRehearse
        self.updateRehearsalStatesFreq = updateRehearsalStatesFreq
        self.rehearseFirstTaskOnly = rehearseFirstTaskOnly

    def __str__(self):

        if self.numRehearse!=0:
            return f"RehearsalThermalDelta-{self.maxEpochs} MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay {self.numRehearse}NumRehearse {self.updateRehearsalStatesFreq}UpdateFreq"
        return f"RehearsalThermalDelta-{self.maxEpochs} MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay {self.fracRehearse}FracRehearse {self.updateRehearsalStatesFreq}UpdateFreq"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        # The weights changes start as a zero matrix
        weights = self.network.weights
        weightChanges = np.zeros_like(self.network.weights)
        for i in range(len(patterns)):
            # Get pattern i and the resultState i
            pattern = patterns[i].copy()
            resultState = self.findRelaxedState(pattern)
            phi = (np.dot(self.network.weights, pattern))

            # The weight changes of this pattern is the outer product of
            # The difference of pattern and result state and this pattern
            # Scale factor is 0.5 so updates are -1 and 1 (one from both sides)
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

        return weights + weightChanges

    def updateRehearsalPatterns(self)->List[np.ndarray]:
        """
        Generate and return a list of patterns from the learned patterns

        Returns:
            List[np.ndarray]: A subset of the learned patterns to rehearse
        """

        randomGen = np.random.default_rng()
        if self.fracRehearse!=0:
            self.rehearsalPatterns = randomGen.choice(self.learnedPatterns, int(self.fracRehearse*len(self.learnedPatterns)), replace=False)
            return  self.rehearsalPatterns

        if self.numRehearse!=0:
            self.rehearsalPatterns = randomGen.choice(self.learnedPatterns, min(self.numRehearse, len(self.learnedPatterns)), replace=False)
            return  self.rehearsalPatterns
        
        # If BOTH are 0
        return self.rehearsalPatterns

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.temperature = self.initTemperature

        # Put new states into our array to track them
        if not self.rehearseFirstTaskOnly or len(self.learnedPatterns) == 0:
            self.learnedPatterns.extend(taskPatterns)
        
        self.numStatesLearned+=len(taskPatterns)

        if self.updateRehearsalStatesFreq=="Task":
            self.updateRehearsalPatterns()