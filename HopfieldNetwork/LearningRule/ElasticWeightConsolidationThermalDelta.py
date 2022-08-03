from HopfieldNetwork.LearningRule.EWCTerm import *
from .AbstractLearningRule import AbstractLearningRule
from .EWCTerm.AbstractEWCTerm import AbstractEWCTerm
from .EWCTerm.NoneEWCTerm import NoneEWCTerm
from typing import List
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

class ElasticWeightConsolidationThermalDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, trainUntilStable:bool=False, temperature:np.float64 = 1, temperatureDecay:np.float64=0, 
        ewcTermGenerator:AbstractEWCTerm=NoneEWCTerm, ewcLambda:np.float64 = 0,
        useOnlyFirstEWCTerm:bool=False, vanillaEpochsFactor:np.float64 = 0):
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
            ewcTermGenerator (AbstractOmega): The method of selecting weight importance. Defaults to None (0)
            ewcLambda (np.float64): Determines the importance of the EWC term in weight updates
            useOnlyFirstEWCTerm (bool, optional): Use only the EWC from the first task. Defaults to false
            vanillaLearningFactor (np.float64, optional): The number of epochs (as ratio of maxEpochs) to learn vanilla
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

        self.numEpochs = 0
        self.ewcTermGenerator = ewcTermGenerator
        self.ewcLambda = ewcLambda
        self.useOnlyFirstEWCTerm = useOnlyFirstEWCTerm
        self.ewcTerms:List[AbstractEWCTerm.EWCTerm] = []

        self.vanillaEpochsFactor = vanillaEpochsFactor

        self.ewcTermGenerator.startTask()

    def __str__(self):
        return f"EWC Learning"

    def infoString(self):
        return f"ElasticWeightConsolidationThermalDelta-{self.maxEpochs} MaxEpochs Temperature{self.temperature} {self.temperatureDecay}Decay {self.ewcTermGenerator.toString()}"

    def setNetworkReference(self, network):
        super().setNetworkReference(network)
        self.ewcTermGenerator.setNetworkReference(network)

    def calculateVanillaChange(self, patterns:List[np.ndarray])->np.ndarray:
        # The weights changes start as a zero matrix
        weightChanges = np.zeros_like(self.network.weights)
        for pattern in patterns:
            resultState = self.findRelaxedState(pattern.copy())
            phi = (np.dot(self.network.weights, pattern))
            weightChanges = weightChanges+np.outer(pattern-resultState, pattern)*np.exp(-1*np.linalg.norm(phi) / self.temperature)
        np.fill_diagonal(weightChanges, 0)
        return weightChanges

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """
        
        # Consider the loss function:
        # J(w) = (w-w_v)**2 + ewcLambda * Omega * (w-w_k)**2
        # Where w_v is the vanilla weight changes from thermal delta
        # And w_k is the ideal weights for task k
        #
        # Notice that taking the derivative and setting to zero we get
        # w = (w_v + ewcLambda * Omega) / (1 + ewcLambda * Omega)
        # This is our approximation of gradient descent

        vanillaTerm = self.network.weights+self.calculateVanillaChange(patterns)
        # vanillaMagnitude = np.max(np.abs(vanillaTerm))
        # vanillaTerm /= vanillaMagnitude

        if self.numEpochs < self.maxEpochs * self.vanillaEpochsFactor:
            weight = vanillaTerm
        else:
            ewcNumerator = np.zeros_like(vanillaTerm)
            ewcDenominator = np.zeros_like(vanillaTerm)
            for e in self.ewcTerms:
                importance = e.getImportance()
                ewcNumerator += importance * e.getTaskWeights()
                ewcDenominator += importance
            weight = (vanillaTerm + self.ewcLambda * ewcNumerator) / (1 + self.ewcLambda * ewcDenominator)           

        self.ewcTermGenerator.epochCalculation(weight=weight)
        self.temperature -= self.temperatureDecay
        self.numEpochs+=1
        return weight

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.numEpochs = 0
        self.temperature = self.initTemperature
        self.numStatesLearned+=len(taskPatterns)
        self.ewcTermGenerator.finishTask()
        if not self.useOnlyFirstEWCTerm or len(self.ewcTerms)==0:
            self.ewcTerms.append(
                self.ewcTermGenerator.generateTerm(self.network.weights.copy(), taskPatterns.copy())
            )
        self.ewcTermGenerator.startTask()