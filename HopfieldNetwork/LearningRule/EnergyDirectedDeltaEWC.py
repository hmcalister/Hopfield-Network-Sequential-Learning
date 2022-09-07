from HopfieldNetwork.LearningRule.EWCTerm import *
from .AbstractLearningRule import AbstractLearningRule
from .EWCTerm.AbstractEWCTerm import AbstractEWCTerm
from .EWCTerm.NoneEWCTerm import NoneEWCTerm
from typing import List
import numpy as np
import warnings


class EnergyDirectedDeltaEWC(AbstractLearningRule):

    def __init__(self, maxEpochs:int=100, trainUntilStable:bool=False, alpha:np.float64=0, 
        ewcTermGenerator:AbstractEWCTerm=NoneEWCTerm, ewcLambda:np.float64 = 0,
        useOnlyFirstEWCTerm:bool=False, vanillaEpochsFactor:np.float64 = 0):
        """
        Create a new EnergyDirectedDeltaEWC Learning Rule
        Delta rule calculates the network state after a single update step
        Then compares this resultState with the targetState and uses this to calculate a weight change.

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 10
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False
            alpha (np.float64, optional): The learning rate hyperparameter for energy directed learning
                Defaults to 0.
            ewcTermGenerator (AbstractOmega): The method of selecting weight importance. Defaults to None (0)
            ewcLambda (np.float64): Determines the importance of the EWC term in weight updates
            useOnlyFirstEWCTerm (bool, optional): Use only the EWC from the first task. Defaults to false
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
        
        self.numEpochs = 0
        self.ewcTermGenerator = ewcTermGenerator
        self.ewcLambda = ewcLambda
        self.useOnlyFirstEWCTerm = useOnlyFirstEWCTerm
        self.ewcTerms:List[AbstractEWCTerm.EWCTerm] = []
        self.vanillaEpochsFactor = vanillaEpochsFactor

        self.ewcTermGenerator.startTask()

    def __str__(self):
            
        return f"EnergyDirectedDeltaEWC"

    def infoString(self):
            
        return f"EnergyDirectedDeltaEWC-{self.maxEpochs} MaxEpochs"

    def setNetworkReference(self, network):
        super().setNetworkReference(network)
        self.ewcTermGenerator.setNetworkReference(network)

    def phi(self, x: np.ndarray):
        return np.tanh(x)

    def derivative_phi(self, x: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (1/np.cosh(x))**2

    def calculateVanillaChange(self, patterns:List[np.ndarray])->np.ndarray:
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
            weightChanges = weightChanges + 0.05 * patternUpdates


        return self.network.weights + weightChanges

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
        vanillaTerm /= np.max(np.abs(vanillaTerm))

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
        self.numEpochs+=1
        return weight/np.max(np.abs(weight))

    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        self.numEpochs = 0
        self.numStatesLearned+=len(taskPatterns)
        self.ewcTermGenerator.finishTask()
        if not self.useOnlyFirstEWCTerm or len(self.ewcTerms)==0:
            self.ewcTerms.append(
                self.ewcTermGenerator.generateTerm(self.network.weights.copy(), taskPatterns.copy())
            )
        self.ewcTermGenerator.startTask()