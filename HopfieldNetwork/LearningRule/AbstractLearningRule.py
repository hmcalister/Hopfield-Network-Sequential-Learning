from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..AbstractHopfieldNetwork import AbstractHopfieldNetwork, RelaxationException

class AbstractLearningRule(ABC):

    @abstractmethod
    def __init__(self):
        """
        Create a new Abstract Learning Rule
        """

        # Define how many update steps are needed for each pattern
        self.updateSteps = 0
        # Define the maximum number of epochs to learn for
        self.maxEpochs = 0
        # Track how many states have been learned so far
        self.numStatesLearned = 0
        # Flag to determine if we should train until stable, defaults to True
        self.trainUntilStable = True
        # Set the heteroassociativeNoiseRatio, how many bits to flip during training
        # This is set at a call to network.learnPatterns
        self.heteroassociativeNoiseRatio = 0
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractLearningRule"

    @abstractmethod
    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        pass

    def setNetworkReference(self, network:AbstractHopfieldNetwork):
        """
        Set a reference to a HopfieldNetwork, for use in getting weights, checking stability...`

        Args:
            network (AbstractHopfieldNetwork): The network to set a reference to
        """

        self.network = network

    def setHeteroassociativeNoiseRatio(self, heteroassociativeNoiseRatio:np.float64):
        """
        Set the heteroassociativeNoiseRatio 

        Args:
            heteroassociativeNoiseRatio (np.float64): The new heteroassociativeNoiseRatio
        """

        self.heteroassociativeNoiseRatio = heteroassociativeNoiseRatio

    def clearHeteroassociativeNoiseRatio(self):
        """
        Set the heteroassociativeNoiseRatio to 0
        """

        self.heteroassociativeNoiseRatio=0

    def findRelaxedState(self, pattern:np.ndarray)->np.ndarray:
        """
        Given a pattern, find the relaxed pattern from the network (after a set number of update steps) 

        Args:
            pattern (np.ndarray): The initial pattern the network will start in 

        Returns:
            np.ndarray: The final pattern, after relaxing some steps 
        """

        self.network.setState(pattern)
        try:
            self.network.relax(self.updateSteps)
        except Exception as e:
            pass
        if self.heteroassociativeNoiseRatio>0:
            self.network.setState(self.network.invertStateUnits(self.network.getState(), self.heteroassociativeNoiseRatio))
        return self.network.getState()