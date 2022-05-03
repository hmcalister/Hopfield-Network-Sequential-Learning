from typing import List, Union

from .AbstractHopfieldNetwork import AbstractHopfieldNetwork

from .UpdateRule.AbstractUpdateRule import AbstractUpdateRule
from .UpdateRule.AsynchronousList import AsynchronousList

from .EnergyFunction.AbstractEnergyFunction import AbstractEnergyFunction
from .EnergyFunction.BipolarEnergyFunction import BipolarEnergyFunction

from .UpdateRule.ActivationFunction.AbstractActivationFunction import AbstractActivationFunction
from .UpdateRule.ActivationFunction.BipolarHeaviside import BipolarHeaviside

from .LearningRule.AbstractLearningRule import AbstractLearningRule
from .LearningRule.Hebbian import Hebbian

import numpy as np

class RelaxationException(Exception):
    pass

class BipolarHopfieldNetwork(AbstractHopfieldNetwork):

    def __init__(self, 
                N:int, 
                weights:np.ndarray=None,
                selfConnections:bool=False):
        """
        Create a new Bipolar Hopfield network with N units.
        Almost all of the options are chosen, enforced for consistency.

        ActivationFunction is BipolarHeaviside.
        UpdateRule is AsynchList.
        LearningRule is Hebbian.
        EnergyFunction is BipolarEnergy.
        
        If weights is supplied, the network weights are set to the supplied weights.

        Args:
            N (int): The size of the network, how many units to have.
            weights (np.ndarray, optional): The weights of the network, if supplied. Intended to be used to recreate a network for testing.
                If supplied, must be a 2-D matrix of float64 with size N*N. If None, random weights are created uniformly around 0.
                Defaults to None.
            selfConnections (bool, optional): Determines if self connections are allowed or if they are zeroed out during learning
                Defaults to False. (No self connections)

        Raises:
            ValueError: If the given weight matrix is not N*N and not None.
        """

        super().__init__(
            N=N,
            energyFunction=BipolarEnergyFunction(),
            activationFunction=BipolarHeaviside(),
            updateRule=AsynchronousList(BipolarHeaviside()),
            learningRule=Hebbian(),
            weights=weights,
            selfConnections=selfConnections
        )

    def __str__(self):
        return ("Hopfield Network: BipolarHopfieldNetwork\n"
            + super().__str__())

    def setState(self, state:np.ndarray):
        """
        Set the state of this network.

        Given state must be a float64 vector of size N. Any other type will raise a ValueError.

        Args:
            state (np.ndarray): The state to set this network to.

        Raises:
            ValueError: If the given state is not a float64 vector of size N.
        """

        super().setState(state.copy())

    def learnPatterns(self, patterns:List[np.ndarray])->None:
        """
        Learn a set of patterns given. This method will use the learning rule given at construction to learn the patterns.
        The patterns are given as a list of np.ndarrays which must each be a vector of size N.

        Args:
            patterns (List[np.ndarray]): The patterns to learn. Each np.ndarray must be a float64 vector of length N (to match the state)

        Returns: None
        """

        return super().learnPatterns(patterns.copy())