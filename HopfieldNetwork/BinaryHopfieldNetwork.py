from typing import List, Union

from .AbstractHopfieldNetwork import AbstractHopfieldNetwork

from .UpdateRule.AbstractUpdateRule import AbstractUpdateRule
from .UpdateRule.AsynchronousList import AsynchronousList

from .EnergyFunction.AbstractEnergyFunction import AbstractEnergyFunction
from .EnergyFunction.BinaryEnergyFunction import BinaryEnergyFunction

from .UpdateRule.ActivationFunction.AbstractActivationFunction import AbstractActivationFunction
from .UpdateRule.ActivationFunction.BinaryHeaviside import BinaryHeaviside

from .LearningRule.AbstractLearningRule import AbstractLearningRule
from .LearningRule.Hebbian import Hebbian
from .LearningRule.MappedBinaryHebbian import MappedBinaryHebbian

import numpy as np

class RelaxationException(Exception):
    pass

class BinaryHopfieldNetwork(AbstractHopfieldNetwork):

    def __init__(self, 
                N:int,
                learningRule:str="MappedHebbian",
                weights:np.ndarray=None,
                selfConnections:bool=False):
        """
        Create a new Binary Hopfield network with N units.
        Almost all of the options are chosen, enforced for consistency.

        ActivationFunction is BinaryHeaviside.
        UpdateRule is AsynchList.
        LearningRule is Hebbian or MappedHebbian.
        EnergyFunction is BinaryEnergy.
        
        If weights is supplied, the network weights are set to the supplied weights.

        Args:
            N (int): The size of the network, how many units to have.
            learningRule (str, optional): The learning rule to use for this network. Can be either "Hebbian" or "MappedHebbian" (default).
            weights (np.ndarray, optional): The weights of the network, if supplied. Intended to be used to recreate a network for testing.
                If supplied, must be a 2-D matrix of float64 with size N*N. If None, random weights are created uniformly around 0.
                Defaults to None.
            selfConnections (bool, optional): Determines if self connections are allowed or if they are zeroed out during learning
                Defaults to False. (No self connections)

        Raises:
            ValueError: If the given weight matrix is not N*N and not None.
        """

        if learningRule=="Hebbian":
            learningRule=Hebbian()
        elif learningRule=="MappedHebbian":
            learningRule=MappedBinaryHebbian()
        else:
            print("ERROR: LEARNING RULE NOT KNOWN")
            exit()

        super().__init__(
            N=N,
            energyFunction=BinaryEnergyFunction(),
            activationFunction=BinaryHeaviside(),
            updateRule=AsynchronousList(BinaryHeaviside()),
            learningRule=learningRule,
            weights=weights,
            selfConnections=selfConnections
        )

    def __str__(self):
        return ("Hopfield Network: BinaryHopfieldNetwork\n"
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

    def compareState(self, state:np.ndarray)->bool:
        """
        Compares the given state to the state of the network right now
        Returns True if the two states are the same, false otherwise
        Accepts the inverse state as equal

        Args:
            state (np.ndarray): The state to compare to

        Returns:
            bool: True if the given state is the same as the network state, false otherwise
        """

        return np.array_equal(self.getState(), state) or np.array_equal(-1*self.getState()+1, state)

    def learnPatterns(self, patterns:List[np.ndarray], allTaskPatterns:List[List[np.ndarray]]=None)->None:
        """
        Learn a set of patterns given. This method will use the learning rule given at construction to learn the patterns.
        The patterns are given as a list of np.ndarrays which must each be a vector of size N.

        Args:
            patterns (List[np.ndarray]): The patterns to learn. Each np.ndarray must be a float64 vector of length N (to match the state)
            allTaskPatterns (List[List[np.ndarray]] or None, optional): If given, will track the task pattern stability by epoch during training.
                Passed straight to measureTaskPatternAccuracy. Defaults to None.

        Returns: None or List[Tuple[List[np.float64], int]]]
            If allTaskPatterns is None, returns None
            If allTaskPatterns is present, returns a list over epochs of tuples. Tuples are of form (list of task accuracies, num stable learned patterns overall)
        """

        return super().learnPatterns(patterns.copy(), allTaskPatterns)