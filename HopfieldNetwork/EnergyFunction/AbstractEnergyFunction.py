from abc import ABC, abstractmethod
import numpy as np

class AbstractEnergyFunction(ABC):

    def __init__(self):
        """
        Create a new AbstractEnergyFunction that calculates the energy of each unit in a network
        """
        pass

    @abstractmethod
    def __call__(self, state:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Given a state (a vector of units) and weights, calculate the energy of each unit

        Args:
            state (np.ndarray): The vector of units to calculate the energies of. Must be of size N
            weights (np.ndarray): The matrix of weights. Must be of size of N*N.

        Returns:
            np.ndarray: The vector of energies corresponding to the units in the state
        """

        pass

    @abstractmethod
    def __str__(self):
        return "AbstractEnergyFunction"