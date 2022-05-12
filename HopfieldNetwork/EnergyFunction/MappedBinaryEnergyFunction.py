from .AbstractEnergyFunction import AbstractEnergyFunction
import numpy as np

class MappedBinaryEnergyFunction(AbstractEnergyFunction):

    def __init__(self):
        """
        Create a new BinaryEnergyFunction that calculates the energy of each unit in a network
        """

    def __call__(self, state:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Given a state (a vector of units) and weights, calculate the energy of each unit
        If energy are negative, unit is stable

        Args:
            state (np.ndarray): The vector of units to calculate the energies of. Must be of size N
            weights (np.ndarray): The matrix of weights. Must be of size of N*N.

        Returns:
            np.ndarray: The vector of energies corresponding to the units in the state
        """

        # Calculate the next states of the network
        nextState = np.dot(weights, 2*state-1)
        # Then map the states to binary field
        # And calculate if states are the same or different by multiplication
        # Now if states are different, energies are positive
        energies = -1*(2*nextState-1)*(2*state-1)
        return energies

    def __str__(self):
        return "MappedBinaryEnergyFunction"