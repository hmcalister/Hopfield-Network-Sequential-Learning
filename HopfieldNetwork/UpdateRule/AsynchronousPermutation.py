from .AbstractUpdateRule import AbstractUpdateRule
import numpy as np

class AsynchronousPermutation(AbstractUpdateRule):

    MAX_STEPS = 500

    def __init__(self, activationFunction, energyFunction):
        """
        Create a new AsynchronousPermutationUpdateRule, which updates only the unstable (E>0) units one at a time in a random order

        Args:
            activationFunction (AbstractActivationFunction): The activation function to use with this update rule
            energyFunction (AbstractEnergyFunction): The energy function to determine what units to update (only unstable units)
        """
        super().__init__(activationFunction)
        self.energyFunction = energyFunction

    def __call__(self, currentState:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Find the next state from a current state and weights

        Args:
            currentState (np.ndarray): The current state of the network. Must have dimension N and type float64
            weights (np.ndarray): The weights of the network. Must have dimension N*N and type float64

        Returns:
            np.ndarray: The next state of the network
        """

        nextState=currentState.copy()
        energies = self.energyFunction(currentState, weights)
        updateOrder = np.where(energies<0)
        np.random.shuffle(updateOrder)
        # print(f"{energies=}")
        # print(f"{updateOrder=}")
        
        for updateIndex in updateOrder:
            nextState[updateIndex] = self.activationFunction(np.dot(weights[updateIndex], nextState))
    
        return nextState

    def __str__(self):
        return f"AsynchronousPermutation({self.energyFunction})"