from .AbstractUpdateRule import AbstractUpdateRule
import numpy as np

class AsynchronousList(AbstractUpdateRule):
    MAX_STEPS = 500

    def __init__(self, activationFunction):
        """
        Create a new AsynchronousListUpdateRule, which updates all units in the network one at a time in a random order

        Args:
            activationFunction (AbstractActivationFunction): The activation function to use with this update rule
        """
        super().__init__(activationFunction)

    def __call__(self, currentState:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Find the next state from a current state and weights

        Args:
            currentState (np.ndarray): The current state of the network. Must have dimension N and type float64
            weights (np.ndarray): The weights of the network. Must have dimension N*N and type float64

        Returns:
            np.ndarray: The next state of the network
        """

        nextState=currentState
        updateOrder = np.arange(currentState.shape[0])
        np.random.shuffle(updateOrder)

        for updateIndex in updateOrder:
            nextState[updateIndex] = self.activationFunction(np.dot(weights[updateIndex], nextState))
            
        return nextState

    def __str__(self): 
        return "AsynchronousList"