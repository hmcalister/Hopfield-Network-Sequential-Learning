from .AbstractUpdateRule import AbstractUpdateRule
import numpy as np

class MappedBinaryAsynchronousList(AbstractUpdateRule):
    MAX_STEPS = 500

    def __init__(self, activationFunction):
        """
        Create a new MappedBinaryAsynchronousListUpdateRule, which updates all units in the network one at a time in a random order
        This is special as it maps Binary to {-1,1} space before applying 

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

        # Set the next state to the current state, being well aware of memory management
        nextState=currentState.copy()
        # Choose the update order. We are updating all of the units 
        # so we create a list of all indices and permute this
        updateOrder = np.arange(currentState.shape[0])
        np.random.shuffle(updateOrder)

        # For each index in order
        for updateIndex in updateOrder:
            noiseVector = self.inputNoise(2*nextState-1)

            # Update that index
            nextState[updateIndex] = self.activationFunction(
                np.dot(weights[updateIndex], (2*nextState-1) + noiseVector)
            )
            
        return nextState

    def __str__(self): 
        return "MappedBinaryAsynchronousList"