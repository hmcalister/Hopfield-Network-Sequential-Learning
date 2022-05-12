from .AbstractUpdateRule import AbstractUpdateRule
import numpy as np

class Synchronous(AbstractUpdateRule):

    MAX_STEPS = 500

    def __init__(self, activationFunction):
        """
        Create a new SynchronousUpdateRule, which updates all units in the network at once

        Args:
            activationFunction (AbstractActivationFunction): The activation function to use with this update rule
        """
        super().__init__(activationFunction)

    def __call__(self, currentState:np.ndarray, weights:np.ndarray, inputNoise:str=None)->np.ndarray:
        """
        Find the next state from a current state and weights

        Args:
            currentState (np.ndarray): The current state of the network. Must have dimension N and type float64
            weights (np.ndarray): The weights of the network. Must have dimension N*N and type float64
            inputNoiseRatio (str or None, optional): String on whether to apply input noise to the units before activation
                - "Absolute": Apply absolute noise to the state, a Gaussian of mean 0 std 1
                - "Relative": Apply relative noise to the state, a Gaussian of mean and std determined by the state vector
                - None: No noise. Default

        Returns:
            np.ndarray: The next state of the network
        """

        noiseVector = self.getInputNoise(inputNoise, currentState)

        # Update the entire unit vector at once and return it
        nextState = self.activationFunction(np.dot(weights, currentState+noiseVector))
        return nextState

    def __str__(self):
        return "Synchronous"