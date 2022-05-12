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

        # Set the next state to this state, keeping memory management in mind
        nextState=currentState.copy()
        # Calculate the energy of the units
        energies = self.energyFunction(currentState, weights)
        # ANd note only the energies above 0 (unstable)
        updateOrder = np.where(energies>=0)
        # Then permute this order
        np.random.shuffle(updateOrder)

        # print(f"{energies=}")
        # print(f"{updateOrder=}")
        
        # For each index to be updated
        for updateIndex in updateOrder:
            if inputNoise is None:
                noiseVector=0
            else:
                noiseVector = self.getInputNoise(inputNoise, np.dot(weights, nextState))

            # Update that index
            nextState[updateIndex] = self.activationFunction(
                np.dot(weights[updateIndex], nextState+noiseVector)
            )
    
        return nextState

    def __str__(self):
        return f"AsynchronousPermutation({self.energyFunction})"