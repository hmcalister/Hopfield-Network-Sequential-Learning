from .ActivationFunction import AbstractActivationFunction
from abc import ABC, abstractmethod
import numpy as np


class AbstractUpdateRule(ABC):
    MAX_STEPS = 500

    @abstractmethod
    def __init__(self, activationFunction:AbstractActivationFunction):
        self.activationFunction = activationFunction
        pass
        
    @abstractmethod
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
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractUpdateRule"

    def getInputNoise(self, inputNoise:str, currentState:np.ndarray)->np.ndarray:
        """
        Generate some input noise.

        Args:
            inputNoise (str or None, optional): String on whether to apply input noise to the units before activation
                - "Absolute": Apply absolute noise to the state, a Gaussian of mean 0 std 1
                - "Relative": Apply relative noise to the state, a Gaussian of mean and std determined by the state vector
                - None: No noise. Default
            currentState (np.ndarray): The current state of the network, to be used for relative noise 

        Returns:
            np.ndarray: The noise vector
        """

        # Apply noise
        if inputNoise is None:
            noiseVector = 0
        elif inputNoise=="Absolute":
            noiseVector = np.random.normal(0, 1, currentState.shape[0])
        elif inputNoise=="Relative":
            if np.abs(np.mean(currentState))==0: return 0
            noiseVector = np.random.normal(0, np.abs(np.mean(currentState)), currentState.shape[0])
            # print(f"{np.max(noiseVector)=},\n{np.max(np.abs(currentState))=}\n")
        else:
            print("INPUT NOISE NOT RECOGNIZED")
            exit()

        return noiseVector

    
