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
    def __call__(self, currentState:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Find the next state from a current state and weights

        Args:
            currentState (np.ndarray): The current state of the network. Must have dimension N and type float64
            weights (np.ndarray): The weights of the network. Must have dimension N*N and type float64

        Returns:
            np.ndarray: The next state of the network
        """
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractUpdateRule"