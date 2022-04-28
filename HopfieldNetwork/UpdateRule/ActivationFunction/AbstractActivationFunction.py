from abc import ABC, abstractmethod
from numpy import ndarray

class AbstractActivationFunction(ABC):

    def __init__(self):
        """
        Return a new instance of an activation function
        """
        pass

    @abstractmethod
    def __call__(self, state:ndarray)->ndarray:
        """
        Apply this activation function to the given state

        Args:
            state (Variable): The state given that the activation function will be applied to

        Returns:
            Variable: The result of the activation function on the state
        """

        pass

    @abstractmethod
    def __str__(self):
        return "AbstractActivationFunction"