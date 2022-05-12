from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray

class Linear(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a Linear activation function
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        # Linear simply returns state
        return state

    def __str__(self):
        return "Linear"
