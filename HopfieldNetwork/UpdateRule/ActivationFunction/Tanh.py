from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, tanh

class Tanh(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a Tanh activation function
        Output [-1, 1]
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        return array(tanh(state))

    def __str__(self):
        return ("Tanh")