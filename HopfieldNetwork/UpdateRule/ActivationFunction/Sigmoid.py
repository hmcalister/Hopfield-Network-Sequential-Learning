from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, exp

class Sigmoid(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a Sigmoid activation function
        Output [0, 1]
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        return array(1/(1+exp(-state)))

    def __str__(self):
        return "Sigmoid"