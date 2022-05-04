from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, heaviside

class BinaryHeaviside(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a BinaryHeaviside activation function
        If input is 0.5, return 0
        Maps (-inf, 0.5] to 0, (0.5,inf) to 1
        Output {0, 1}
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        out=array(heaviside(state-0.5,0))
        return out

    def __str__(self):
        return "BinaryHeaviside"