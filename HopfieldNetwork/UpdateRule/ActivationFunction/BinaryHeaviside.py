from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, heaviside

class BinaryHeaviside(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a BinaryHeaviside activation function
        If input is zero, return 0
        Output {0, 1}
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        out=array(heaviside(state,0))
        return out

    def __str__(self):
        return "BinaryHeaviside"