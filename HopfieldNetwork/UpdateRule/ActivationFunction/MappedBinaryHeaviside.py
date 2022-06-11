from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, heaviside

class MappedBinaryHeaviside(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a BinaryHeaviside activation function
        If input is 0, return 0
        Maps (-inf, 0] to 0, (0,inf) to 1
        Output {0, 1}
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        out=array(heaviside(state,0))
        return out

    def __str__(self):
        return "MappedBinaryHeaviside"