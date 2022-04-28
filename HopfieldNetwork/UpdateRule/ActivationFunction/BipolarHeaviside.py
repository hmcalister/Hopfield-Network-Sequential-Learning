from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, heaviside

class BipolarHeaviside(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a BipolarHeaviside activation function
        If input is zero, return 1
        Output {-1, 1}
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        out=array(heaviside(state,1))
        out[out==0] = -1
        return out

    def __str__(self):
        return "BipolarHeaviside"