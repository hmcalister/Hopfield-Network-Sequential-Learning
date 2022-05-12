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
        
        # BipolarHeaviside heavisides and maps 0 to 0, then maps all 0 to -1
        out=array(heaviside(state,0))
        out[out==0] = -1
        return out

    def __str__(self):
        return "BipolarHeaviside"