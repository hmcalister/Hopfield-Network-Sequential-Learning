from .AbstractActivationFunction import AbstractActivationFunction
from numpy import array, ndarray, sign

class Sign(AbstractActivationFunction):

    def __init__(self):
        """
        Return a new instance of a Sign activation function
        Output {-1, 0, 1}
        """
        pass

    def __call__(self, state:ndarray)->ndarray:
        
        return array(sign(state))

    def __str__(self):
        return "Sign"