from .AbstractPatternDistanceFunction import AbstractPatternDistanceFunction
import numpy as np

class EuclideanPatternDistance(AbstractPatternDistanceFunction):
    """
    Define a callable class that measures the Euclidean/L2 pattern distance
    """

    def __init__(self):
        pass

    def __call__(self, pattern1:np.ndarray, pattern2:np.ndarray)->np.float64:
        """
        Given two patterns, return the distance between the two patterns.
        The two patterns must have the same size.

        Args:
            pattern1 (np.ndarray): The first pattern
            pattern2 (np.ndarray): The second pattern

        Returns:
            np.float64: The distance between pattern1 and pattern2
        """

        difference = (pattern1-pattern2)**2
        return np.sum(difference)
