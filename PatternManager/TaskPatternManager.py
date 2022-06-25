from typing import List, Tuple
import numpy as np

class TaskPatternManager():
    """
    A manager for the tasks and nearby mappings for one task in a sequential learning problem
    """

    def __init__(self, name:str, taskPatterns:List[np.ndarray], nearbyMappings:List[Tuple[np.ndarray, np.ndarray]]):
        """
        Create a new task manager given the patterns and nearby mappings for this task

        Args:
            name (str): The name of this task
            taskPatterns (List[np.ndarray]): The task patterns for this task
            nearbyMappings (List[Tuple[np.ndarray, np.ndarray]]): The mappings from nearby patterns to task patterns
        """

        self.name:str = name
        self.taskPatterns:List[np.ndarray] = taskPatterns
        self.nearbyMappings:List[Tuple[np.ndarray, np.ndarray]] = nearbyMappings
        # The final epoch this task was trained on
        self.startEpoch:int = 0

    def __str__(self):
        return f"{self.name}: {len(self.taskPatterns)} States"

    def getTaskPatterns(self)->List[np.ndarray]:
        """
        Get the task patterns for this task

        Returns:
            List[np.ndarray]: The task patterns for this task
        """

        return self.taskPatterns

    def getNearbyMappings(self)->List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the nearby mappings for this task

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: The nearby mappings for this task
        """

        return self.nearbyMappings