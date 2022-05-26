from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class PseudorehearsalHebbian(AbstractLearningRule):

    def __init__(self, maxEpochs:int=1, numRehearse:np.int64=0, fracRehearse:np.float64=0, numPseudorehearsalSamples:np.int64=0,
        updateRehearsalStatesFreq:str="Epoch", keepPreviousWeights:bool=True, trainUntilStable:bool=False):
        """
        Create a new RehearsalHebbian Learning Rule

        Notice that fracRehearse=1, keepPreviousWeights=False is a "best case scenario" for sequential learning

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 1
            numRehearse (np.int64, optional): The number of tasks to rehearse. Absolute, not relative.
                Defaults to 0. Still randomly selected. If also present with fracRehearse, fracRehearse takes priority
            fracRehears (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 0. (all patterns)
            updateRehearsalStatesFreq (str, optional): When to update the rehearsal states. 'Epoch' chooses new states every call/epoch
                'Task' chooses new states every task
            keepPreviousWeights (bool, optional): Flag to keep the previous weights of the network, or throw these away
                Defaults to True.
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False.
        """

        # Hebbian is a single calculation, no need for updates or multiple epochs
        self.updateSteps = 0
        self.maxEpochs = maxEpochs
        self.trainUntilStable = trainUntilStable

        # Hebbian does actually use numStatesLearned
        self.numStatesLearned = 0

        # Rehearsal requires knowledge of what we have learned so far
        self.learnedPatterns = []
        self.stableStates = []
        self.rehearsalPatterns = []
        self.numRehearse = numRehearse
        self.fracRehearse = fracRehearse
        self.numPseudorehearsalSamples = numPseudorehearsalSamples
        self.updateRehearsalStatesFreq = updateRehearsalStatesFreq
        self.keepPreviousWeights = keepPreviousWeights

    def __str__(self):
        if self.numRehearse!=0:
            return f"PseudorehearsalHebbian-{self.numRehearse}NumRehearse {self.updateRehearsalStatesFreq}UpdateFreq {self.numPseudorehearsalSamples}numPseudorehearsalSamples"
        return f"PseudorehearsalHebbian-{self.fracRehearse}FracRehearse {self.updateRehearsalStatesFreq}UpdateFreq {self.numPseudorehearsalSamples}numPseudorehearsalSamples"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        # Weight changes start as zero matrix
        weights = self.network.weights
        weightChanges = np.zeros_like(self.network.weights)

        # For every pattern calculate a weight change
        for pattern in patterns:
            # The weight change of this pattern is outer product of pattern with itself
            weightChanges = weightChanges+np.outer(pattern, pattern)

        # rehearse previous states
        if self.updateRehearsalStatesFreq=="Epoch":
            self.updateRehearsalPatterns()
        for pattern in self.rehearsalPatterns:
            weightChanges = weightChanges+np.outer(pattern, pattern)

        if not self.keepPreviousWeights:
            self.numStatesLearned+=len(patterns)
            return (1/self.numStatesLearned)*weightChanges

        # If numStatesLearned is zero we are learning the first task
        if self.numStatesLearned==0:
            # And return the scaled weight changes
            return weights+(1/len(patterns))*weightChanges

        # Otherwise, we are on a sequential task
        else:
            # First, scale the weights back up by number of patterns learned (before this task)
            weights = weights*self.numStatesLearned
            # Add the weight changes
            weights += weightChanges
            # And scale back down
            weights /= (self.numStatesLearned+len(patterns))
            return weights

    def updateRehearsalPatterns(self)->List[np.ndarray]:
        """
        Generate and return a list of patterns from the learned patterns

        Returns:
            List[np.ndarray]: A subset of the learned patterns to rehearse
        """

        randomGen = np.random.default_rng()

        if self.numRehearse!=0:
            self.rehearsalPatterns = randomGen.choice(self.stableStates, min(self.numRehearse, len(self.stableStates)), replace=False)
            return  self.rehearsalPatterns
        
        if self.fracRehearse!=0:
            self.rehearsalPatterns = randomGen.choice(self.stableStates, int(self.fracRehearse*len(self.stableStates)), replace=False)
            return  self.rehearsalPatterns

        # If BOTH are 0
        return self.rehearsalPatterns

    def updateStableStates(self)->List[np.ndarray]:
        """
        Update the list of stable states, appending to any previous stable states

        Returns:
            List[np.ndarray]: The new list of stable states
        """
            
        # self.stableStates = []
        for _ in range(self.numPseudorehearsalSamples):
            stablePattern = self.network.getStablePattern()
            if stablePattern is None:
                continue
            self.stableStates.append(stablePattern.copy())

        return self.stableStates


    def finishTask(self, taskPatterns:List[np.ndarray]):
        """
        Finish a task and do any post-processing, to be called after all epochs are run

        Args:
            taskPatterns (List[np.ndarray]): The task patterns from this task
        """

        # Put new states into our array to track them
        self.learnedPatterns.extend(taskPatterns)
        self.numStatesLearned+=len(taskPatterns)

        self.updateStableStates()

        if self.updateRehearsalStatesFreq=="Task":
            self.updateRehearsalPatterns()