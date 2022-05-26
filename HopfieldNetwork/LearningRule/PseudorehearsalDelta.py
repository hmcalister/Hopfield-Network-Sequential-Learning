from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class PseudorehearsalDelta(AbstractLearningRule):

    def __init__(self, maxEpochs:int=1, numRehearse:np.int64=0, fracRehearse:np.float64=0, numPseudorehearsalSamples:np.int64=0,
        updateRehearsalStatesFreq:str="Epoch", keepPreviousStableStates:bool = True, keepPreviousWeights:bool=True, trainUntilStable:bool=False):
        """
        Create a new PseudorehearsalDelta Learning Rule

        Notice that fracRehearse=1, keepPreviousWeights=False is a "best case scenario" for sequential learning

        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 1
            numRehearse (np.int64, optional): The number of tasks to rehearse. Absolute, not relative.
                Defaults to 0. Still randomly selected. If also present with fracRehearse, fracRehearse takes priority
            fracRehears (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 0. (all patterns)
            updateRehearsalStatesFreq (str, optional): When to update the rehearsal states. 'Epoch' chooses new states every call/epoch
                'Task' chooses new states every task
            keepPreviousStableStates (bool, optional): Flag to keep previous stable states when updating stable states.
                Defaults to True. States are kept, new states are added
            keepPreviousWeights (bool, optional): Flag to keep the previous weights of the network, or throw these away
                Defaults to True.
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False.
        """

        # Delta rule requires a single update step 
        self.updateSteps = 1
        self.trainUntilStable = trainUntilStable

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs

        # Delta does actually use numStatesLearned
        self.numStatesLearned = 0

        # Rehearsal requires knowledge of what we have learned so far
        self.learnedPatterns = []
        self.stableStates = []
        self.rehearsalPatterns = []
        self.numRehearse = numRehearse
        self.fracRehearse = fracRehearse
        self.numPseudorehearsalSamples = numPseudorehearsalSamples
        self.updateRehearsalStatesFreq = updateRehearsalStatesFreq
        self.keepPreviousStableStates = keepPreviousStableStates

        self.keepPreviousWeights = keepPreviousWeights

    def __str__(self):
        if self.numRehearse!=0:
            return f"PseudorehearsalDelta-{self.numRehearse}NumRehearse {self.updateRehearsalStatesFreq}UpdateFreq {self.numPseudorehearsalSamples}numPseudorehearsalSamples"
        return f"PseudorehearsalDelta-{self.fracRehearse}FracRehearse {self.updateRehearsalStatesFreq}UpdateFreq {self.numPseudorehearsalSamples}numPseudorehearsalSamples"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        weightChanges = np.zeros_like(self.network.weights)
        for i in range(len(patterns)):
            # Get pattern i and the resultState i
            pattern = patterns[i].copy()
            resultState = self.findRelaxedState(pattern)

            # The weight changes of this pattern is the outer product of
            # The difference of pattern and result state and this pattern
            # Scale factor is 0.5 so updates are -1 and 1 (one from both sides)
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)

        if self.updateRehearsalStatesFreq=="Epoch":
            self.updateRehearsalPatterns()
        for pattern in self.rehearsalPatterns:
            resultState = self.findRelaxedState(pattern)
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)

        return self.network.weights + weightChanges

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
        return []

    def updateStableStates(self)->List[np.ndarray]:
        """
        Update the list of stable states, appending to any previous stable states

        Returns:
            List[np.ndarray]: The new list of stable states
        """

        # if len(self.stableStates)>0:
        #     return

        attemptCounter = 0
        # Give ourselves some space to find the stable states requested
        maxAttempts = 1 * self.numPseudorehearsalSamples
        newStableStates = []
        while len(newStableStates) < (self.numPseudorehearsalSamples) and attemptCounter < maxAttempts:
            attemptCounter += 1
            stablePattern = self.network.getStablePattern()
            if stablePattern is None:
                continue
            newStableStates.append(stablePattern.copy())
        print(f"NewStableStates: {len(newStableStates)}")
        
        if self.keepPreviousStableStates:
            # Keep the old stable states and just append
            self.stableStates.extend(newStableStates)
        else:
            # Overwrite the old stable states
            self.stableStates = newStableStates.copy()
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