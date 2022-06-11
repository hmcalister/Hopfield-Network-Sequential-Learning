from .AbstractLearningRule import AbstractLearningRule
from typing import List
import numpy as np


class AbstractPseudorehearsalLearningRule(AbstractLearningRule):
    
    def __init__(self, numRehearse:np.int64=0, fracRehearse:np.float64=0, numPseudorehearsalSamples:np.int64=0,
        updateRehearsalStatesFreq:str="Epoch", keepFirstTaskPseudoitems:bool=False, rejectLearnedStatesAsPseudoitems:bool=False,
        requireUniquePseudoitems:bool=True):
        """
        Args:
            maxEpochs (int, optional): The epochs to train. Defaults to 1
            numRehearse (np.int64, optional): The number of tasks to rehearse. Absolute, not relative.
                Defaults to 0. Still randomly selected. If also present with fracRehearse, fracRehearse takes priority
            fracRehears (np.float64, optional): The fraction of previous patterns to rehearse with each new call
                Defaults to 0. (all patterns)
            updateRehearsalStatesFreq (str, optional): When to update the rehearsal states. 'Epoch' chooses new states every call/epoch
                'Task' chooses new states every task
            keepFirstTaskPseudoitems (bool, optional): Flag to keep only the first batch of pseudoitems. 
                Useful if we are interesting in protecting first epoch. Defaults to False
            rejectLearnedStatesAsPseudoitems (bool, optional): Flag to reject learned states as pseudoitems. Defaults to False
            requireUniquePseudoitems (bool, optional): Flag to find all unique pseudoitems, rather than allowing some repeated items
                Defaults to True.
        """

        self.numStatesLearned = 0
        self.learnedPatterns = []
        self.stableStates = []
        self.rehearsalPatterns = []
        self.numRehearse = numRehearse
        self.fracRehearse = fracRehearse
        self.numPseudorehearsalSamples = numPseudorehearsalSamples
        self.updateRehearsalStatesFreq = updateRehearsalStatesFreq
        self.keepFirstTaskPseudoitems = keepFirstTaskPseudoitems
        self.rejectLearnedStatesAsPseudoitems = rejectLearnedStatesAsPseudoitems
        self.requireUniquePseudoitems = requireUniquePseudoitems

        self.maxStableStatesAttemptsFactor = 20

    def isPatternInSet(self, pattern:np.ndarray, set:np.ndarray):
        inversePattern = self.network.invertStateUnits(pattern.copy(), 1)
        return np.any(np.all(pattern == set,axis=1)) or np.any(np.all(inversePattern == set,axis=1))

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

        if self.keepFirstTaskPseudoitems and len(self.stableStates)>0:
            return self.stableStates

        # Give ourselves some space to find the stable states requested
        maxAttempts = self.maxStableStatesAttemptsFactor * self.numPseudorehearsalSamples
        newStableStates = np.empty((0,self.network.N))
        
        for previouslyStableState in self.stableStates:
            self.network.setState(previouslyStableState)
            stillStable = self.network.isStable()
            if stillStable:
                newStableStates = np.vstack([newStableStates, previouslyStableState.copy()])
            print(f"NewStableStates: {len(newStableStates)}/{self.numPseudorehearsalSamples}, Previous Stable State: {stillStable=}"+" "*80, end="\r")
            
        print()
        attemptCounter = 0

        while len(newStableStates) < (self.numPseudorehearsalSamples) and attemptCounter < maxAttempts:

            # if attemptCounter==self.maxStableStatesAttemptsFactor and len(newStableStates)==0: return []

            attemptCounter += 1
            stablePattern = self.network.getStablePattern()

            print(f"NewStableStates: {len(newStableStates)}/{self.numPseudorehearsalSamples}, Attempt: {attemptCounter}/{maxAttempts}, Current State Stable: {self.network.isStable()}"+" "*80, end="\r")
            
            if stablePattern is None:
                continue
            if self.rejectLearnedStatesAsPseudoitems and self.isPatternInSet(stablePattern, self.learnedPatterns):
                continue
            if self.requireUniquePseudoitems and self.isPatternInSet(stablePattern, newStableStates):
                continue

            newStableStates = np.vstack([newStableStates, stablePattern.copy()])
        
        print()
        uniqueStates = np.empty((0,self.network.N))
        for pattern in newStableStates:
            if not self.isPatternInSet(pattern, uniqueStates):
                # print(pattern)
                uniqueStates = np.vstack([uniqueStates, pattern.copy()])
        
        totalLearnedStatesFound = 0
        for pattern in uniqueStates:
            if self.isPatternInSet(pattern, self.learnedPatterns):
                totalLearnedStatesFound+=1

        print(f"Learned States Found: {totalLearnedStatesFound}")
        print(f"Total Unique Stable States: {len(uniqueStates)}")

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
        print()
        if self.updateRehearsalStatesFreq=="Task":
            self.updateRehearsalPatterns()