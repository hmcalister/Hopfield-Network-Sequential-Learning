from .AbstractPseudorehearsalLearningRule import AbstractPseudorehearsalLearningRule
from typing import List
import numpy as np


class PseudorehearsalDelta(AbstractPseudorehearsalLearningRule):

    def __init__(self, maxEpochs:int=1, numRehearse:np.int64=0, fracRehearse:np.float64=0, numPseudorehearsalSamples:np.int64=0,
        updateRehearsalStatesFreq:str="Epoch", keepFirstTaskPseudoitems:bool=False, rejectLearnedStatesAsPseudoitems:bool=False,
        requireUniquePseudoitems:bool=True, trainUntilStable:bool=False):
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
            keepFirstTaskPseudoitems (bool, optional): Flag to keep only the first batch of pseudoitems. 
                Useful if we are interesting in protecting first epoch. Defaults to False
            rejectLearnedStatesAsPseudoitems (bool, optional): Flag to reject learned states as pseudoitems. Defaults to False
            requireUniquePseudoitems (bool, optional): Flag to find all unique pseudoitems, rather than allowing some repeated items
                Defaults to True.
            trainUntilStable (bool, optional): Flag to train current pattern until stable.
                Defaults to False.
        """

        super().__init__(
            numRehearse = numRehearse, 
            fracRehearse = fracRehearse, 
            numPseudorehearsalSamples = numPseudorehearsalSamples,
            updateRehearsalStatesFreq = updateRehearsalStatesFreq, 
            keepFirstTaskPseudoitems = keepFirstTaskPseudoitems, 
            rejectLearnedStatesAsPseudoitems = rejectLearnedStatesAsPseudoitems,
            requireUniquePseudoitems = requireUniquePseudoitems
        )

        # Delta rule requires a single update step 
        self.updateSteps = 1
        self.trainUntilStable = trainUntilStable

        # The number of epochs to run for is set by constructor argument
        # Notice this can take a very long time, and usually Delta converges very quickly
        self.maxEpochs = maxEpochs       

    def __str__(self):
        return f"PseudorehearsalDelta"

    def infoString(self):
        pseudoSamplesStr = None
        if self.numRehearse!=0:
            pseudoSamplesStr = f"{self.numRehearse}NumRehearse"
        else: 
            pseudoSamplesStr = f"{self.fracRehearse}FracRehearse"

        rejectStr = ""
        if self.rejectLearnedStatesAsPseudoitems:
            rejectStr = "RejectLearnedStates"

        return f"PseudorehearsalDelta-{pseudoSamplesStr} {self.updateRehearsalStatesFreq}UpdateFreq {self.numPseudorehearsalSamples}numPseudorehearsalSamples {rejectStr}"

    def __call__(self, patterns:List[np.ndarray])->np.ndarray:
        """
        Learn a set of patterns and return the weights

        Args:
            patterns (List[np.ndarray]): A list of patterns to learn

        Returns:
            np.ndarray: The new weights of the network after learning
        """

        weightChanges = np.zeros_like(self.network.weights)
        for pattern in patterns:
            resultState = self.findRelaxedState(pattern.copy())
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)

        if self.updateRehearsalStatesFreq=="Epoch":
            self.updateRehearsalPatterns()

        heteroassociativeNoiseRatioCopy = self.heteroassociativeNoiseRatio
        self.heteroassociativeNoiseRatio = 0
        for pattern in self.rehearsalPatterns:
            resultState = self.findRelaxedState(pattern.copy())
            weightChanges = weightChanges+0.5*np.outer(pattern-resultState, pattern)
        self.heteroassociativeNoiseRatio = heteroassociativeNoiseRatioCopy

        return self.network.weights + weightChanges

    