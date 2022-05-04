from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from .EnergyFunction.AbstractEnergyFunction import AbstractEnergyFunction
from .UpdateRule.AbstractUpdateRule import AbstractUpdateRule
from .UpdateRule.ActivationFunction.AbstractActivationFunction import AbstractActivationFunction
from .LearningRule.AbstractLearningRule import AbstractLearningRule

import numpy as np

class RelaxationException(Exception):
    pass

class AbstractHopfieldNetwork(ABC):

    @abstractmethod
    def __init__(self, 
                N:int,
                energyFunction:AbstractEnergyFunction,
                activationFunction:AbstractActivationFunction, 
                updateRule:AbstractUpdateRule,
                learningRule:AbstractLearningRule, 
                weights:np.ndarray=None,
                selfConnections:bool=False):
        """
        Create a new Hopfield network with N units.
        
        If weights is supplied, the network weights are set to the supplied weights.

        Args:
            N (int): The size of the network, how many units to have.
            weights (np.ndarray, optional): The weights of the network, if supplied. Intended to be used to recreate a network for testing.
                If supplied, must be a 2-D matrix of float64 with size N*N. If None, random weights are created uniformly around 0.
                Defaults to None.
            activationFunction (AbstractActivationFunction): The activation function to use with this network. 
                Must implement HopfieldNetwork.ActivationFunction.AbstractActivationFunction
                The given functions in HopfieldNetwork.ActivationFunction do this.
            updateRule (AbstractUpdateRule): The update rule for this network.
                Must implement HopfieldNetwork.UpdateRule.AbstractUpdateRule
                The given methods in HopfieldNetwork.UpdateRule do this.
            learningRule (AbstractLearningRule): The learning rule for this network.
                Must implement HopfieldNetwork.LearningRule.AbstractUpdateRule
                The given methods in HopfieldNetwork.LearningRule do this.
            weights (np.ndarray, optional): The weights of this network. Must be of dimension N*N.
                Used for reproducibility. If None then results in zero matrix of dimension N*N. Defaults to None.
            selfConnections (bool, optional): Determines if self connections are allowed or if they are zeroed out during learning
                Defaults to False. (No self connections)

        Raises:
            ValueError: If the given weight matrix is not N*N and not None.
        """

        self.N:int = N

        self.weights:np.ndarray = None
        if weights is not None:
            if weights.shape!=(N,N): raise ValueError()
            if weights.dtype != np.float64: raise ValueError()
            self.weights = weights.copy()
        else:
            self.weights = np.zeros([N,N])
        
        self.energyFunction = energyFunction
        self.activationFunction = activationFunction
        self.updateRule = updateRule
        self.learningRule = learningRule
        self.selfConnections = selfConnections

        self.state:np.ndarray = np.ones(N)

    @abstractmethod
    def __str__(self):
        return (f"\tEnergy Function: {self.energyFunction}\n"
        + f"\tActivation Function: {self.activationFunction}\n"
        + f"\tUpdate Rule: {self.updateRule}\n"
        + f"\tLearning Rule: {self.learningRule}\n")

    def getWeights(self)->np.ndarray:
        """
        Get the weights of this network

        Returns:
            np.ndarray: the 2-D matrix of weights of this network, of type float 32
        """

        return self.weights.copy()

    def getState(self)->np.ndarray:
        """
        Get the state of this network

        Returns:
            np.ndarray: The float64 vector of size N representing the state of this network
        """

        return self.state.copy()

    @abstractmethod
    def setState(self, state:np.ndarray):
        """
        Set the state of this network.

        Given state must be a float64 vector of size N. Any other type will raise a ValueError.

        This method is abstract so implementing classes must explicitly implement it, 
        meaning any extra checking is explicitly added or eschewed before a call to super.

        This method creates a copy of the given state to ensure no weird side effects occur.

        Args:
            state (np.ndarray): The state to set this network to.

        Raises:
            ValueError: If the given state is not a float64 vector of size N.
        """

        if state.shape != (self.N,): raise ValueError()
        if state.dtype != np.float64: raise ValueError()
        self.state = state.copy()

    def relax(self, maxSteps:int=None)->np.ndarray:
        """
        From the current state of the network, update until a stable state is found. Return this stable state.

        Args:
            maxSteps (int, optional): The maximum number of steps to update for until stopping. If None then set to the max steps of the update rule.
                Defaults to None

        Returns:
            np.ndarray: The final state of the network after relaxing

        Raises:
            RelaxationException: If the maximum number of steps is reached during relaxation
        """

        if maxSteps==None:
            maxSteps=self.updateRule.MAX_STEPS

        current_step = 0
        # TODO: Better way to calculate this? Stability of Network...
        while np.any(self.unitEnergies()<=0):
            if current_step > maxSteps:
                raise RelaxationException()
            self.state = self._updateStep()
            current_step+=1
        
        # We must have reached a stable state, so we can return this state
        return self.state

    def _updateStep(self)->np.ndarray:
        """
        Perform a single update step from the current state to the next state.
        This method returns the next network state, so to affect the change 
        ensure you set the network state to the output of this function.
        This method depends on the update rule selected and activation function.

        Returns:
            np.ndarray: The result of a single update step from the current state
        """

        return self.updateRule(self.state, self.weights)

    def networkEnergy(self)->np.float64:
        """
        Get the total energy of the network. If the energy is greater than 0, the network is stable.
        TODO: Get this working. Always the same equation?? Greater than 0 stable????

        Returns:
            float64: Dimension 1, a single value of float64. The total energy of this network in the current state.
        """

        return -0.5*np.sum(self.energyFunction(self.state, self.weights))

    def unitEnergies(self)->np.ndarray:
        """
        Get the energy of all units. If the energy is less than 0, the unit is stable.

        Returns:
            np.ndarray: A float64 vector of all unit energies. index i is the energy of unit i.
        """

        return self.energyFunction(self.state, self.weights)

    def compareState(self, state:np.ndarray)->bool:
        """
        Compares the given state to the state of the network right now
        Returns True if the two states are the same, false otherwise
        Accepts the inverse state as equal

        Args:
            state (np.ndarray): The state to compare to

        Returns:
            bool: True if the given state is the same as the network state, false otherwise
        """

        return np.array_equal(self.getState(), state) or np.array_equal(-1*self.getState(), state)


    def measureTaskPatternStability(self, taskPatterns:List[List[np.ndarray]])->Tuple[List[np.float64], int]:
        """
        Measure the fraction of task patterns that are stable/remembered with the current network weights.
        The taskPatterns param is a list of lists. Ew. 
        The first index is the task number. taskPatterns[0] is a list of all patterns in the first task.
        The second index is a specific pattern from a task

        TODO: Fix this method up to be more sensible

        Args:
            taskPatterns (List[List[np.ndarray]]): A nested list of all task patterns.

        Returns:
            Tuple[List[np.float64], int]: A list of floats measuring the fraction of stable task patterns learned
                The returned list at index 0 is the accuracy of the network on the first task
                Also, the number of actual stable patterns
        """

        taskAccuracies = []
        numStable = 0

        for task in taskPatterns:
            taskAccuracy = 0
            for pattern in task:
                self.setState(pattern)
                try:
                    self.relax(self.learningRule.updateSteps)
                except RelaxationException as e:
                    pass

                if self.compareState(pattern):
                    taskAccuracy+=1
                    numStable+=1
            taskAccuracy/=len(task)
            taskAccuracies.append(taskAccuracy)

        return taskAccuracies, numStable

    def measureTestAccuracy(self, testPatternMappings:List[Tuple[np.ndarray, np.ndarray]])->np.float64:
        """
        Given a mapping from input to expected output patterns, calculate the accuracy of the network on those mappings
        TODO: Maybe look at more refined accuracy measurements? Hamming dist?
        TODO: Fix this method up to be more sensible

        Args:
            testPatternsDict (List[Tuple[[np.ndarray, np.ndarray]]): The test patterns to measure the accuracy on.
                A list of tuples like (input, expectedOutput)

        Returns:
            np.float64: The accuracy of this network on relaxing from the given inputs to expected outputs.
                1 if all relaxations are as expected
                0 if no relaxations were expected
        """

        accuracy = 0

        for i, (input,expectedOutput) in enumerate(testPatternMappings):
            self.setState(input)
            try:
                self.relax()
            except RelaxationException as e:
                continue
            if self.compareState(expectedOutput):
                accuracy+=1

        return accuracy/len(testPatternMappings)

    @abstractmethod
    def learnPatterns(self, patterns:List[np.ndarray], allTaskPatterns:List[List[np.ndarray]]=None)->Union[None, Tuple[List[List[np.float64]], List[int]]]:
        """
        Learn a set of patterns given. This method will use the learning rule given at construction to learn the patterns.
        The patterns are given as a list of np.ndarrays which must each be a vector of size N.
        Any extra checking of these patterns must be done by an implementing network, hence the abstractmethod decorator.

        Args:
            patterns (List[np.ndarray]): The patterns to learn. Each np.ndarray must be a float64 vector of length N (to match the state)
            allTaskPatterns (List[List[np.ndarray]] or None, optional): If given, will track the task pattern stability by epoch during training.
                Passed straight to measureTaskPatternAccuracy. Defaults to None.

        Returns: None or List[Tuple[List[np.float64], int]]]
            If allTaskPatterns is None, returns None
            If allTaskPatterns is present, returns a list over epochs of tuples. Tuples are of form (list of task accuracies, num stable learned patterns overall)
        """

        taskAccuracies = []
        numStableByEpoch = []

        # Ensure patterns are correct type and shape
        for pattern in patterns:
            if pattern.shape != (self.N,): raise ValueError()
            if pattern.dtype != np.float64: raise ValueError()
        
        # Loop until we reach the maximum number of epochs in the learning rule
        for epoch in range(self.learningRule.epochs):
            resultStates = []

            if self.learningRule.epochs>=1:
                print(f"Epoch: {epoch+1}/{self.learningRule.epochs}", end="\r")
            

            # If the learning rule needs the current network predictions (e.g. delta)
            if self.learningRule.updateSteps>0:
                # Calculate that prediction for each pattern and store it
                for pattern in patterns:
                    self.setState(pattern)
                    try:
                        self.relax(self.learningRule.updateSteps)
                        resultStates.append(self.getState())
                    except RelaxationException as e:
                        resultStates.append(self.getState())

            # Set the weights to the output of the learning rule
            self.weights = self.learningRule(patterns, resultStates, self.weights).copy()

            if allTaskPatterns is not None:
                acc,numStable=self.measureTaskPatternStability(allTaskPatterns)
                taskAccuracies.append(acc)
                numStableByEpoch.append(numStable)

            # If we are removing self connections, do that now
            if not self.selfConnections:
                np.fill_diagonal(self.weights, 0)
        # self.learningRule.numStatesLearned+=len(patterns)
        print()

        if allTaskPatterns is not None:
            return taskAccuracies, numStableByEpoch