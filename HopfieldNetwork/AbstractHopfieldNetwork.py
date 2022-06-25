from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from scipy.special import erf, erfinv

from .EnergyFunction.AbstractEnergyFunction import AbstractEnergyFunction
from .UpdateRule.AbstractUpdateRule import AbstractUpdateRule
from .UpdateRule.ActivationFunction.AbstractActivationFunction import AbstractActivationFunction
from .LearningRule.AbstractLearningRule import AbstractLearningRule

import numpy as np


class RelaxationException(Exception):
    """
    RelaxationException is raised when network relaxing does not reach
    a stable state within a certain number of update steps.
    This essentially is just a notification that the network failed to stabilize
    But still returning the state
    """
    pass


class AbstractHopfieldNetwork(ABC):

    # CONSTRUCTOR -----------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(self,
                 N: int,
                 energyFunction: AbstractEnergyFunction,
                 activationFunction: AbstractActivationFunction,
                 updateRule: AbstractUpdateRule,
                 learningRule: AbstractLearningRule,
                 allowableLearningStateError: np.float64 = 0,
                 patternManager=None,
                 weights: np.ndarray = None,
                 selfConnections: bool = False):
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
            allowableStabilityError (np.float64, optional): The allowable error (as a ratio of all units) for a pattern to be stable
                Implemented as a check of Hamming distance against the intended pattern during learning.
                If 0 (default), the pattern must be exactly the same.
            patternManager : The pattern manager that creates the patterns for this network. Optional, Defaults to None
            weights (np.ndarray, optional): The weights of this network. Must be of dimension N*N.
                Used for reproducibility. If None then results in zero matrix of dimension N*N. Defaults to None.
            selfConnections (bool, optional): Determines if self connections are allowed or if they are zeroed out during learning
                Defaults to False. (No self connections)

        Raises:
            ValueError: If the given weight matrix is not N*N and not None.
        """

        self.N: int = N
        self.networkName: str = "AbstractHopfieldNetwork"

        self.weights: np.ndarray = None
        if weights is not None:
            if weights.shape != (N, N):
                raise ValueError()
            if weights.dtype != np.float64:
                raise ValueError()
            self.weights = weights.copy()
        else:
            self.weights = np.zeros([N, N])

        self.energyFunction = energyFunction
        self.activationFunction = activationFunction
        self.updateRule = updateRule
        self.learningRule = learningRule
        self.selfConnections = selfConnections
        self.allowableLearningStateError = allowableLearningStateError
        self.patternManager = patternManager

        # The total number of epochs we have trained for
        self.epochs = 0

        self.state: np.ndarray = np.ones(N)

    # CLASS METHODS -----------------------------------------------------------------------------------------------------
    @classmethod
    def hammingDistance(cls, state1: np.ndarray, state2: np.ndarray):
        """
        Find the Hamming distance between two states.
        For discrete states this is simply a count of different units.
        TODO Find a measure for continuous states.

        Args:
            state1 (np.ndarray): The first state
            state2 (np.ndarray): The second state

        Return:
            np.int32: the sum of the different units between the two states
        """

        return np.sum(state1 != state2)

    # DESCRIPTION METHODS -----------------------------------------------------------------------------------------------------

    @abstractmethod
    def __str__(self):
        """
        This to string method is intended to be called by implementing networks to avoid repetition.
        """

        return f"Hopfield Network: {self.networkName}"

    def getNetworkDescriptionString(self):
        return (str(self)+"\n"
                + f"Units: {self.N}\n"
                + f"Energy Function: {self.energyFunction}\n"
                + f"Activation Function: {self.activationFunction}\n"
                + f"Update Rule: {self.updateRule}\n"
                + f"Learning Rule: {self.learningRule}\n"
                + f"Allowable Learning State Error: {self.allowableLearningStateError}"
                )

    def getNetworkDescriptionJSON(self):
        return {
            "Hopfield Network": self.networkName,
            "Units": self.N,
            "Energy Function": str(self.energyFunction),
            "Activation Function": str(self.activationFunction),
            "Update Rule": str(self.updateRule),
            "Learning Rule": str(self.learningRule),
            "Allowable Learning State Error": self.allowableLearningStateError
        }

    def getHebbianMaxRatio(self) -> np.float64:
        """
        Generates and returns the Hebbian maximum capacity ratio p_max/N for this network.
        This is calculated from Hertz, J. Introduction to the Theory of Neural Computation, pg. 18-19
        Based on self.allowableLearningStateError, if this is 0 then this method returns None.

        Returns:
            np.float64: The theoretical maximum capacity of this network given the allowable state error
                Which is effectively a measure of how many units can be unstable in this network while still being considered stable.
        """

        if self.allowableLearningStateError == 0:
            return None
        return 0.5*np.power(erfinv(1 - 2*self.allowableLearningStateError), -2)

    # GETTERS AND SETTERS -----------------------------------------------------------------------------------------------------

    def getWeights(self) -> np.ndarray:
        """
        Get the weights of this network
        Notice a copy is returned to avoid memory issues

        Returns:
            np.ndarray: the 2-D matrix of weights of this network, of type float 32
        """

        return self.weights.copy()

    def getState(self) -> np.ndarray:
        """
        Get the state of this network
        Notice a copy is returned to avoid memory issues

        Returns:
            np.ndarray: The float64 vector of size N representing the state of this network
        """

        return self.state.copy()

    @abstractmethod
    def setState(self, state: np.ndarray):
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

        if state.shape != (self.N,):
            raise ValueError()
        if state.dtype != np.float64:
            raise ValueError()
        self.state = state.copy()

    # DYNAMICS AND RELAXATION METHODS -----------------------------------------------------------------------------------------------------

    def _updateStep(self) -> np.ndarray:
        """
        Perform a single update step from the current state to the next state.
        This method returns the next network state, so to affect the change 
        ensure you set the network state to the output of this function.
        This method depends on the update rule selected and activation function.

        Returns:
            np.ndarray: The result of a single update step from the current state
        """

        return self.updateRule(self.state, self.weights)

    def relax(self, maxSteps: int = None) -> np.ndarray:
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

        # Find the max steps, either from the arg or from the update rule
        if maxSteps is None:
            maxSteps = self.updateRule.MAX_STEPS

        # Set the current step to 0 to count up until max steps
        currentStep = 0
        # TODO: Better way to calculate this? Stability of Network...
        # Stability of network determined by finding if any units have positive energy
        while not self.isStable():
            # If we are above the max steps allowed, we raise an error
            # The network failed to converge
            if currentStep >= maxSteps:
                raise RelaxationException()
            # Step forward one step
            self.state = self._updateStep()
            currentStep += 1

        # We must have reached a stable state, so we can return this state
        return self.state.copy()

    def getInverseState(self, state:np.ndarray) -> np.ndarray:
        return self.activationFunction(-1*state.copy())

    def compareState(self, state1: np.ndarray, state2:np.ndarray=None, allowableHammingDistanceRatio: np.float64 = 0) -> bool:
        """
        Compares the given state to the state of the network right now
        Returns True if the two states are the same, false otherwise
        Accepts the inverse state as equal

        Args:
            state (np.ndarray): The state to compare to
            allowableHammingDistance (np.float64): The allowable units to be incorrect (as a ratio of all units)
                If 0 (default) then a straight comparison is used

        Returns:
            bool: True if the given state is the same as the network state, false otherwise
        """

        if state2 is None:
            state2 = self.getState()

        invertState = self.getInverseState(state1)

        if allowableHammingDistanceRatio == 0:
            # Generally, a state is equal to itself or the negate of itself.
            # Negate in binary (or with state space centered around 0) has negation as *-1
            return np.array_equal(state1, state2) or np.array_equal(invertState, state2)
        else:
            hammingDistance = min(
                self.hammingDistance(state1, state2),
                self.hammingDistance(invertState, state2)
            )

            return hammingDistance <= self.N*allowableHammingDistanceRatio

    # STATE STABILITY METHODS -----------------------------------------------------------------------------------------------------

    def invertStateUnits(self, state: np.ndarray, inverseRatio: np.float64) -> np.ndarray:
        """
        Invert a number of units (determined by inverse ratio) in the given state and return it
        Inverse is done by multiplying the units by -1 and passing through the activation

        Args:
            state (np.ndarray): The state to flip units on
            inverseRatio (np.float64): The number of units to flip, expressed as a fraction of all units

        Returns:
            np.ndarray: A copy of the original state with a number of units flipped
        """

        maxFlip = int(np.ceil(inverseRatio*state.shape[0]))
        numFlip = np.random.randint(0, maxFlip+1)
        flipIndices = np.arange(state.shape[0])
        np.random.shuffle(flipIndices)
        flipIndices = flipIndices[:numFlip]
        flipMask = np.zeros(state.shape[0], dtype=bool)
        flipMask[flipIndices]=True

        newState = state.copy()
        invertState = self.activationFunction(-1*state.copy())
        np.copyto(newState, invertState, where=flipMask)

        # print(f"ORG: {self.state}")
        # print(f"INV: {invertState}")
        # print(f"NEW: {newState}")
        # print(f"O&N: {1*(self.state == newState)}")
        # print(f"INDEX: {flipIndices}")
        # print()

        # newState = self.activationFunction(newState)


        return newState

    def networkEnergy(self) -> np.float64:
        """
        Get the total energy of the network. If the energy is greater than 0, the network is stable.
        TODO: Get this working. Always the same equation?? Greater than 0 stable????

        Returns:
            float64: a single value of float64. The total energy of this network in the current state.
        """

        return -0.5*np.sum(self.energyFunction(self.state, self.weights))

    def unitEnergies(self) -> np.ndarray:
        """
        Get the energy of all units. If the energy is less than 0, the unit is stable.

        Returns:
            np.ndarray: A float64 vector of all unit energies. index i is the energy of unit i.
        """

        return self.energyFunction(self.state, self.weights)

    def isStable(self) -> bool:
        """
        Checks if the current state of the network is stable

        Returns:
            bool: True if all units in this network are stable. E<0
        """

        return np.all(self.unitEnergies() < 0)

    def checkAllPatternStability(self, patterns: List[np.ndarray]) -> bool:
        """
        Check if all of the given patterns are stable

        Args:
            patterns (List[np.ndarray]): A list of patterns to check the stability of

        Returns:
            bool: True if all patterns are stable
        """

        # We see if all of our current patterns are stable, then carry on
        for pattern in patterns:
            # Set the network to the current pattern
            self.setState(pattern)
            try:
                self.relax(1)
            except RelaxationException as e:
                # More than likely we will encounter a relaxation error, this is fine
                pass

            # If the relaxed state is the same as the pattern, we have a stable state
            if not self.compareState(pattern, allowableHammingDistanceRatio=self.allowableLearningStateError):
                # If we are not stable on even a single pattern, we must carry on
                return False
        return True

    def measureTaskPatternStability(self, taskPatterns: List[List[np.ndarray]]) -> Tuple[List[np.float64], int]:
        """
        Measure the fraction of task patterns that are stable/remembered with the current network weights.
        The taskPatterns param is a list of lists. Ew. 
        The first index is the task number. taskPatterns[0] is a list of all patterns in the first task.
        The second index is a specific pattern from a task.
        This method tests each pattern in each task. For each pattern, it checks if that pattern is stable
        i.e. set the network equal to that state, relax the network, then check the state is the same.

        This method returns two values. The first is a list of stable states by task(expressed as a fraction of total states for that task).
        The second is a raw count of total stable states.

        TODO: Fix this method up to be more sensible?

        Args:
            taskPatterns (List[List[np.ndarray]]): A nested list of all task patterns.

        Returns:
            Tuple[List[np.float64], int]: A list of floats measuring the fraction of stable task patterns learned
                The returned list at index 0 is the accuracy of the network on the first task
                Also, the number of actual stable patterns
        """

        # Track both stabilities and total stable states
        taskStabilities = []
        numStable = 0

        # First index of taskPatterns indexes tasks
        for task in taskPatterns:
            if len(task) == 0:
                taskStabilities.append(0)
                continue
            # This task starts with an accuracy of 0
            taskAccuracy = 0
            # The next index is over patterns within a task
            for pattern in task:
                # Set the network to the current pattern
                self.setState(pattern)
                # The following section might be better replaced by a simple energy check??
                # Would this work for continuos state space...

                # Relax the network until stable
                # We can tell if the network is stable after a single step
                try:
                    self.relax(1)
                except RelaxationException as e:
                    # More than likely we will encounter a relaxation error, this is fine
                    pass

                # If the relaxed state is the same as the pattern, we have a stable state
                if self.compareState(pattern, allowableHammingDistanceRatio=self.allowableLearningStateError):
                    taskAccuracy += 1
                    numStable += 1
            # Task accuracy is scaled to a fraction of total of this tasks patterns
            taskAccuracy /= len(task)
            taskStabilities.append(taskAccuracy)

        return taskStabilities, numStable

    def measureTestAccuracy(self, testPatternMappings: List[Tuple[np.ndarray, np.ndarray]]) -> np.float64:
        """
        Given a mapping from input to expected output patterns, calculate the accuracy of the network on those mappings
        TODO: Maybe look at more refined accuracy measurements? Hamming dist? Dice Coefficient?
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

        for i, (input, expectedOutput) in enumerate(testPatternMappings):
            self.setState(input)
            try:
                self.relax()
            except RelaxationException as e:
                continue
            if self.compareState(expectedOutput):
                accuracy += 1

        return accuracy/len(testPatternMappings)

    # LEARNING METHODS ------------------------------------------------------------------------------------------------------------

    def generatePattern(self) -> np.ndarray:
        """
        Get a pattern from the pattern manager. If no pattern manager is set, raise an error

        Returns:
            np.ndarray: A pattern from the pattern manager
        """
        if self.patternManager is None:
            raise ValueError

        return self.patternManager.generatePattern()

    def getStablePattern(self, pattern:np.ndarray=None) -> Union[np.ndarray, None]:
        """
        Generate a random pattern from the pattern manager, and relax this until we find a stable state
        Return the stable state. Note that we do not require or even check if this state is a learned state

        Returns:
            np.ndarray: A stable state of the network
            None: If the relaxed state was not stable
        """

        if pattern is None:
            pattern = self.generatePattern()
        self.setState(pattern)
        try:
            pattern = self.relax(100)
        except RelaxationException:
            pass
        if self.isStable():
            return self.state.copy()
        else:
            return None

    @abstractmethod
    def learnPatterns(self, patterns: List[np.ndarray], allTaskPatterns: List[List[np.ndarray]] = None,
                      heteroassociativeNoiseRatio: np.float64 = 0, inputNoise: str = None) -> Union[None, Tuple[List[List[np.float64]], List[int]]]:
        """
        Learn a set of patterns given. This method will use the learning rule given at construction to learn the patterns.
        The patterns are given as a list of np.ndarrays which must each be a vector of size N.
        Any extra checking of these patterns must be done by an implementing network, hence the abstractmethod decorator.

        If allTaskPatterns is given, network is tested for task pattern stability over epochs.
        Given list must be passed directly to measureTaskPatternStability so see this method for details.
        Returned type is ugly. First element is a list of lists, showing task pattern stability over epochs. 
            The first list indexes over epochs, the second list indexes over tasks.
        Second element is a list of ints, indicating a total stable patterns over epochs.

        Args:
            patterns (List[np.ndarray]): The patterns to learn. Each np.ndarray must be a float64 vector of length N (to match the state)
            allTaskPatterns (List[List[np.ndarray]] or None, optional): If given, will track the task pattern stability by epoch during training.
                Passed straight to measureTaskPatternAccuracy. Defaults to None.
            heteroassociativeNoiseRatio (np.float64, optional): The fraction of units to add a noise term to before calculating error.
                Must be between 0 and 1. Defaults to 0.
            inputNoise (str or None, optional): String on whether to apply input noise to the units before activation
                - "Absolute": Apply absolute noise to the state, a Gaussian of mean 0 std 1
                - "Relative": Apply relative noise to the state, a Gaussian of mean and std determined by the state vector
                - None: No noise. Default

        Returns: None or Tuple[ List[List[np.float64]], List[int] ]
            If allTaskPatterns is None, returns None
            If allTaskPatterns is present, returns a Tuple 
                The first element is a list of taskAccuracies over epochs (2-D array)
                The second element is a list of numStablePatterns by epoch
        """
        # Ensure patterns are correct type and shape
        for pattern in patterns:
            if pattern.shape != (self.N,):
                raise ValueError()
            if pattern.dtype != np.float64:
                raise ValueError()

        # If allTaskPatterns given we need to track these...
        taskAccuracies = []
        numStableByEpoch = []

        self.learningRule.setNetworkReference(self)

        currentTaskEpochs = 0
        # Loop until we reach the maximum number of epochs in the learning rule
        while currentTaskEpochs < (self.learningRule.maxEpochs):
            currentTaskEpochs += 1
            self.epochs += 1

            # Set the learning and update rule noise information
            self.learningRule.setHeteroassociativeNoiseRatio(heteroassociativeNoiseRatio)
            self.updateRule.setInputNoiseType(inputNoise)

            # Set the weights to the output of the learning rule
            self.weights = self.learningRule(patterns).copy()

            # Clear the noise information
            self.learningRule.clearHeteroassociativeNoiseRatio()
            self.updateRule.clearInputNoiseType()

            # Print some information for debugging
            if self.learningRule.maxEpochs > 1:
                if len(taskAccuracies) > 0 and len(allTaskPatterns) < 20:
                # Notice we add a large number of spaces on the end, to nicely overwrite any older lines
                    print(f"Epoch: {currentTaskEpochs}/{self.learningRule.maxEpochs} : {taskAccuracies[-1]}"+" "*80, end="\r")
                else:
                    print(f"Epoch: {currentTaskEpochs}/{self.learningRule.maxEpochs}"+" "*80, end="\r")

            # If we are removing self connections, do that now
            if not self.selfConnections:
                np.fill_diagonal(self.weights, 0)

            if allTaskPatterns is not None:
                acc, numStable = self.measureTaskPatternStability(allTaskPatterns)
                taskAccuracies.append(acc)
                numStableByEpoch.append(numStable)

            # If we are only training until stable
            if self.learningRule.trainUntilStable:
                # If we HAVE learned all the patterns, we can stop training!
                if self.checkAllPatternStability(patterns):
                    break

        # END While taskEpochs<self.learningRule.maxEpochs
        print()
        self.learningRule.finishTask(patterns)


        if allTaskPatterns is not None:
            return taskAccuracies, numStableByEpoch
