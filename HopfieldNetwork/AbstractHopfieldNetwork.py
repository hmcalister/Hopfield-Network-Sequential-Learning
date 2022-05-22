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

    @abstractmethod
    def __init__(self, 
                N:int,
                energyFunction:AbstractEnergyFunction,
                activationFunction:AbstractActivationFunction, 
                updateRule:AbstractUpdateRule,
                learningRule:AbstractLearningRule,
                allowableLearningStateError:np.float64=0,
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
            allowableStabilityError (np.float64, optional): The allowable error (as a ratio of all units) for a pattern to be stable
                Implemented as a check of Hamming distance against the intended pattern during learning.
                If 0 (default), the pattern must be exactly the same.
            weights (np.ndarray, optional): The weights of this network. Must be of dimension N*N.
                Used for reproducibility. If None then results in zero matrix of dimension N*N. Defaults to None.
            selfConnections (bool, optional): Determines if self connections are allowed or if they are zeroed out during learning
                Defaults to False. (No self connections)

        Raises:
            ValueError: If the given weight matrix is not N*N and not None.
        """

        self.N:int = N
        self.networkName:str = "AbstractHopfieldNetwork"

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
        self.allowableLearningStateError = allowableLearningStateError

        # The total number of epochs we have trained for
        self.epochs = 0

        self.state:np.ndarray = np.ones(N)

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
        + f"Allowable Learning State Error: {self.allowableLearningStateError}")

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

    def getWeights(self)->np.ndarray:
        """
        Get the weights of this network
        Notice a copy is returned to avoid memory issues

        Returns:
            np.ndarray: the 2-D matrix of weights of this network, of type float 32
        """

        return self.weights.copy()

    def getState(self)->np.ndarray:
        """
        Get the state of this network
        Notice a copy is returned to avoid memory issues

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

        # Find the max steps, either from the arg or from the update rule
        if maxSteps is None:
            maxSteps=self.updateRule.MAX_STEPS

        # Set the current step to 0 to count up until max steps
        currentStep = 0
        # TODO: Better way to calculate this? Stability of Network...
        # Stability of network determined by finding if any units have positive energy
        while np.any(self.unitEnergies()>=0):
            # If we are above the max steps allowed, we raise an error
            # The network failed to converge
            if currentStep >= maxSteps:
                raise RelaxationException()
            # Step forward one step
            self.state = self._updateStep()
            currentStep+=1
        
        # We must have reached a stable state, so we can return this state
        return self.state.copy()

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
            float64: a single value of float64. The total energy of this network in the current state.
        """

        return -0.5*np.sum(self.energyFunction(self.state, self.weights))

    def unitEnergies(self)->np.ndarray:
        """
        Get the energy of all units. If the energy is less than 0, the unit is stable.

        Returns:
            np.ndarray: A float64 vector of all unit energies. index i is the energy of unit i.
        """

        return self.energyFunction(self.state, self.weights)

    def compareState(self, state:np.ndarray, allowableHammingDistance:np.float64=0)->bool:
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

        if allowableHammingDistance==0:
            # Generally, a state is equal to itself or the negate of itself.
            # Negate in binary (or with state space centered around 0) has negation as *-1
            return np.array_equal(self.getState(), state) or np.array_equal(-1*self.getState(), state)
        else:
            hammingDistance = min(
                self.hammingDistance(self.state, state),
                self.hammingDistance(-1*self.state, state)
            )

            return hammingDistance <= self.N*allowableHammingDistance


    def measureTaskPatternStability(self, taskPatterns:List[List[np.ndarray]])->Tuple[List[np.float64], int]:
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
                if self.compareState(pattern, self.allowableLearningStateError):
                    taskAccuracy+=1
                    numStable+=1
            # Task accuracy is scaled to a fraction of total of this tasks patterns
            taskAccuracy/=len(task)
            taskStabilities.append(taskAccuracy)

        return taskStabilities, numStable

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
    def learnPatterns(self, patterns:List[np.ndarray], allTaskPatterns:List[List[np.ndarray]]=None,
        heteroassociativeNoiseRatio:np.float64=0, inputNoise:str=None)->Union[None, Tuple[List[List[np.float64]], List[int]]]:
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

        # If allTaskPatterns given we need to track these...
        taskAccuracies = []
        numStableByEpoch = []

        # Give the learning rule heteroassociative and input noise information
        self.learningRule.heteroassociativeNoiseRatio = heteroassociativeNoiseRatio
        self.learningRule.setNetworkReference(self)
        
        # Give the update rule the inputNoise information
        self.updateRule.setInputNoiseType(inputNoise)

        # print(f"{heteroassociativeNoiseRatio=}\n{inputNoise=}")

        # Ensure patterns are correct type and shape
        for pattern in patterns:
            if pattern.shape != (self.N,): raise ValueError()
            if pattern.dtype != np.float64: raise ValueError()
        
        currentTaskEpochs = 0
        # Loop until we reach the maximum number of epochs in the learning rule
        while currentTaskEpochs < (self.learningRule.maxEpochs):
            currentTaskEpochs+=1
            self.epochs+=1

            # Print some information for debugging
            if self.learningRule.maxEpochs>1:
                # Notice we add a large number of spaces on the end, to nicely overwrite any older lines
                if len(taskAccuracies)>0 and len(allTaskPatterns)<20:
                    print(f"Epoch: {currentTaskEpochs}/{self.learningRule.maxEpochs} : {taskAccuracies[-1]}"+" "*80, end="\r")
                else:
                    print(f"Epoch: {currentTaskEpochs}/{self.learningRule.maxEpochs}"+" "*80, end="\r")


            # Set the learning and update rule noise information
            self.learningRule.setHeteroassociativeNoiseRatio(heteroassociativeNoiseRatio)
            self.updateRule.setInputNoiseType(inputNoise)

            # Set the weights to the output of the learning rule
            self.weights = self.learningRule(patterns).copy()

            # Clear the noise information
            self.learningRule.clearHeteroassociativeNoiseRatio()
            self.updateRule.clearInputNoiseType()
            

            # If we are removing self connections, do that now
            if not self.selfConnections:
                np.fill_diagonal(self.weights, 0)

            
            if allTaskPatterns is not None:
                acc,numStable=self.measureTaskPatternStability(allTaskPatterns)
                taskAccuracies.append(acc)
                numStableByEpoch.append(numStable)

            # If we are only training until stable
            if self.learningRule.trainUntilStable:
                # We see if all of our current patterns are stable, then carry on
                learnedAllPatterns = True
                for pattern in patterns:
                    # Set the network to the current pattern
                    self.setState(pattern)
                    try:
                        self.relax(1)
                    except RelaxationException as e:
                        # More than likely we will encounter a relaxation error, this is fine
                        pass

                    # If the relaxed state is the same as the pattern, we have a stable state
                    if not self.compareState(pattern, self.allowableLearningStateError):
                        # If we are not stable on even a single pattern, we must carry on
                        learnedAllPatterns = False
                        break
                # If we HAVE learned all the patterns, we can stop training!
                if learnedAllPatterns:
                    break

        print()

        # Give the learning rule heteroassociative and input noise information
        self.learningRule.heteroassociativeNoiseRatio = heteroassociativeNoiseRatio
        # Give the update rule the inputNoise information
        self.updateRule.inputNoise = inputNoise

        if allTaskPatterns is not None:
            return taskAccuracies, numStableByEpoch

    def invertStateUnits(self, state:np.ndarray, inverseRatio:np.float64)->np.ndarray:
        """
        Invert a number of units (determined by inverse ratio) in the given state and return it
        Inverse is done by multiplying the units by -1 

        Args:
            state (np.ndarray): The state to flip units on
            inverseRatio (np.float64): The number of units to flip, expressed as a fraction of all units

        Returns:
            np.ndarray: A copy of the original state with a number of units flipped
        """

        newState = state.copy()

        flipIndices = np.arange(newState.shape[0])
        np.random.shuffle(flipIndices)

        flipVector = np.ones_like(newState)
        np.put(flipVector, flipIndices[:int(np.ceil(inverseRatio*newState.shape[0]))], -1)
        newState*=flipVector

        return newState

    @classmethod
    def hammingDistance(cls, state1:np.ndarray, state2:np.ndarray):
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

        return np.sum(state1!=state2)
    
    def getHebbianMaxRatio(self)->np.float64:
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

