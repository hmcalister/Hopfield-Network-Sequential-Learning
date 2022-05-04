from typing import Callable, List, Tuple
from .PatternDistanceFunction import AbstractPatternDistanceFunction, HammingPatternDistance
from .TaskPatternManager import TaskPatternManager
import numpy as np

class SequentialLearningPatternManager():

    def __init__(self, patternSize:int, mappingFunction:Callable, name:str="SequentialLearningPatternManager",
        patternDistanceFunction:AbstractPatternDistanceFunction=HammingPatternDistance()):
        """
        Create a new SequentialLearningPatternManager.
        Will create patterns for a several tasks.
        Exposes methods to create a number of tasks each with a number of patterns to learn
        The size given should be the size of the network.

        Args:
            size (int): The size of the patterns to generate
            mappingFunction (Callable): A function to map from a Gaussian random value to the correct pattern domain
                e.g. for a bipolar network we can use a BipolarHeaviside function that maps (-inf,0) to -1, [0,inf) to 1
            name (str): The name of this task manager.
            patternDistanceFunction (AbstractPatternDistanceFunction, optional): A function to measure the distance between two patterns
                Defaults to HammingPatternDistance
        """


        self.name:str = name
        self.patternSize:int = patternSize
        self.mappingFunction:Callable = mappingFunction
        self.patternDistanceFunction = patternDistanceFunction

    def __str__(self)->str:
        return self.name

    def comparePatterns(pattern1:np.ndarray, pattern2:np.ndarray)->bool:
        """
        Check if two patterns are the same (with the possibility of negation)

        Args:
            pattern1 (np.ndarray): The first pattern to check
            pattern2 (np.ndarray): The second pattern to check

        Returns:
            bool: _description_
        """

        return np.array_equal(pattern1, pattern2) or np.array_equal(-1*pattern1, pattern2)


    def _generatePattern(self)->np.ndarray:
        """
        Generate a new pattern of size self.size from a vector of gaussian random variables passed through the mapping function

        Returns:
            np.ndarray: A new patterns of size self.size
        """

        return self.mappingFunction(np.random.normal(size=self.patternSize))

    def _generateNearbyPattern(self, pattern:np.ndarray, fracChange:np.float64 = 0.2)->np.ndarray:
        """
        Create a pattern that is nearby to the given pattern.
        The nearby pattern is created by changing a random number of units (determined by fracChange)

        Args:
            pattern (np.ndarray): The base pattern, the pattern to generate a nearby pattern of
            fracChange (np.float64): The fraction of units to alter

        Returns:
            np.ndarray: A pattern that is nearby to the given pattern
        """

        numUnits = int(np.ceil(self.patternSize * fracChange))
        nearbyPattern = pattern.copy()

        # Get a set of indices for the pattern
        indices = np.arange(self.patternSize)
        # Shuffle the indices
        np.random.shuffle(indices)
        # Take the first few, this ensures unique indices
        indices = indices[:numUnits]

        # Use a donor pattern to get new values for indices
        donorPattern = self._generatePattern()
        for index in indices:
            nearbyPattern[index] = donorPattern[index]

        # If we didn't actually change the pattern, try again!
        if SequentialLearningPatternManager.comparePatterns(nearbyPattern, pattern): return self._generateNearbyPattern(pattern)

        return nearbyPattern

    def createTasks(self, numTasks:int, numPatternsPerTask:int, numNearbyMappingsPerPattern:int=0)->List[TaskPatternManager]:
        """
        Create a number of tasks to be learned sequentially by a Hopfield network. Each task will have
        (numPatternsPerTask) patterns, and each pattern will have (numNearbyMappingsPerPattern) nearby mappings to test.
        Tasks are named automatically like Task_1, Task_2...
        Please note: To measure the random mappings you should use SequentialLearningPatternManager.createRandomMappings()
            This method takes a list of patterns (the tasks learned so far) so we can create random mappings according to what
            the network has seen so far.
        Also note: The total number of nearbyMappings created will be numTasks*numPatternsPerTask*numNearbyMappingsPerPattern 
            This can grow very quickly!!! Beware!!!

        Args:
            numTasks (int): The number of tasks to create
            numPatternsPerTask (int): The number of patterns to create per task
            numNearbyMappingsPerPattern (int): The number of nearby mappings to create per task

        Returns:
            List[TaskPatternManager]: A list of TaskPatternManagers that represent the tasks. 
                Each TaskPatternManager holds the taskPatterns and nearbyMappings for that task
        """

        self.numTasks:int = numTasks
        self.numPatternsPerTask:int = numPatternsPerTask
        self.numNearbyMappingsPerPattern:int = numNearbyMappingsPerPattern

        # self.allTaskPatterns is organized in a specific way: all patterns for one task are contiguous
        # Task i has patterns from [self.numTasks*i:self.numTasks*(i+1)]
        self.allTaskPatterns:List[np.ndarray] = self._createTaskPatterns(self.numTasks*self.numPatternsPerTask)
        # self.allNearbyMappings is organized in a specific way: all nearbyMappings for one task are contiguous
        # Task i has nearby mappings from [self.numPatternsPerTask*self.numNearbyMappingsPerPattern*i:self.numPatternsPerTask*self.numNearbyMappingsPerPattern*(i+1)]
        self.allNearbyMappings:List[Tuple[np.ndarray, np.ndarray]] = self._createNearbyMappings(self.numNearbyMappingsPerPattern)

        self.taskPatternManagers:List[TaskPatternManager] = []
        for task in range(self.numTasks):
            taskPatterns = self.allTaskPatterns[self.numPatternsPerTask*task : self.numPatternsPerTask*(task+1)]
            nearbyMappings = self.allNearbyMappings[self.numPatternsPerTask*self.numNearbyMappingsPerPattern*task : 
                    self.numPatternsPerTask*self.numNearbyMappingsPerPattern*(task+1)]
            self.taskPatternManagers.append(TaskPatternManager(
                name=f"Task_{task}",
                taskPatterns=taskPatterns.copy(),
                nearbyMappings=nearbyMappings.copy()
            ))
            
            # for task in self.taskPatternManagers:
            #     print(f"{task}: {task.taskPatterns}")
            # print()

        return self.taskPatternManagers

    def _createTaskPatterns(self, numPatterns:int)->List[np.ndarray]:
        """
        Create a number of patterns for the network to learn.
        Patterns created from this method are guarunteed to be unique (including negated patterns)
        Patterns may be correlated. A future method may implement orthogonal patterns.
        List of patterns is returned but also stored in this object.
        Warning: Will overwrite previous patterns stored

        Args:
            numPatterns (int): The number of patterns to create

        Returns:
            List[np.ndarray]: The patterns created from this method

        Raises:
            ValueError: If we are unable to create the number of unique patterns requested
        """

        # Define how many times we attempt to create unique patterns
        # If we exceed this, we throw an error
        MAX_ATTEMPTS = 10*numPatterns
        currSteps = 0
        taskPatterns = []

        while(len(taskPatterns) < numPatterns):
            # Ensure we have not tried too many times to make this pattern
            currSteps+=1
            if currSteps > MAX_ATTEMPTS:
                raise ValueError()
            
            # Generate a new pattern
            currPattern = self._generatePattern()

            # Check if the pattern is unique
            # If the pattern is not unique, restart the loop
            patternUnique = True
            for pattern in taskPatterns:
                if SequentialLearningPatternManager.comparePatterns(currPattern, pattern): 
                    patternUnique = False
                    break
            if not patternUnique: continue

            # Add this pattern to the list of patterns and reset the step counter
            taskPatterns.append(currPattern.copy())
            currSteps = 0

        self.allTaskPatterns:List[np.ndarray] = taskPatterns
        # Remove any previous nearby and random pattern maps
        self.nearbyMappings = None
        self.randomMappings = None

        return self.allTaskPatterns

    def _createNearbyMappings(self, numNearby:int)->List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a number of patterns that are (roughly) nearby to the task patterns
        Returns a list with key value pairs (nearbyPattern, desiredLearnedPattern)
        Requires the existence of self.allTaskPatterns i.e. call self._createTaskPatterns()

        Args:
            numNearby (int): The number of nearby patterns to make for each task pattern

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of key value pairs of the form (nearbyPattern, desiredTaskPattern)

        Raises:
            ValueError: If we are unable to create the number of unique patterns requested
        """

        # Define how many times we attempt to create unique patterns
        # If we exceed this, we throw an error
        MAX_ATTEMPTS = 10*numNearby
        currSteps = 0
        nearbyMappings = []

        # Create a number of nearby patterns for every task pattern
        for desiredTaskPattern in self.allTaskPatterns:
            desiredNearbyCount = 0
            currSteps=0
            while desiredNearbyCount < numNearby:
                # Ensure we have not tried too many times to make this pattern
                currSteps+=1
                if currSteps > MAX_ATTEMPTS:
                    raise ValueError()
                
                currNearby = self._generateNearbyPattern(desiredTaskPattern)

                # Check if the pattern is unique
                # If the pattern is not unique, restart the loop
                patternUnique = True
                for pattern, expected in nearbyMappings:
                    if SequentialLearningPatternManager.comparePatterns(currNearby, pattern):
                        patternUnique = False
                        break
                if not patternUnique: continue

                # Check that this nearby is actually closest to the desired pattern
                patternClosestToDesired=True
                desiredTaskPatternDistance = self.patternDistanceFunction(desiredTaskPattern, currNearby)
                # Check the distance to the desired taskPattern to the other taskPatterns
                for pattern in self.allTaskPatterns:
                    # Skip the desired task pattern
                    if SequentialLearningPatternManager.comparePatterns(desiredTaskPattern, pattern): continue
                    
                    # If the the current pattern is closer to a different pattern, we drop the current pattern
                    if self.patternDistanceFunction(pattern, currNearby) <= desiredTaskPatternDistance:
                        patternClosestToDesired = False
                        break
                if not patternClosestToDesired: continue

                # Finally, we have a currNearby that is unique and closest to our desiredTaskPattern
                # Add it to the dict
                nearbyMappings.append( (currNearby.copy(), desiredTaskPattern) )
                desiredNearbyCount += 1
                # And reset the steps for the next pattern
                currSteps = 0
            
        # Finally we have finished creating our nearby patterns
        self.nearbyMappings = nearbyMappings
        return self.nearbyMappings

    def createRandomMappings(self, numRandom:int, learnedPatterns:List[np.ndarray])->List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a number of patterns that are random and check what pattern we expect them to map to in the given learned patterns
        Returns a list with key value pairs (randomPattern, desiredLearnedPattern)

        Args:
            numRandom (int): The number of random patterns to make
            learnedPatterns (List[np.ndarray]): The list of learned patterns of the network so far

        Returns:
            List[Tuple[np.ndarray, nd.ndarray]]: A list of key value pairs of the form (randomPattern, desiredLearnedPattern)
        """

        randomMappings = []

        while len(randomMappings) < numRandom:
            # Create a new random pattern
            currPattern = self._generatePattern()

            distances = np.array([self.patternDistanceFunction(currPattern, pattern) for pattern in learnedPatterns])
            minDistance = np.min(distances)
            # We check that the min distance is unique
            # We find all indices that are less than or equal to the minDistance
            # Then sum the count of these indices
            # This sum should be exactly one
            if np.sum(distances <= minDistance) > 1:
                continue

            # We now know that there is a unique minimum distance
            # Let's find the taskPattern that it corresponds to
            minIndex = np.argmin(minDistance)
            minTaskPattern = learnedPatterns[minIndex]

            # Finally, set the key value pair
            randomMappings.append( (currPattern.copy(), minTaskPattern) )

        self.randomMappings = randomMappings
        return self.randomMappings
            
    def createRandomMappings(self, numRandom:int, learnedPatterns:List[np.ndarray], changeRatio:np.float64)->List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a number of random patterns that are nearby the given pattterns
        and check what pattern we expect them to map to in the given learned patterns
        Returns a list with key value pairs (randomPattern, desiredLearnedPattern)

        Args:
            numRandom (int): The number of random patterns to make
            learnedPatterns (List[np.ndarray]): The list of learned patterns of the network so far
            changeRatio(np.float64): The fraction of units to change in each pattern

        Returns:
            List[Tuple[np.ndarray, nd.ndarray]]: A list of key value pairs of the form (randomPattern, desiredLearnedPattern)
        """

        randomMappings = []

        while len(randomMappings) < numRandom:
            # Choose a random pattern from the learned patterns
            parentPattern = learnedPatterns[np.random.randint(len(learnedPatterns))]
            # Create a new random pattern
            currPattern = self._generateNearbyPattern(parentPattern, changeRatio)

            distances = np.array([self.patternDistanceFunction(currPattern, pattern) for pattern in learnedPatterns])
            minDistance = np.min(distances)
            # We check that the min distance is unique
            # We find all indices that are less than or equal to the minDistance
            # Then sum the count of these indices
            # This sum should be exactly one
            if np.sum(distances <= minDistance) > 1:
                continue

            # We now know that there is a unique minimum distance
            # Let's find the taskPattern that it corresponds to
            minIndex = np.argmin(minDistance)
            minTaskPattern = learnedPatterns[minIndex]

            # Finally, set the key value pair
            randomMappings.append( (currPattern.copy(), minTaskPattern) )

        self.randomMappings = randomMappings
        return self.randomMappings