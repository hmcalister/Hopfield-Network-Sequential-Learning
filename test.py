from typing import List
import numpy as np
import PatternManager
import HopfieldNetwork


mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(15, mappingFunction)

tasks:List[PatternManager.TaskPatternManager] = patternManager.createTasks(
    numTasks=3,
    numPatternsPerTask=5,
    numNearbyMappingsPerPattern=5
)

for task in tasks:
    print(task)
    for taskPattern in task.getTaskPatterns():
        print(taskPattern)
    for nearbyMapping in task.getNearbyMappings():
        print(f"{nearbyMapping[0]} -> \n{nearbyMapping[1]} : {patternManager.patternDistanceFunction(nearbyMapping[0], nearbyMapping[1])}")
    print(len(task.getNearbyMappings()))

randomMappings = patternManager.createRandomMappings(500, patternManager.allTaskPatterns)
for randomMapping in randomMappings:
    print(f"{randomMapping[0]} -> \n{randomMapping[1]} : {patternManager.patternDistanceFunction(randomMapping[0], randomMapping[1])}")
print(len(randomMappings))