import HopfieldNetwork
import PatternManager
import numpy as np

N = 1000
energyFunction = HopfieldNetwork.EnergyFunction.BinaryEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BinaryHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
learningRule = HopfieldNetwork.LearningRule.Hebbian()

BipolarHeaviside = HopfieldNetwork.UpdateRule.ActivationFunction.BinaryHeaviside()

network = HopfieldNetwork.BinaryHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule
)

mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BinaryHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

tasks = patternManager.createTasks(
    numTasks=3,
    numPatternsPerTask=10,
    numNearbyMappingsPerPattern=0
)

seenPatterns = []

print(network)
for task in tasks:
    # print(task.getTaskPatterns())
    seenPatterns.extend(task.getTaskPatterns()) 
    randomMappings = patternManager.createRandomMappings(100, seenPatterns, changeRatio=0.005)
    acc=network.learnPatterns(task.getTaskPatterns(), randomMappings)
    print(acc)
    # print(network.weights)
