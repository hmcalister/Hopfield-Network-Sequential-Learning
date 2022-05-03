import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
N = 1000
network = HopfieldNetwork.BinaryHopfieldNetwork(N)
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BinaryHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

# energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
# activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
# updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
# learningRule = HopfieldNetwork.LearningRule.Hebbian()

# BipolarHeaviside = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()

# network = HopfieldNetwork.GeneralHopfieldNetwork(
#     N=N,
#     energyFunction=energyFunction,
#     activationFunction=activationFunction,
#     updateRule=updateRule,
#     learningRule=learningRule
# )


tasks = patternManager.createTasks(
    numTasks=10,
    numPatternsPerTask=10,
    numNearbyMappingsPerPattern=0
)

seenPatterns = []
taskPatternStabilities = None
numStableOverEpochs = []

print(network)
for task in tasks:
    print(f"{task}")
    seenPatterns.extend(task.getTaskPatterns())
    # print(f"{seenPatterns=}")

    # randomMappings = patternManager.createRandomMappings(100, seenPatterns, changeRatio=0.05)

    network.learnPatterns(task.taskPatterns)
    # print(network.getWeights())

    acc,numStable=network.measureTaskPatternStability([task.taskPatterns for task in patternManager.taskPatternManagers])
    if taskPatternStabilities is None:
        taskPatternStabilities = acc.copy()
    else:
        taskPatternStabilities = np.vstack([taskPatternStabilities, acc.copy()]) 
    print(f"{acc=}")

    numStableOverEpochs.append(numStable)
    print(f"{numStable=}")
    print()

plotTaskPatternStability(taskPatternStabilities, fileName="graphs/MappedBinaryTaskPatternStability.png")
plotTotalStablePatterns(numStableOverEpochs, N, fileName="graphs/MappedBinaryTotalStablePatterns.png")