import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
N = 1000

mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
learningRule = HopfieldNetwork.LearningRule.Delta(epochs=10)

network = HopfieldNetwork.GeneralHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule
)


tasks = patternManager.createTasks(
    numTasks=3,
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
    # print(f"Task Patterns: {task.taskPatterns}")
    # print(f"{seenPatterns=}")

    # randomMappings = patternManager.createRandomMappings(100, seenPatterns, changeRatio=0.05)

    accuracies, numStable = network.learnPatterns(task.taskPatterns, [task.taskPatterns for task in patternManager.taskPatternManagers])
    print(network.getWeights())

    # acc,numStable=network.measureTaskPatternStability([task.taskPatterns for task in patternManager.taskPatternManagers])
    if taskPatternStabilities is None:
        taskPatternStabilities = np.array(accuracies).copy()
    else:
        taskPatternStabilities = np.vstack([taskPatternStabilities, accuracies.copy()])
    print(f"Most Recent Epoch Accuracy: {accuracies[-1]}")

    numStableOverEpochs.extend(numStable)
    print(f"Most Recent Epoch Stable States: {numStable[-1]}")
    print()

plotTaskPatternStability(taskPatternStabilities, title="1000 Unit Bipolar Hopfield - Pattern Stability by Task",
    legend=[str(task) for task in tasks])

plotTotalStablePatterns(numStableOverEpochs, N, title="1000 Unit Bipolar Hopfield - Total Stable Patterns")