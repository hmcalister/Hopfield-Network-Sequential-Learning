import HopfieldNetwork
import PatternManager
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

N = 1000
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
learningRule = HopfieldNetwork.LearningRule.Hebbian()

BipolarHeaviside = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()

network = HopfieldNetwork.BipolarHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule
)

mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

tasks = patternManager.createTasks(
    numTasks=3,
    numPatternsPerTask=33,
    numNearbyMappingsPerPattern=0
)

seenPatterns = []
taskAccuraciesOverEpochs = None
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
    if taskAccuraciesOverEpochs is None:
        taskAccuraciesOverEpochs = acc.copy()
    else:
        taskAccuraciesOverEpochs = np.vstack([taskAccuraciesOverEpochs, acc.copy()]) 
    print(f"{acc=}")

    numStable = network.measureNumStablePatterns(patternManager.allTaskPatterns)
    numStableOverEpochs.append(numStable)
    print(f"{numStable=}")


    print()

xRange = np.arange(len(taskAccuraciesOverEpochs))
for i in range(len(taskAccuraciesOverEpochs)):
    plt.plot(xRange[i:], taskAccuraciesOverEpochs[i:, i])
plt.ylim(-0.05,1.05)
plt.legend([str(task) for task in tasks], bbox_to_anchor=(1.04, 0.5), loc='center left')
plt.title("Accuracies by Task")
plt.xlabel("Task Number")
plt.ylabel("Task Accuracy")
plt.tight_layout()
plt.show()

plt.plot(numStableOverEpochs)
plt.axhline(0.14*N, color='r', linestyle='--', label="Hebbian Max")
plt.axhline(max(numStableOverEpochs), color='b', linestyle='--', label="Actual Max")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
plt.title("Total stable patterns")
plt.xlabel("Task Number")
plt.ylabel("Total stable patterns")
plt.tight_layout()
plt.show()