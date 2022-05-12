from turtle import update
import HopfieldNetwork
from HopfieldNetwork import LearningRule
from HopfieldNetwork import UpdateRule
import PatternManager
from HopfieldUtils import *
import numpy as np

np.set_printoptions(precision=2)
N = 1000

mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)


energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)
learningRule = HopfieldNetwork.LearningRule.Delta(epochs=100)
allowableLearningStateError = 0.001

inputNoise = None
heteroassociativeNoiseRatio = 0.05

network = HopfieldNetwork.GeneralHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule,
    allowableLearningStateError=allowableLearningStateError
)

tasks = patternManager.createTasks(
    numTasks=3,
    numPatternsPerTask=100
)

seenPatterns = []
taskPatternStabilities = None
numStableOverEpochs = []

print(network.getNetworkDescriptionString())
print()

for task in tasks:
    print(f"{task}")
    seenPatterns.extend(task.getTaskPatterns())
    # print(f"Task Patterns: {task.taskPatterns}")
    # print(f"{seenPatterns=}")

    # randomMappings = patternManager.createRandomMappings(100, seenPatterns, changeRatio=0.05)

    accuracies, numStable = network.learnPatterns(task.taskPatterns, [task.taskPatterns for task in patternManager.taskPatternManagers], 
        heteroassociativeNoiseRatio=heteroassociativeNoiseRatio, inputNoise=inputNoise)
    # print(network.getWeights())

    if taskPatternStabilities is None:
        taskPatternStabilities = np.array(accuracies).copy()
    else:
        taskPatternStabilities = np.vstack([taskPatternStabilities, accuracies.copy()])
    print(f"Most Recent Epoch Accuracy: {accuracies[-1]}")

    numStableOverEpochs.extend(numStable)
    print(f"Most Recent Epoch Stable States: {numStable[-1]}")
    print()

titleBasis = f"Bipolar {network.N} Unit - {network.learningRule}\n {inputNoise} Input, {heteroassociativeNoiseRatio} Heteroassociative Noise, {network.allowableLearningStateError} Allowable Stability Error"
fileNameBasis = f"{network.N}Bipolar-{network.learningRule}-{inputNoise}Input{heteroassociativeNoiseRatio}HeteroassociativeNoise-{network.allowableLearningStateError}AllowableError"

plotTaskPatternStability(taskPatternStabilities, taskEpochBoundaries=[network.learningRule.epochs*i for i in range(len(tasks))], 
    title=f"{titleBasis}\n Stability by Task",
    legend=[str(task) for task in tasks], figsize=(12,6),
    # fileName=f"graphs/{fileNameBasis}-StabilityByTask.png"
    )

plotTotalStablePatterns(numStableOverEpochs, N, hebbianMaximumCapacity=network.getHebbianMaxRatio(),
    title=f"{titleBasis}\n Total Stable Patterns", 
    figsize=(12,6),
    # fileName=f"graphs/{fileNameBasis}-TotalStablePatterns.png"
    )

saveDataAsJSON(f"data/{fileNameBasis}.json", 
    networkDescription = network.getNetworkDescriptionJSON(),
    trainingInformation= {
        "inputNoise": inputNoise,
        "heteroassociativeNoiseRatio": heteroassociativeNoiseRatio
    },
    taskPatternStabilities = taskPatternStabilities.tolist(),
    taskEpochBoundaries = [network.learningRule.epochs*i for i in range(len(tasks))],
    numStableOverEpochs = numStableOverEpochs,
    weights=network.weights.tolist(),
    tasks=[np.array(task.taskPatterns).tolist() for task in patternManager.taskPatternManagers])