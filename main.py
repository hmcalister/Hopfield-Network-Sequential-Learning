import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np

np.set_printoptions(precision=2)
N = 100

# HYPERPARAMS ---------------------------------------------------------------------------------------------------------
# Pattern generation params ---------------------------------------------------
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

# Network params---------------------------------------------------------------
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)



# learningRule = HopfieldNetwork.LearningRule.Hebbian()
# learningRule = HopfieldNetwork.LearningRule.RehearsalHebbian(maxEpochs=1, fracRehearse=0.2, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalHebbian(maxEpochs=1, numRehearse=2, numPseudorehearsalSamples=10, updateRehearsalStatesFreq="Epoch")

# learningRule = HopfieldNetwork.LearningRule.Delta(maxEpochs=100)
# learningRule = HopfieldNetwork.LearningRule.RehearsalDelta(maxEpochs=100, numRehearse=3, updateRehearsalStatesFreq="Epoch")
learningRule = HopfieldNetwork.LearningRule.PseudorehearsalDelta(maxEpochs=10, numRehearse=3, keepPreviousStableStates=False, 
    numPseudorehearsalSamples=32, updateRehearsalStatesFreq="Epoch")

# Network noise/error params --------------------------------------------------
allowableLearningStateError = 0.01
inputNoise = None
heteroassociativeNoiseRatio = 0

# SETUP ---------------------------------------------------------------------------------------------------------------
# Create network
network = HopfieldNetwork.GeneralHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule,
    allowableLearningStateError=allowableLearningStateError,
    patternManager=patternManager
)

# Create tasks
numPatternsByTask = [20]
numPatternsByTask.extend([1 for i in range(10)])
# numPatternsByTask.extend([1 for i in range(10)])
tasks = patternManager.createTasks(
    numPatternsByTask=numPatternsByTask
)

# We have currently seen no patterns
seenPatterns = []
# We declare an empty matrix of stabilities
# First index is epoch (currently 0) second is task index
taskPatternStabilities = np.empty(shape=(0, len(tasks)))
# And we track stability over epochs
numStableOverEpochs = []

# Print network details
print(network.getNetworkDescriptionString())
print()

# TRAINING ------------------------------------------------------------------------------------------------------------
for task in tasks:
    seenPatterns.extend(task.getTaskPatterns())
    
    print(f"{task}")
    # print(f"Task Patterns: {task.taskPatterns}")
    # print(f"{seenPatterns=}")

    # This task has started, note this
    task.startEpoch = network.epochs
    # Learn the patterns
    accuracies, numStable = network.learnPatterns(
        patterns=task.taskPatterns, 
        allTaskPatterns=patternManager.allTaskPatterns, 
        heteroassociativeNoiseRatio=heteroassociativeNoiseRatio, 
        inputNoise=inputNoise
    )

    taskPatternStabilities = np.vstack([taskPatternStabilities, accuracies.copy()])
    numStableOverEpochs.extend(numStable)

    print(f"Most Recent Epoch Stable States: {numStable[-1]}")
    print()


# GRAPHING ------------------------------------------------------------------------------------------------------------
titleBasis = f"Bipolar {network.N} Unit\n {network.learningRule}\n {network.allowableLearningStateError} Allowable Stability Error, {heteroassociativeNoiseRatio} Heteroassociative Noise"
fileNameBasis = f"{network.N}Bipolar-{network.learningRule}-{network.allowableLearningStateError}AllowableStabilityError-{heteroassociativeNoiseRatio}-HeteroassociativeNoise"
taskEpochBoundaries = [task.startEpoch for task in tasks]

plotSingleTaskStability(taskPatternStabilities[:, 0]*(len(tasks[0].taskPatterns)), taskEpochBoundaries[0],
    title=f"{titleBasis}\n Stability of {str(tasks[0])}",
    legend=[str(tasks[0])], figsize=(12,6),
    fileName=f"graphs/{fileNameBasis}--StabilityOfTask0.png"
    )

plotTaskPatternStability(taskPatternStabilities, taskEpochBoundaries=taskEpochBoundaries, 
    title=f"{titleBasis}\n Stability by Task",
    legend=[str(task) for task in tasks], figsize=(12,6),
    fileName=f"graphs/{fileNameBasis}--StabilityByTask.png"
    )

plotTotalStablePatterns(numStableOverEpochs, N,
    title=f"{titleBasis}\n Total Stable Patterns", 
    figsize=(12,6),
    fileName=f"graphs/{fileNameBasis}--TotalStablePatterns.png"
    )

saveDataAsJSON(f"data/{fileNameBasis}.json", 
    networkDescription = network.getNetworkDescriptionJSON(),
    trainingInformation= {
        "inputNoise": inputNoise,
        "heteroassociativeNoiseRatio": heteroassociativeNoiseRatio
    },
    taskPatternStabilities = taskPatternStabilities.tolist(),
    taskEpochBoundaries = taskEpochBoundaries,
    numStableOverEpochs = numStableOverEpochs,
    weights=network.weights.tolist(),
    tasks=[np.array(task.taskPatterns).tolist() for task in patternManager.taskPatternManagers])