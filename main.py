import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
N = 64

numPatternsByTask = [20]
numPatternsByTask.extend([1 for _ in range(3)])

# HYPERPARAMS ---------------------------------------------------------------------------------------------------------
# Pattern generation params ---------------------------------------------------
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(
    N, mappingFunction)

# Network params---------------------------------------------------------------
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)


EPOCHS = 1000
TEMPERATURE = 1000
DECAY_RATE = np.round((1) * (TEMPERATURE/EPOCHS), 3)

# learningRule = HopfieldNetwork.LearningRule.Delta(EPOCHS, trainUntilStable=False)

# learningRule = HopfieldNetwork.LearningRule.EnergyDirectedDelta(EPOCHS, trainUntilStable=False, alpha=0.)
 
learningRule = HopfieldNetwork.LearningRule.EnergyDirectedDeltaEWC(EPOCHS, trainUntilStable=False, alpha=0.7, 
        ewcTermGenerator=HopfieldNetwork.LearningRule.EWCTerm.SignCounterTerm(), ewcLambda=0.4,
        useOnlyFirstEWCTerm=True, vanillaEpochsFactor=0.0)

# learningRule = HopfieldNetwork.LearningRule.ElasticWeightConsolidationThermalDelta(
#     maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=0.0*DECAY_RATE,
#     ewcTermGenerator=HopfieldNetwork.LearningRule.EWCTerm.WeightDecayTerm(), ewcLambda=0.01,
#     useOnlyFirstEWCTerm=True, vanillaEpochsFactor=0.8)

# Network noise/error params --------------------------------------------------
allowableLearningStateError = 0.02
inputNoise = None
heteroassociativeNoiseRatio = 0.0

# SETUP ---------------------------------------------------------------------------------------------------------------
# Create network
network = HopfieldNetwork.GeneralHopfieldNetwork(
    N=N,
    energyFunction=energyFunction,
    activationFunction=activationFunction,
    updateRule=updateRule,
    learningRule=learningRule,
    allowableLearningStateError=allowableLearningStateError,
    patternManager=patternManager,
    weights=np.random.normal(size=(N, N))
)

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
    # print(f"Task Patterns:")
    # for pattern in task.getTaskPatterns():
    #     print(pattern)

    # This task has started, note this
    task.startEpoch = network.epochs
    # Learn the patterns
    accuracies, numStable = network.learnPatterns(
        patterns=task.taskPatterns,
        allTaskPatterns=patternManager.allTaskPatterns,
        heteroassociativeNoiseRatio=heteroassociativeNoiseRatio,
        inputNoise=inputNoise
    )

    # print(f"Network Weights:\n{network.weights}")

    taskPatternStabilities = np.vstack(
        [taskPatternStabilities, accuracies.copy()])
    numStableOverEpochs.extend(numStable)

    print(f"Most Recent Epoch Stable States: {numStable[-1]}")
    print()


# GRAPHING ------------------------------------------------------------------------------------------------------------
titleBasis = f"{network.N} Neuron, {network.learningRule}\n{network.allowableLearningStateError} Allowable Stability Error\n{heteroassociativeNoiseRatio} Heteroassociative Noise"
fileNameBasis = f"{network.N}Bipolar-{network.learningRule.infoString()}-{network.allowableLearningStateError}AllowableStabilityError-{heteroassociativeNoiseRatio}HeteroassociativeNoise"
taskEpochBoundaries = [task.startEpoch for task in tasks]

# plotSingleTaskStability(taskPatternStabilities[:, 0]*(len(tasks[0].taskPatterns)), taskEpochBoundaries[0],
#     title=f"{titleBasis}\n Stability of First Task",
#     legend=[str(tasks[0])], figsize=(12,6),
#     fileName=f"graphs/{fileNameBasis}--StabilityOfTask0.png"
#     )

plotTaskPatternStability(taskPatternStabilities, taskEpochBoundaries=taskEpochBoundaries, plotAverage=False,
                         title=f"{titleBasis}\n Stability by Task",
                         legend=[str(task) for task in tasks], figsize=(12, 6),
                         #  fileName=f"graphs/{fileNameBasis}--StabilityByTask.png"
                         )

# plotTotalStablePatterns(numStableOverEpochs,
#     title=f"{titleBasis}\n Total Stable States",
#     figsize=(12,6),
#     fileName=f"graphs/{fileNameBasis}--TotalStablePatterns.png"
#     )

# saveDataAsJSON(f"data/{fileNameBasis}.json",
#                networkDescription=network.getNetworkDescriptionJSON(),
#                trainingInformation={
#                    "inputNoise": inputNoise,
#                    "heteroassociativeNoiseRatio": heteroassociativeNoiseRatio
#                },
#                taskPatternStabilities=taskPatternStabilities.tolist(),
#                taskEpochBoundaries=taskEpochBoundaries,
#                numStableOverEpochs=numStableOverEpochs,
#                weights=network.weights.tolist(),
#                tasks=[np.array(task.taskPatterns).tolist() for task in patternManager.taskPatternManagers])
