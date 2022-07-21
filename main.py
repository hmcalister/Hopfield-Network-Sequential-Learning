import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
N = 64

numPatternsByTask = [50]
numPatternsByTask.extend([1 for _ in range(5)])

# HYPERPARAMS ---------------------------------------------------------------------------------------------------------
# Pattern generation params ---------------------------------------------------
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(
    N, mappingFunction)

# Network params---------------------------------------------------------------
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
# updateRule = HopfieldNetwork.UpdateRule.Synchronous(activationFunction)
# updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(
    activationFunction, energyFunction)


EPOCHS = 500
# learningRule = HopfieldNetwork.LearningRule.Hebbian()
# learningRule = HopfieldNetwork.LearningRule.RehearsalHebbian(maxEpochs=EPOCHS, fracRehearse=0.2, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalHebbian(maxEpochs=EPOCHS, numRehearse=2, numPseudorehearsalSamples=10, updateRehearsalStatesFreq="Epoch")

# learningRule = HopfieldNetwork.LearningRule.Delta(maxEpochs=EPOCHS)
# learningRule = HopfieldNetwork.LearningRule.RehearsalDelta(maxEpochs=EPOCHS, numRehearse=3, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalDelta(maxEpochs=EPOCHS,
#     fracRehearse=1, trainUntilStable=False,
#     numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch",
#     keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True,
#     rejectLearnedStatesAsPseudoitems=False)

TEMPERATURE = 1000
DECAY_RATE = np.round((1) * (TEMPERATURE/EPOCHS), 3)
# learningRule = HopfieldNetwork.LearningRule.ThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE)
# learningRule = HopfieldNetwork.LearningRule.RehearsalThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE,
#     temperatureDecay=DECAY_RATE,
#     fracRehearse=1, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE,
#     fracRehearse=1, trainUntilStable=False,
#     numPseudorehearsalSamples=2048, updateRehearsalStatesFreq="Epoch",
#     keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True,
#     rejectLearnedStatesAsPseudoitems=True)

learningRule = HopfieldNetwork.LearningRule.ElasticWeightConsolidationThermalDelta(
        maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=0.0*DECAY_RATE,
        ewcTermGenerator=HopfieldNetwork.LearningRule.EWCTerm.HebbianTerm, ewcLambda=0.05,
        useOnlyFirstEWCTerm=True)

# Network noise/error params --------------------------------------------------
allowableLearningStateError = 0.02
inputNoise = None
heteroassociativeNoiseRatio = 0.05

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
                         fileName=f"graphs/{fileNameBasis}--StabilityByTask.png"
                         )

# plotTaskPatternStability(taskPatternStabilities, taskEpochBoundaries=taskEpochBoundaries, plotAverage=False,
#     title=f"{titleBasis}\n Stability by Task",
#     legend=[str(task) for task in tasks], figsize=(12,6),
#     fileName=f"graphs/{fileNameBasis}--StabilityByTask.png"
#     )

# plotTotalStablePatterns(numStableOverEpochs,
#     title=f"{titleBasis}\n Total Stable States",
#     figsize=(12,6),
#     fileName=f"graphs/{fileNameBasis}--TotalStablePatterns.png"
#     )

saveDataAsJSON(f"data/{fileNameBasis}.json",
               networkDescription=network.getNetworkDescriptionJSON(),
               trainingInformation={
                   "inputNoise": inputNoise,
                   "heteroassociativeNoiseRatio": heteroassociativeNoiseRatio
               },
               taskPatternStabilities=taskPatternStabilities.tolist(),
               taskEpochBoundaries=taskEpochBoundaries,
               numStableOverEpochs=numStableOverEpochs,
               weights=network.weights.tolist(),
               tasks=[np.array(task.taskPatterns).tolist() for task in patternManager.taskPatternManagers])
