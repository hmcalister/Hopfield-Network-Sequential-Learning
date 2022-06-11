import copy
import HopfieldNetwork
import PatternManager
from HopfieldUtils import *
import numpy as np

np.set_printoptions(precision=2)
N = 64
NUMBER_RUNS = 10
MAX_EPOCHS = 1000
TEMPERATURE = 1000
DECAY_RATE = np.round((1) * (TEMPERATURE/MAX_EPOCHS),3)

numPatternsByTask = [40]
numPatternsByTask.extend([1 for _ in range(4)])

# HYPERPARAMS ---------------------------------------------------------------------------------------------------------
# Pattern generation params ---------------------------------------------------
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

# Network params---------------------------------------------------------------
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)

learning_rules = [
    # (HopfieldNetwork.LearningRule.Delta(maxEpochs=MAX_EPOCHS), "Vanilla Delta"),

    # (HopfieldNetwork.LearningRule.RehearsalDelta(maxEpochs=MAX_EPOCHS, fracRehearse=1, updateRehearsalStatesFreq="Epoch"), "Rehearsal"),

    # (HopfieldNetwork.LearningRule.PseudorehearsalDelta(maxEpochs=MAX_EPOCHS, fracRehearse=1, trainUntilStable=False,
    #     numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch", keepFirstTaskPseudoitems=True,
    #     requireUniquePseudoitems=True, rejectLearnedStatesAsPseudoitems=False),
    # "Pseudorehearsal"),

    # (HopfieldNetwork.LearningRule.PseudorehearsalDelta(maxEpochs=MAX_EPOCHS, fracRehearse=1, trainUntilStable=False,
    #     numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch", keepFirstTaskPseudoitems=True,
    #     requireUniquePseudoitems=True, rejectLearnedStatesAsPseudoitems=True),
    # "Spurious Pseudorehearsal"),

    (HopfieldNetwork.LearningRule.ThermalDelta(maxEpochs=MAX_EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE), "Thermal Delta"),

    (HopfieldNetwork.LearningRule.RehearsalThermalDelta(maxEpochs=MAX_EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE,
        fracRehearse=1, updateRehearsalStatesFreq="Epoch", rehearseFirstTaskOnly=True), "Thermal Rehearsal"),

    (HopfieldNetwork.LearningRule.PseudorehearsalThermalDelta(maxEpochs=MAX_EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE,
        fracRehearse=1, trainUntilStable=False,
        numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch", 
        keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True, 
        rejectLearnedStatesAsPseudoitems=False), "Thermal Pseudorehearsal"),

    (HopfieldNetwork.LearningRule.PseudorehearsalThermalDelta(maxEpochs=MAX_EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE,
        fracRehearse=1, trainUntilStable=False,
        numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch", 
        keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True, 
        rejectLearnedStatesAsPseudoitems=True), "Spurious Thermal Pseudorehearsal")
]
# Network noise/error params --------------------------------------------------
allowableLearningStateError = 0.02
inputNoise = None
heteroassociativeNoiseRatio = 0.05


# Array for each learning rule results
results_by_learning_rule = []


for learningRule in learning_rules:
    
    currLearningRuleResults = np.zeros(shape=(len(numPatternsByTask)*MAX_EPOCHS, len(numPatternsByTask)))
    run = 0
    while run < NUMBER_RUNS:
        print(f"RUN: {run+1}/{NUMBER_RUNS}")
        # SETUP ---------------------------------------------------------------------------------------------------------------
        # Create network
        network = HopfieldNetwork.GeneralHopfieldNetwork(
            N=N,
            energyFunction=energyFunction,
            activationFunction=activationFunction,
            updateRule=updateRule,
            learningRule=copy.deepcopy(learningRule[0]),
            allowableLearningStateError=allowableLearningStateError,
            patternManager=patternManager
        )

        # numPatternsByTask.extend([1 for i in range(10)])
        tasks = patternManager.createTasks(
            numPatternsByTask=numPatternsByTask
        )

        taskPatternStabilities = np.empty(shape=(0, len(numPatternsByTask)))
        numStableOverEpochs = []
        seenPatterns = []

        # Print network details
        print(network.getNetworkDescriptionString())
        print()

        # TRAINING ------------------------------------------------------------------------------------------------------------
        for task in tasks:
            seenPatterns.extend(task.getTaskPatterns())
            
            print(f"{task}")

            # This task has started, note this
            task.startEpoch = network.epochs
            # Learn the patterns
            accuracies, numStable = network.learnPatterns(
                patterns=task.taskPatterns, 
                allTaskPatterns=patternManager.allTaskPatterns, 
                heteroassociativeNoiseRatio=heteroassociativeNoiseRatio, 
                inputNoise=inputNoise
            )

            if isinstance(network.learningRule, HopfieldNetwork.LearningRule.PseudorehearsalDelta) and len(network.learningRule.stableStates) == 0:
                break

            taskPatternStabilities = np.vstack([taskPatternStabilities, accuracies.copy()])
            numStableOverEpochs.extend(numStable)

            print(f"Most Recent Epoch Stable States: {numStable[-1]}")
            print()

        if isinstance(network.learningRule, HopfieldNetwork.LearningRule.PseudorehearsalDelta) and len(network.learningRule.stableStates) == 0:
            print("ERR NO PSEUDOREHEARSAL STATES FOUND\n\n")
            continue

        run += 1 
        currLearningRuleResults += taskPatternStabilities.copy()
    currLearningRuleResults /= NUMBER_RUNS
    results_by_learning_rule.append(currLearningRuleResults[:, 0].copy())

plt.figure(figsize=(12,6))
for i in range(len(results_by_learning_rule)):
    plt.plot(results_by_learning_rule[i], label=learning_rules[i][1])

plt.title(f"Average results of first task by learning rule")
plt.xlabel("Epochs")
plt.ylabel("Average Task Accuracy")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.show()