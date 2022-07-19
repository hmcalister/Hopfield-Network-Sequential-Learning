import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import HopfieldNetwork
from HopfieldNetwork.AbstractHopfieldNetwork import RelaxationException
import PatternManager
from HopfieldUtils import *
import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)
N = 64

numPatternsByTask = [20]

# HYPERPARAMS ---------------------------------------------------------------------------------------------------------
# Pattern generation params ---------------------------------------------------
mappingFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
patternManager = PatternManager.SequentialLearningPatternManager(N, mappingFunction)

# Network params---------------------------------------------------------------
energyFunction = HopfieldNetwork.EnergyFunction.BipolarEnergyFunction()
activationFunction = HopfieldNetwork.UpdateRule.ActivationFunction.BipolarHeaviside()
# updateRule = HopfieldNetwork.UpdateRule.Synchronous(activationFunction)
# updateRule = HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction)
# updateRule = HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)


EPOCHS=1000
# learningRule = HopfieldNetwork.LearningRule.Hebbian()
# learningRule = HopfieldNetwork.LearningRule.RehearsalHebbian(maxEpochs=EPOCHS, fracRehearse=0.2, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalHebbian(maxEpochs=EPOCHS, numRehearse=2, numPseudorehearsalSamples=10, updateRehearsalStatesFreq="Epoch")

learningRule = HopfieldNetwork.LearningRule.Delta(maxEpochs=EPOCHS)
# learningRule = HopfieldNetwork.LearningRule.RehearsalDelta(maxEpochs=EPOCHS, numRehearse=3, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalDelta(maxEpochs=EPOCHS, 
#     fracRehearse=1, trainUntilStable=False,
#     numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch",
#     keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True,
#     rejectLearnedStatesAsPseudoitems=False)

TEMPERATURE = 500
DECAY_RATE = np.round((1) * (TEMPERATURE/EPOCHS),3)
# learningRule = HopfieldNetwork.LearningRule.ThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE)
# learningRule = HopfieldNetwork.LearningRule.RehearsalThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE, 
#     temperatureDecay=DECAY_RATE,
#     numRehearse=3, updateRehearsalStatesFreq="Epoch")
# learningRule = HopfieldNetwork.LearningRule.PseudorehearsalThermalDelta(maxEpochs=EPOCHS, temperature=TEMPERATURE, temperatureDecay=DECAY_RATE, 
#     fracRehearse=1, trainUntilStable=False,
#     numPseudorehearsalSamples=512, updateRehearsalStatesFreq="Epoch", 
#     keepFirstTaskPseudoitems=True, requireUniquePseudoitems=True, 
#     rejectLearnedStatesAsPseudoitems=False)

# Network noise/error params --------------------------------------------------
allowableLearningStateError = 0.0
inputNoise = None
heteroassociativeNoiseRatio = 0.0

tasks = patternManager.createTasks(
    numPatternsByTask=numPatternsByTask
)

def testUpdateRuleConvergence(updateRule, numTrials):
    network = HopfieldNetwork.GeneralHopfieldNetwork(
        N=N,
        energyFunction=energyFunction,
        activationFunction=activationFunction,
        updateRule=updateRule,
        learningRule=learningRule,
        allowableLearningStateError=allowableLearningStateError,
        patternManager=patternManager,
        weights=np.random.normal(size=(N,N))
    )

    # Print network details
    print(network.getNetworkDescriptionString())
    print()

    network.learnPatterns(
        patterns=tasks[0].taskPatterns, 
        heteroassociativeNoiseRatio=heteroassociativeNoiseRatio, 
        inputNoise=inputNoise
    )

    # print("Learned Patterns")
    # for pattern in tasks[0].taskPatterns:
    #     print(f"{pattern}")
    # print()

    num_epochs_to_converge = []
    did_not_converge_count = 0
    for i in range(numTrials):
        print(f"TRIAL {i}/{numTrials}", end="\r")
        pattern = tasks[0].taskPatterns[i%numPatternsByTask[0]].copy()
        pattern = network.invertStateUnits(pattern, 0.1)
        epochCount = 0
        network.setState(pattern)
        while not network.isStable() and epochCount<100:
            # print(f"{epochCount}: {network.getState()}")
            try:
                network.relax(1)
            except RelaxationException:
                pass
            epochCount+=1
        if epochCount>=100:
            did_not_converge_count+=1
        else:
            num_epochs_to_converge.append(epochCount)
    print()
    print(f"Average Epochs to Convergence: {np.average(num_epochs_to_converge)}")
    print(f"Fraction of Trials that did not converge: {did_not_converge_count/numTrials}")
    print()
    return (np.average(num_epochs_to_converge), did_not_converge_count/numTrials)

updateRules = [HopfieldNetwork.UpdateRule.Synchronous(activationFunction),
HopfieldNetwork.UpdateRule.AsynchronousList(activationFunction),
HopfieldNetwork.UpdateRule.AsynchronousPermutation(activationFunction, energyFunction)]

convergenceAverages = []
fracNotConverge = []
for updateRule in updateRules:
    result = testUpdateRuleConvergence(updateRule, 1000)
    convergenceAverages.append(result[0])
    fracNotConverge.append(result[1])

plt.bar(["Sync", "AsyncList", "AsyncPerm"], convergenceAverages, color=["r", "g", "b"])
plt.ylabel("Epochs")
plt.xlabel("Update Rule")
plt.title("Average number of update cycles to converge")
plt.show()

plt.bar(["Sync", "AsyncList", "AsyncPerm"], fracNotConverge, color=["r", "g", "b"])
# plt.ylabel("Count")
plt.xlabel("Update Rule")
plt.title("Fraction of trials that did not converge")
plt.show()