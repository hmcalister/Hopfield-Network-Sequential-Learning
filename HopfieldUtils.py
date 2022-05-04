from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plotTaskPatternStability(taskPatternStabilities:np.ndarray, title:str=None, legend:List[str]=None, plotAverage:bool=True, fileName:str=None):
    """
    Plot the task pattern stability over many epochs, with each task getting its own line

    Args:
        taskPatternAccuracies (np.ndarray): A numpy array of dimension (numEpochs, numTasks)
            The first index walks over epochs, while the second index walks over tasks
        title (str or None, optional): Title of the graph. Defaults to None.
        legend (List[str] or None, optional): A list of strings to use as the legend of the plot.
            Do not include the average legend. If None, use default legend. Defaults to None
        plotAverage (bool, optional): A boolean to also plot the average task pattern stability over all tasks
            Defaults to True.
        fileName (str, optional): If not None, saves the plot to the file name. Defaults to None.
    """

    xRange = np.arange(taskPatternStabilities.shape[0])

    for i in range(taskPatternStabilities.shape[1]):
        label=f"Task {i+1}"
        if legend is not None:
            label = legend[i]
        # The index [i:, i] will select the i-th column (task i) but only
        # from time i onwards, so we do not plot tasks before they are learned
        plt.plot(xRange[i:], taskPatternStabilities[i:, i], marker="x", label=label)

    if plotAverage:
        avgStability = []
        for i in range(len(taskPatternStabilities)):
            avgStability.append(np.average(taskPatternStabilities[i, :i+1]))
        plt.plot(avgStability, color='k', linestyle="-.", linewidth=3, label="Average Stability")

    plt.ylim(-0.05,1.05)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Task Pattern Stability")
    plt.tight_layout()

    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
        plt.cla()

def plotTotalStablePatterns(numStableOverEpochs:List[int], N:int=None, title:str=None, fileName:str=None):
    """
    Plot the total number of stable learned patterns over epochs. 

    Args:
        numStableOverEpochs (List[int]): The number of stable learned patterns by epoch
        N (int or None, optional): The number of units in the network. If not None plots a line
                At the Hebbian maximum stable patterns. Defaults to None.
        title (str or None, optional): Title of the graph. Defaults to None.
        fileName (str, optional): If not None, saves the plot to the file name. Defaults to None.
    """

    plt.plot(numStableOverEpochs)
    if N is not None:
        plt.axhline(0.14*N, color='r', linestyle='--', label="Hebbian Max")
    plt.axhline(max(numStableOverEpochs), color='b', linestyle='--', label="Actual Max")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Total stable learned patterns")
    plt.tight_layout()
    
    if fileName is None:
        plt.show()
    else:
        plt.savefig(fileName)
        plt.cla()