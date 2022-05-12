import datetime
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import json

def plotTaskPatternStability(taskPatternStabilities:np.ndarray, taskEpochBoundaries:List[int], plotAverage:bool=True, title:str=None, 
    legend:List[str]=None, figsize=(8,6), fileName:str=None):
    """
    Plot the task pattern stability over many epochs, with each task getting its own line

    Args:
        taskPatternAccuracies (np.ndarray): A numpy array of dimension (numEpochs, numTasks)
            The first index walks over epochs, while the second index walks over tasks
        taskEpochBoundaries (List[int]): The list of epochs where each task starts being learned.
        plotAverage (bool, optional): A boolean to also plot the average task pattern stability over all tasks
            Defaults to True.
        title (str or None, optional): Title of the graph. Defaults to None.
        legend (List[str] or None, optional): A list of strings to use as the legend of the plot.
            Do not include the average legend. If None, use default legend. Defaults to None
        figsize (Tuple[int, int]): The size of the figure.
        fileName (str, optional): If not None, saves the plot to the file name. Defaults to None.
    """

    xRange = np.arange(taskPatternStabilities.shape[0])

    plt.figure(figsize=figsize)

    for i in range(taskPatternStabilities.shape[1]):
        label=f"Task {i+1}"
        if legend is not None:
            label = legend[i]
        # The index [i:, i] will select the i-th column (task i) but only
        # from time i onwards, so we do not plot tasks before they are learned
        plt.plot(xRange[taskEpochBoundaries[i]:], taskPatternStabilities[taskEpochBoundaries[i]:, i], marker="x", label=label)

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

def plotTotalStablePatterns(numStableOverEpochs:List[int], N:int=None, hebbianMaximumCapacity:np.float64=None,
    title:str=None, figsize=(8,6), fileName:str=None):
    """
    Plot the total number of stable learned patterns over epochs. 

    Args:
        numStableOverEpochs (List[int]): The number of stable learned patterns by epoch
        N (int or None, optional): The number of units in the network. If not None plots a line
                At the Hebbian maximum stable patterns. Defaults to None.
        hebbianMaximumCapacity (np.float64, optional): The maximum capacity of the network (expressed as a ratio of total units),
            or None (default). If None, no maximum capacity line is plotted.
        title (str or None, optional): Title of the graph. Defaults to None.
        figsize (Tuple[int, int]): The size of the figure.
        fileName (str, optional): If not None, saves the plot to the file name. Defaults to None.
    """

    plt.figure(figsize=figsize)

    plt.plot(numStableOverEpochs)
    if hebbianMaximumCapacity is not None:
        plt.axhline(hebbianMaximumCapacity*N, color='r', linestyle='--', label="Hebbian Max")
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

def saveDataAsJSON(fileName:str=None, **kwargs):
    """
    Save the given data (in kwargs) as a JSON file.
    The intention is to give a network description (from network.getNetworkDescriptionJSON), as well as
    any taskPatternStability and numStableOverEpochs.

    Args:
        fileName (str, optional): The name of the file to save to. If None, saves using a time stamp
            Defaults to None.
        kwargs: The names and values of items to store in the JSON file. Please be consistent!!!
    """

    if fileName is None:
        fileName = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    data = {}
    for (key, value) in kwargs.items():
        data[key]=value

    with open(fileName, 'w') as f:
        json.dump(data, f)

    print(f"SAVED TO {fileName}")