from cProfile import label
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

dataFiles = [
    "1000Bipolar-Delta-100 Epochs-NoneInput0HeteroassociativeNoise-0AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-AbsoluteInput0HeteroassociativeNoise-0AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-NoneInput0.05HeteroassociativeNoise-0.001AllowableError.json",
    "1000Bipolar-Delta-100 Epochs-NoneInput0.05HeteroassociativeNoise-0.01AllowableError.json",
    "1000Bipolar-Delta-100 Epochs-AbsoluteInput0.05HeteroassociativeNoise-0.01AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-AbsoluteInput0.05HeteroassociativeNoise-0AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-RelativeInput0HeteroassociativeNoise-0AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-NoneNoise-0.01AllowableError.json",
    # "1000Bipolar-Delta-100 Epochs-NoneNoise-0.001AllowableError.json",

    # "1000Bipolar-Hebbian-NoneNoise-0AllowableError.json",
    # "1000Bipolar-Hebbian-NoneNoise-0.001AllowableError.json",
    # "1000Bipolar-Hebbian-NoneNoise-0.005AllowableError.json",
    # "1000Bipolar-Hebbian-NoneNoise-0.01AllowableError.json",
]


for dataFile in dataFiles:
    dataFile = f"data/{dataFile}"
    print(dataFile)
    shutil.copyfile(dataFile, f"data/data.json")
    try:
        with open(f"data/data.json", "rb") as f:
            data = json.load(f)
            plt.plot(np.asarray(data["numStableOverEpochs"]), 
                label=f"{data['trainingInformation']['inputNoise']}, {data['trainingInformation']['heteroassociativeNoiseRatio']}, {data['networkDescription']['Allowable Learning State Error']}")
    except Exception as e:
        print(type(e))
        print(e)
        pass


plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', title="Input, Heteroassociative, State Error")
plt.title("1000 Unit Bipolar Hopfield Network - Total Stable Patterns\n 3 Tasks, 100 Patterns/Task\n Noise Experiments")
plt.xlabel("Epoch")
plt.ylabel("Total stable learned patterns")
plt.tight_layout()
plt.show()