from typing import Callable
from .ActivationFunction import AbstractActivationFunction
from abc import ABC, abstractmethod
import numpy as np


class AbstractUpdateRule(ABC):
    MAX_STEPS = 500

    @abstractmethod
    def __init__(self, activationFunction:AbstractActivationFunction):
        self.activationFunction = activationFunction
        self.inputNoiseType:str = None
        self.inputNoise:Callable = self.noInputNoise
        pass
        
    @abstractmethod
    def __call__(self, currentState:np.ndarray, weights:np.ndarray)->np.ndarray:
        """
        Find the next state from a current state and weights

        Args:
            currentState (np.ndarray): The current state of the network. Must have dimension N and type float64
            weights (np.ndarray): The weights of the network. Must have dimension N*N and type float64

        Returns:
            np.ndarray: The next state of the network
        """
        pass

    @abstractmethod
    def __str__(self):
        return "AbstractUpdateRule"

    def clearInputNoiseType(self):
        """
        Remove the input noise, setting it to None (noInputNoise)
        """

        self.setInputNoiseType(None)

    def setInputNoiseType(self, inputNoise:str):
        """
        Set the input noise for this update rule

        Args:
            inputNoise (str or None, optional): String on whether to apply input noise to the units before activation
                - "Absolute": Apply absolute noise to the state, a Gaussian of mean 0 std 1
                - "Relative": Apply relative noise to the state, a Gaussian of mean and std determined by the state vector
                - None: No noise. Default

        Return:
            Callable: A function that outputs the inputNoise vector determined by the inputNoise type
        """

        self.inputNoiseType = inputNoise
        if inputNoise is None or inputNoise=="None":
            noiseFunction = self.noInputNoise
        elif inputNoise=="Absolute":
            noiseFunction = self.absoluteInputNoise
        elif inputNoise=="Relative":
            noiseFunction = self.relativeInputNoise
            # print(f"{np.max(noiseVector)=},\n{np.max(np.abs(currentState))=}\n")
        else:
            print("INPUT NOISE NOT RECOGNIZED")
            exit()

        self.inputNoise = noiseFunction

    def noInputNoise(self, currentState:np.ndarray) -> np.ndarray:
        """
        Return a zero vector like the current state, i.e. no input noise

        Args:
            currentState (np.ndarray): The input state

        Returns:
            np.ndarray: A zero vector like the input state
        """

        return np.zeros_like(currentState)

    def absoluteInputNoise(self, currentState:np.ndarray) -> np.ndarray:
        """
        Return a noise vector that is absolute, mean 0 std 1 

        Args:
            currentState (np.ndarray): The input state

        Returns:
            np.ndarray: A noise vector that is absolute
        """

        return np.random.normal(0, 1, currentState.shape[0])

    def relativeInputNoise(self, currentState:np.ndarray) -> np.ndarray:
        """
        Return a noise vector that is relative, mean 0 std is mean of currentState

        Args:
            currentState (np.ndarray): The input state

        Returns:
            np.ndarray: A noise vector that is relative
        """
        if np.abs(np.mean(currentState))==0: 
            return np.zeros_like(currentState)
        return np.random.normal(0, np.abs(np.std(currentState)), currentState.shape[0])


    
