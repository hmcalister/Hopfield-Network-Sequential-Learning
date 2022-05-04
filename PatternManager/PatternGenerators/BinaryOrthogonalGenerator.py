import numpy as np

class BinaryOrthogonalGenerator():

    def __init__(self, N:int, activeUnits:int):
        """
        Return a new instance of a BinaryOrthogonalGenerator mapping function
        Takes a size of vector to return (N), and a ratio to set to ones (activeUnits)
        e.g. if N=10 and activeUnits=2, we will produce patterns like (1,1,0,0...), (0,0,1,1,....) etc
        Actual patterns have random unit placement but the concept remains

        Args:
            N (int): The length of the pattern to create
            activeUnits (int): The number of active units in a pattern. Should be (significantly) less than N
        """

        self.N = N
        self.activeUnits = activeUnits

        self.unitOrder = np.random.permutation(np.arange(self.N))
        self.createdPatterns = 0


    def __call__(self, state:np.ndarray)->np.ndarray:
        state = np.zeros(self.N)
        for i in range(self.activeUnits):
            unitOrderIndex = (self.activeUnits*self.createdPatterns+i)%self.N
            activeUnit = self.unitOrder[unitOrderIndex]
            state[activeUnit] = 1
        self.createdPatterns+=1

        return state

    def __str__(self):
        return "BinaryOrthogonalGenerator"