# Hopfield Network: Numpy Implementation 

This is a *very* abstract implementation of a Hopfield network. The abstractions taken here are:

- Hopfield Network (To allow for different networks, like binary [0,1] or bipolar [-1,1])
- Energy Function (To define different energy functions to decide if a unit is stable)
- Learning Rule (To create different weight matrices when learning, which can help increase capacity)
- Update Rule (To update the network in different ways, which can help with stability)
- Activation function (To investigate how continuous states, as well as gradients during updates)

This implementation is intended to investigate sequential learning. Therefore, we also implement some more abstractions in the TaskPatternManager which allows for different patterns to be made for each task. 

This project is a work in progress. There are many bugs and some methods may be refactored or greatly changed in future!