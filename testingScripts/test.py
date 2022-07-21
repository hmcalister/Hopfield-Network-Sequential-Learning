import numpy as np

w_e = np.array([[1,1,1],[1,2,4],[3,3,3],])
w_v = np.array([[0,0,0],[0,0,0],[0,0,0],])

l = 1
o = np.array([[1,1,1],[0,2,2],[0,1,1],])

w = (w_v + l*o*w_e) / (1+l*o)

print(o)
print(w_e)
print()
print(o*w_e)
print()
print(w)