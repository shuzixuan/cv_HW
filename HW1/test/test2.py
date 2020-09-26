import numpy as np


A = np.array([[2,1,-2,5],[3,0,1,5],[1,1,-1,5]])
b = np.transpose(np.array([[-3,5,-2]]))
print(A.shape, b.shape)
x = np.linalg.solve(A,b)
print(x)