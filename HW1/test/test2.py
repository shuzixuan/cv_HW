import numpy as np
import cupy as cp
import time

s = time.time()
x_gpu = cp.ones((1000,1000))
e = time.time()
print(e - s)