import cv2 as cv
import time
import cupy as cp

def convolution(matrix, kernel):
    m = matrix.shape[0]
    n = matrix.shape[1]
    x = kernel.shape[0]
    y = kernel.shape[1]
    res = cp.zeros((m-x+1, n-y+1))
    for i in range(m - x + 1):
        for j in range(n - y + 1):
            res[i, j] = cp.sum(cp.multiply(matrix[i:i+x, j:j+y], kernel))
    return res


def laplace(m: cp.array):
    l = cp.array([[0,1,0],[1,-4,1],[0,1,0]])
    res = convolution(m, l)
    min_v = cp.min(res)
    max_v = cp.max(res)
    res = (res + abs(min_v)) * 255 / (abs(min_v) + max_v) + 1
    return res


def compute_A():
    a = cp.zeros((262*321, 262*321), cp.int8)
    for i in range(0, 262):
        for j in range(0, 321):
            x = cp.zeros((262, 321))
            if i == 0 or j == 0 or i == 261 or j == 320:
                x[i, j] = 1
            else:
                x[i, j] = -4
                x[i-1, j], x[i, j-1], x[i+1, j], x[i, j+1] = 1, 1, 1, 1
            a[i+j-2, :] = x.flatten()
    return a


def poisson(s, g, A):
    background = g[398:662,898:1221]
    l_background = laplace(background)
    l_sheep = laplace(s)
    replace = l_background.copy()
    replace[2:260,2:319] = l_sheep
    b = cp.transpose(cp.array([replace.flatten()]))
    print(1)
    # print(A.shape, b.shape)
    res = cp.linalg.solve(A, b)
    return res


# load images
sheep = cv.imread("sheep.png")
sheep_R = sheep[:,:,0]
sheep_G = sheep[:,:,1]
sheep_B = sheep[:,:,2]
grass = cv.imread("grass.jpg")
grass_R = grass[:,:,0]
grass_G = grass[:,:,1]
grass_B = grass[:,:,2]
A = compute_A()

# target for sheep is grass[400:660,900:1219]
start = time.time()
R = poisson(sheep_R, grass_R, A)
print("R finished")
end = time.time()
print(end-start)
