import cv2 as cv
import numpy as np


def convolution(matrix: np.array, kernel: np.array):
    m = matrix.shape[0]
    n = matrix.shape[1]
    x = kernel.shape[0]
    y = kernel.shape[1]
    res = np.zeros([m-x+1, n-y+1])
    for i in range(m - x + 1):
        for j in range(n - y + 1):
            res[i, j] = np.sum(np.multiply(matrix[i:i+x, j:j+y], kernel))
    return res


img = cv.imread("sheep.png")
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('gray', gray)
gray = gray.astype(np.float)
# gray = np.array([[255,255,255,255],[255,255,255,255],[255,255,255,255],[255,255,255,255]])
prewitt = np.array([[0,1,0],[1,-4,1],[0,1,0]])
img2 = convolution(gray, prewitt)
min_v=np.min(img2)
max_v=np.max(img2)
img2=(img2+abs(min_v))*255/(abs(min_v)+max_v)+1
print(min_v,max_v)
print(img2.shape)
cv.imshow('', img2.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()

