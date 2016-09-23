from scipy import misc
import sys
import numpy as np

if __name__ == "__main__" :
    lena = misc.imread(sys.argv[1])
    h, w = lena.shape
    rotated = np.zeros([h,w], dtype='uint8')
    for i in range(h):
        for j in range(w):
            rotated[i][j] = lena[h-1-i][w-1-j]

    misc.imsave('ans2.png', rotated)

