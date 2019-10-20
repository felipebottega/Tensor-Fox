import numpy as np


def Mntensor(n):
    """This functions returns the tensor of the n x n matrix multiplication and a possible rank."""

    Mn = np.zeros((n**2,n**2,n**2))

    for i in range(0,n):
        for j in range(0,n):
            for k in range(0,n):
                Mn[i*n+j,j*n+k,i*n+k] = 1
    r = int(np.ceil( n**(np.log(7)/np.log(2)) ))

    return Mn, r
