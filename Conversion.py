"""
Conversion Module
 
 This module, as the name indicates, cares about converting objects into another objects. Below we list all funtions presented in this module.
 
 - x2cpd
 
 - cpd2tens
 
 - unfold
 
 - foldback
"""

import numpy as np
import sys
import scipy.io
from numba import jit, njit, prange


@njit(nogil=True, cache=True)
def x2CPD(x, Lambda, X, Y, Z, r, m, n, p):
    """
    Given the point x (the flattened CPD), this function breaks it in parts, to
    form the CPD of S. This program return the following arrays: 
    Lambda = [Lambda_1,...,Lambda_r],
    X = [X_1,...,X_r],
    Y = [Y_1,...,Y_r],
    Z = [Z_1,...,Z_r].
    
    Then we have that T_approx = (X,Y,Z)*diag(Lambda_l), where diag(Lambda_l) is
    a diagonal r x r x r tensor.
    
    Inputs
    ------
    x: float 1-D ndarray
    r: int
    m,n,p: int
        
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)
    """
    
    Lambda = x[0:r]
    X = x[r:r+r*m].reshape(r,m).transpose()
    Y = x[r+r*m:r+r*(m+n)].reshape(r,n).transpose()
    Z = x[r+r*(m+n):r+r*(m+n+p)].reshape(r,p).transpose()
        
    return Lambda, X, Y, Z


@njit(nogil=True, parallel=True)
def CPD2tens(T_aux, Lambda, X, Y, Z, r):
    """
    Converts the arrays Lambda, X, Y, Z to tensor in coordinate format.

    Inputs
    ------
    T_aux: float 3-D ndarray
        This array will receive the coordinates of the approximated tensor.
    We define it outside of this function because this function is called
    several times, and would be too much time costly to create a new tensor
    for every call.
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)

    Outpus
    ------
    T_aux: float 3-D ndarray
    """

    # Obtain the dimensions of the tensor.
    m = X.shape[0]
    n = Y.shape[0]
    p = Z.shape[0]
    
    s = 0.0
    
    for i in prange(0,m):
        for j in range(0,n):
            for k in range(0,p):
                s = 0.0
                for l in range(0,r):
                    s += Lambda[l]*X[i,l]*Y[j,l]*Z[k,l]
                T_aux[i,j,k] = s
    return T_aux


def unfold(T, m, n, p, mode):
    """
    Every tensor T of order 3 has 3 unfoldings, one for each "direction".
    It is commom to denote the unfoldings by T_(1), T_(2), T(3). These 
    unfoldings can be viewed as special kind of transformations of T. 
    They are important for computing the HOSVD of T.
    
    Inputs
    ------
    T: float 3-D ndarray
    m, n, p: int
    mode: 1,2,3
        mode == 1 commands the function to construc the unfolding-1 of T.
    Similarly we can have mode == 2 or mode == 3.
    """
    
    # Test for consistency.
    if (mode != 1) and (mode != 2) and (mode != 3):
        sys.exit('Invalid mode value.')
    
    if mode == 1:
        # Construct mode-1 fibers.
        fibers1 = []
        for k in range(0,p):
            for j in range(0,n):
                fibers1.append(T[:,j,k].reshape(m,1))

        # Construct unfolding-1.
        T1 = np.hstack(fibers1)
    
        return T1
    
    if mode == 2:
        # Construct mode-2 fibers.
        fibers2 = []
        for k in range(0,p):
            for i in range(0,m):
                fibers2.append(T[i,:,k].reshape(n,1))

        # Construct unfolding-2.
        T2 = np.hstack(fibers2)
    
        return T2
    
    if mode == 3:
        # Construct mode-3 fibers.
        fibers3 = []
        for j in range(0,n):
            for i in range(0,m):
                fibers3.append(T[i,j,:].reshape(p,1))

        # Construct unfolding-3.
        T3 = np.hstack(fibers3)
    
        return T3


def foldback(A, m, n, p, mode):
    """ 
    Given an unfolding A of T and the mode l, we construct the tensor 
    T such that T_(l) = A.
    
    Inputs
    ------
    A: float 2-D ndarray
        There are three possibilites which must be respected:
        1) A has shape (m,np) and we have mode == 1.
        2) A has shape (n,mp) and we have mode == 2.
        3) A has shape (p,mn) and we have mode == 3.
    m, n, p: int
    mode: 1,2,3
    
    """
    
    # Test for consistency.
    mA, nA, pA = A.shape
    
    if ((mA, nA*pA) != (m,n*p)) or ((nA, mA*pA) != (n,m*p)) or ((pA, mA*nA) != (p,m*n)):
        sys.exit('Invalid dimensions given.')
        
    elif ((mA, nA*pA) == (m,n*p)) and (mode != 1):
        sys.exit('Invalid mode value.')
            
    elif ((nA, mA*pA) == (n,m*p)) and (mode != 2):
        sys.exit('Invalid mode value.')
            
    elif ((pA, mA*nA) == (p,m*n)) and (mode != 3):
        sys.exit('Invalid mode value.')
    
    T = np.zeros((m,n,p))
    
    if mode == 1:
        s = 0
        for k in range(0,p):
            for j in range(0,n): 
                T[:,j,k] = A[:,s]
                s += 1
                
    if mode == 2:
        s = 0
        for k in range(0,p):
            for i in range(0,m): 
                T[i,:,k] = A[:,s]
                s += 1
                
    if mode == 3:
        s = 0
        for j in range(0,n):
            for i in range(0,m): 
                T[i,j,:] = A[:,s]
                s += 1
     
    return T