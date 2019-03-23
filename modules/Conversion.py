"""
Conversion Module
 
 This module, as the name indicates, cares about converting objects into another objects. Below we list all funtions presented in this module.
 
 - x2cpd
 
 - cpd2tens
 
 - tens_entries
 
 - unfold
 
 - foldback

"""

import numpy as np
import sys
import scipy.io
from numba import jit, njit, prange
import Auxiliar as aux


@njit(nogil=True)
def x2cpd(x, X, Y, Z, m, n, p, r):   
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
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    m, n, p: int
    r: int
        
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    """ 
    s = 0
    for l in range(r):
        for i in range(m):
            X[i,l] = x[s]
            s += 1
            
    for l in range(r):
        for j in range(n):
            Y[j,l] = x[s]
            s += 1
            
    for l in range(r):
        for k in range(p):
            Z[k,l] = x[s]
            s += 1
            
    X, Y, Z = aux.equalize(X, Y, Z, r)
          
    return X, Y, Z


@njit(nogil=True)
def cpd2tens(T_aux, X, Y, Z, temp, m, n, p, r):
    """
    Converts the arrays Lambda, X, Y, Z to tensor in coordinate format.

    Inputs
    ------
    T_aux: float 3-D ndarray
        This array will receive the coordinates of the approximated tensor.
    We define it outside of this function because this function is called
    several times, and would be too much time costly to create a new tensor
    for every call.
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    m, n, p: int
    r: int

    Outpus
    ------
    T_aux: float 3-D ndarray
        Tensor (X,Y,Z) in coordinate format. 
    """
    
    for k in range(p):
        for l in range(r):
            temp[:,l] = Z[k,l]*X[:,l]
        T_aux[:,:,k] = np.dot(temp, Y.T)
        
    return T_aux


@njit(nogil=True)
def unfold(T, Tl, m, n, p, mode):   
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

    Outputs
    -------
    This function returns a matrix of one of the following shapes: (m, n*p), 
    (n, m*p) or (p, m*n). Each one is a possible unfolding of T.
    """
 
    if mode == 1:
        for i in range(0,m):
            temp = T[i,:,:].T
            Tl[i, :] = temp.ravel()
    
    if mode == 2:
        for j in range(0,n):
            temp = T[:,j,:].T
            Tl[j, :] = temp.ravel()
    
    if mode == 3:
        for k in range(0,p):
            temp = T[:,:,k].T
            Tl[k,:] = temp.ravel()

    return Tl


@njit(nogil=True)
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

    Outputs
    -------
    T: float 3-D ndarray
        The reconstructed tensor.
    """
    
    T = np.zeros((m,n,p), dtype = np.float64)
    At = A.transpose()
    
    if mode == 1:
        s = 0
        for k in range(0,p):
            for j in range(0,n): 
                T[:,j,k] = At[s,:]
                s += 1
                
    if mode == 2:
        s = 0
        for k in range(0,p):
            for i in range(0,m): 
                T[i,:,k] = At[s,:]
                s += 1
                
    if mode == 3:
        s = 0
        for j in range(0,n):
            for i in range(0,m): 
                T[i,j,:] = At[s,:]
                s += 1
     
    return T
