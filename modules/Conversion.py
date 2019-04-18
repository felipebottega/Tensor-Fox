"""
 Conversion Module
 
 This module, as the name indicates, cares about converting objects into another objects.

"""

import numpy as np
import itertools
import sys
import scipy.io
from numba import jit, njit, prange
import Auxiliar as aux
import Critical as crt


@njit(nogil=True)
def x2cpd(x, X, Y, Z, m, n, p, r):   
    """
    Given the point x (the flattened CPD), this function breaks it in parts, to
    form the CPD of S. This program return the following arrays: 
    X = [X_1,...,X_r],
    Y = [Y_1,...,Y_r],
    Z = [Z_1,...,Z_r].
    
    Then we have that T_approx = (X,Y,Z)*I, where I is the diagonal r x r x r tensor.
    
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
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    """ 

    s = 0
    for l in range(r):
        X[:,l] = x[s:s+m]
        s = s+m
            
    for l in range(r):
        Y[:,l] = x[s:s+n]
        s = s+n
            
    for l in range(r):
        Z[:,l] = x[s:s+p]
        s = s+p
            
    X, Y, Z = aux.equalize(X, Y, Z, r)
          
    return X, Y, Z


def cpd2tens(factors, dims):
    """
    Converts the factor matrices to tensor in coordinate format using
    a Khatri-Rao product formula.

    Inputs
    ------
    factors: list of L floats 2-D ndarray of shape (dims[i], r) each
    dims: tuple of L ints

    Outputs
    ------
    T_approx: float L-D ndarray
        Tensor (factors[0],...,factors[L-1])*I in coordinate format. 
    """

    L = len(dims)
    prod_dims = np.prod(np.array(dims[1:]))
    T1_approx = np.zeros((dims[0], prod_dims))
    M = factors[1]
   
    for l in range(2,L):
        a1, a2 = factors[l].shape
        b1, b2 = M.shape
        N = np.zeros((a1*b1, a2))
        M = crt.khatri_rao(factors[l], M, N)        

    T1_approx = np.dot(factors[0], M.T)
    T_approx = T1_approx.reshape(dims, order='F')
    return T_approx


@njit(nogil=True)
def unfold(T, Tl, m, n, p, mode):   
    """
    Every tensor T of order 3 has 3 unfoldings, one for each "direction".
    It is commom to denote the unfoldings by T_(1), T_(2), T(3). These 
    unfoldings can be viewed as special kind of transformations of T. 
    They are important for computing the MLSVD of T.
    
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


def high_unfold1_generic(T, T1, dims, l, idx, s):
    """
    Computes the first unfolding of a tensor with any high order. The first
    call must initiate with l=L (the total number of modes), idx=() and s=0.
    This function works with any order L but it is slow.
    """

    l -= 1
    
    if l < 0:
        s = 0
    
    elif l > 0:
        for i in range(dims[l]):
            idx_temp = (i,) + idx
            T1, s = high_unfold1_generic(T, T1, dims, l, idx_temp, s)
                
    else:
        for i in range(dims[l]):
            idx_temp = (i,) + idx
            T1[i,s] = T[idx_temp]
        s += 1
        
    return T1, s


def high_unfold1(T, dims):
    """
    Computes the first unfolding of a tensor up to order L=12. 
    """
 
    L = len(dims)
    T1 = np.zeros((dims[0], np.prod(dims[1:])))

    if L == 4:
        return crt.unfold1_order4(T, T1, dims)
    if L == 5:
        return crt.unfold1_order5(T, T1, dims)
    if L == 6:
        return crt.unfold1_order6(T, T1, dims)
    if L == 7:
        return crt.unfold1_order7(T, T1, dims)
    if L == 8:
        return crt.unfold1_order8(T, T1, dims)
    if L == 9:
        return crt.unfold1_order9(T, T1, dims)
    if L == 10:
        return crt.unfold1_order10(T, T1, dims)
    if L == 11:
        return crt.unfold1_order11(T, T1, dims)
    if L == 12:
        return crt.unfold1_order12(T, T1, dims)

    return T1


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


@njit(nogil=True)
def transform(X, Y, Z, m, n, p, r, a, b, factor, symm):
    """
    Depending on the choice of the user, this function can project the entries of X, Y, Z
    in a given interval (this is very useful with we have constraints at out disposal), it 
    can make the corresponding tensor symmetric or non-negative.
    It is advisable to transform the tensor so that its entries have mean zero and variance
    1, this way choosing low=-1 and upp=1 works the best way possible. We also remark that
    it is always better to choose low and upp such that low = -upp.
    
    Inputs
    ------
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    m, n, p: int
    r: int
    low, upp: float
    symm: bool
    fix: float 2-D ndarray of shape (m, r)
        
    Outputs
    -------
    X: float 2-D ndarray of shape (m, r)
    Y: float 2-D ndarray of shape (n, r)
    Z: float 2-D ndarray of shape (p, r)
    """ 

    if a != 0 and b != 0:
        eps = 0.02
        B =   np.log( (b-a)/eps - 1 )/( factor*(b-a)/2 - eps )
        A = -B*(a+b)/2
        X = a + (b-a) * 1/( 1 + np.exp(-A-B*X) )
        Y = a + (b-a) * 1/( 1 + np.exp(-A-B*Y) )
        Z = a + (b-a) * 1/( 1 + np.exp(-A-B*Z) )
        
    if symm:
        X = (X+Y+Z)/3
        Y = X
        Z = X
    
    return X, Y, Z
