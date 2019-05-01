"""
 Conversion Module
 =================
 This module cares about converting objects into another objects.
"""

# Python modules
import numpy as np
from numpy import empty, zeros, prod, float64, dot, log, exp
from numpy.linalg import norm
from numba import njit, prange

# Tensor Fox modules
import Conversion as cnv
import Critical as crt
import MultilinearAlgebra as mlinalg


@njit(nogil=True)
def x2cpd(x, X, Y, Z, m, n, p, r):   
    """
    Given the point x (the flattened CPD), this function breaks it in parts, to form the CPD of S. This program return the 
    following arrays: 
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
            
    X, Y, Z = cnv.equalize(X, Y, Z, r)
          
    return X, Y, Z


def cpd2tens(T_approx, factors, dims):
    """
    Converts the factor matrices to tensor in coordinate format using a Khatri-Rao product formula.

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
    T1_approx = empty((dims[0], prod(dims[1:])), dtype = float64)
    M = factors[1]
   
    for l in range(2,L):
        a1, a2 = factors[l].shape
        b1, b2 = M.shape
        M = mlinalg.khatri_rao(factors[l], M)        

    T1_approx = dot(factors[0], M.T)
    T_approx = foldback(T1_approx, 1, dims)
    return T_approx


def unfold(T, mode, dims):
    """
    Computes any unfolding of a tensor up to order L = 12. 
    """
 
    L = len(dims)
    Tl = empty((dims[mode-1], prod(dims)//dims[mode-1]), order='F')    
    func_name = "unfold" + str(mode) + "_order" + str(L)
    Tl = getattr(crt, func_name)(T, Tl, dims)
    return Tl


def foldback(Tl, mode, dims):
    """
    Computes the tensor with dimension dims given an unfolding with its mode. Attention: dims are the dimensions of the 
    output tensor, not the input.
    """
 
    L = len(dims)
    T = empty(dims, order='F')
    func_name = "foldback" + str(mode) + "_order" + str(L)
    T = getattr(crt, func_name)(T, Tl, dims)
    
    return T


def normalize(factors):
    """ 
    Normalize the columns of the factors to have unit column norm and scale Lambda accordingly. This function returns 
    Lambda and the normalized factors. 
    """

    r = factors[0].shape[1]
    Lambda = zeros(r)
    L = len(factors)
    
    for l in range(0,r):
        norms = zeros(L)
        for ll in range(L):
            W = factors[ll]
            # Save norm of the l-th column of the ll factor and normalize the current factor.
            norms[ll] = norm(W[:,l]) 
            W[:,l] = W[:,l]/norms[ll]
            # Update factors accordingly.
            factors[ll] = W 

        Lambda[l] = prod(norms)
        
    return Lambda, factors


def denormalize(Lambda, X, Y, Z):
    """
    By undoing the normalization of the factors this function makes it unnecessary the use of the diagonal tensor Lambda. 
    This is useful when one wants the CPD described only by the triplet (X, Y, Z).
    """

    R = Lambda.size
    X_new = zeros(X.shape)
    Y_new = zeros(Y.shape)
    Z_new = zeros(Z.shape)
    for r in range(R):
        if Lambda[r] >= 0:
            a = Lambda[r]**(1/3)
            X_new[:,r] = a*X[:,r]
            Y_new[:,r] = a*Y[:,r]
            Z_new[:,r] = a*Z[:,r]
        else:
            a = (-Lambda[r])**(1/3)
            X_new[:,r] = -a*X[:,r]
            Y_new[:,r] = a*Y[:,r]
            Z_new[:,r] = a*Z[:,r]
            
    return X_new, Y_new, Z_new


@njit(nogil=True)
def equalize(X, Y, Z, r):
    """ 
    After a Gauss-Newton iteration we have an approximated CPD with factors X_l ⊗ Y_l ⊗ Z_l. They may have very differen 
    magnitudes and this can have effect on the convergence rate. To improve this we try to equalize their magnitudes by 
    introducing scalars a, b, c such that X_l ⊗ Y_l ⊗ Z_l = (a*X_l) ⊗ (b*Y_l) ⊗ (c*Z_l) and |a*X_l| = |b*Y_l| = |c*Z_l|. 
    Notice that we must have a*b*c = 1.
    
    To find good values for a, b, c, we can search for critical points of the function 
    f(a,b,c) = (|a*X_l|-|b*Y_l|)^2 + (|a*X_l|-|c*Z_l|)^2 + (|b*Y_l|-|c*Z_l|)^2.
    Using Lagrange multipliers we find the solution 
        a = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|X_l|,
        b = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|Y_l|,
        c = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|Z_l|.    
    We can see that this solution satisfy the conditions mentioned.
    """
    
    for l in range(0, r):
        X_nr = norm(X[:,l])
        Y_nr = norm(Y[:,l])
        Z_nr = norm(Z[:,l])
        if (X_nr != 0) and (Y_nr != 0) and (Z_nr != 0) :
            numerator = (X_nr*Y_nr*Z_nr)**(1/3)
            X[:,l] = (numerator/X_nr)*X[:,l]
            Y[:,l] = (numerator/Y_nr)*Y[:,l]
            Z[:,l] = (numerator/Z_nr)*Z[:,l] 
            
    return X, Y, Z


@njit(nogil=True)
def transform(X, Y, Z, m, n, p, r, a, b, factor, symm):
    """
    Depending on the choice of the user, this function can project the entries of X, Y, Z in a given interval (this is 
    very useful with we have constraints at out disposal), it can make the corresponding tensor symmetric or non-negative.
    It is advisable to transform the tensor so that its entries have mean zero and variance 1, this way choosing low=-1 
    and upp=1 works the best way possible. We also remark that it is always better to choose low and upp such that 
    low = -upp.
    
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
        B =   log( (b-a)/eps - 1 )/( factor*(b-a)/2 - eps )
        A = -B*(a+b)/2
        X = a + (b-a) * 1/( 1 + exp(-A-B*X) )
        Y = a + (b-a) * 1/( 1 + exp(-A-B*Y) )
        Z = a + (b-a) * 1/( 1 + exp(-A-B*Z) )
        
    if symm:
        X = (X+Y+Z)/3
        Y = X
        Z = X
    
    return X, Y, Z


@njit(nogil=True, parallel=True)
def vec(M, Bv, num_rows, r):
    """ 
    Take a matrix M with shape (num_rows, r) and stack vertically its columns to form the matrix Bv = vec(M) with shape 
    (num_rows*r,).
    """
    
    for j in prange(0, r):
        Bv[j*num_rows : (j+1)*num_rows] = M[:,j]
        
    return Bv


@njit(nogil=True, parallel=True)
def vect(M, Bv, num_cols, r):
    """ 
    Take a matrix M with shape (r, num_cols) and stack vertically its rows to form the matrix Bv = vec(M) with shape 
    (num_cols*r,).
    """
    
    for i in prange(0, r):
        Bv[i*num_cols : (i+1)*num_cols] = M[i,:]
        
    return Bv
