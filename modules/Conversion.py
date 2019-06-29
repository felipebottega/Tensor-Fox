"""
 Conversion Module
 =================
 This module cares about converting objects into another objects.
"""

# Python modules
from numpy import empty, zeros, prod, int64, dot, log, exp
from numpy.linalg import norm
from numpy.random import randn
from numba import njit, prange

# Tensor Fox modules
import Critical as crt
import MultilinearAlgebra as mlinalg


@njit(nogil=True)
def x2cpd(x, X, Y, Z, m, n, p, R):
    """
    Given the point x (the flattened CPD), this function breaks it in parts, to form the CPD. Then this function return
    the following arrays:
    X = [X_1,...,X_R],
    Y = [Y_1,...,Y_R],
    Z = [Z_1,...,Z_R].
    Then we have that T_approx = (X,Y,Z)*I, where I is the diagonal R x R x R tensor.
    
    Inputs
    ------
    x: float 1-D ndarray
    X: float 2-D ndarray of shape (m, R)
    Y: float 2-D ndarray of shape (n, R)
    Z: float 2-D ndarray of shape (p, R)
    m, n, p: int
    R: int
        
    Outputs
    -------
    X: float 2-D ndarray of shape (m, R)
    Y: float 2-D ndarray of shape (n, R)
    Z: float 2-D ndarray of shape (p, R)
    """ 

    s = 0
    for r in range(R):
        X[:, r] = x[s:s+m]
        s = s+m
            
    for r in range(R):
        Y[:, r] = x[s:s+n]
        s = s+n
            
    for r in range(R):
        Z[:, r] = x[s:s+p]
        s = s+p
            
    X, Y, Z = equalize(X, Y, Z, R)
          
    return X, Y, Z


def cpd2tens(T_approx, factors, dims):
    """
    Converts the factor matrices to tensor in coordinate format using a Khatri-Rao product formula.

    Inputs
    ------
    factors: list of L floats 2-D ndarray of shape (dims[i], R) each
    dims: tuple of L ints

    Outputs
    ------
    T_approx: float L-D ndarray
        Tensor (factors[0],...,factors[L-1])*I in coordinate format. 
    """

    L = len(dims)
    M = factors[1]
   
    for l in range(2, L):
        N = empty((M.shape[0]*factors[l].shape[0], M.shape[1]))
        M = mlinalg.khatri_rao(factors[l], M, N)

    T1_approx = dot(factors[0], M.T)
    T_approx = foldback(T_approx, T1_approx, 1, dims)
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


def foldback(T, Tl, mode, dims):
    """
    Computes the tensor with dimension dims given an unfolding with its mode. Attention: dims are the dimensions of the 
    output tensor, not the input.
    """
 
    L = len(dims)
    func_name = "foldback" + str(mode) + "_order" + str(L)
    T = getattr(crt, func_name)(T, Tl, dims)
    
    return T


def normalize(factors):
    """ 
    Normalize the columns of the factors to have unit column norm and scale Lambda accordingly. This function returns 
    Lambda and the normalized factors. 
    """

    R = factors[0].shape[1]
    Lambda = zeros(R)
    L = len(factors)
    
    for r in range(R):
        norms = zeros(L)
        for l in range(L):
            W = factors[l]
            # Save norm of the l-th column of the ll factor and normalize the current factor.
            norms[l] = norm(W[:, r])
            W[:, r] = W[:, r]/norms[l]
            # Update factors accordingly.
            factors[l] = W

        Lambda[r] = prod(norms)
        
    return Lambda, factors


def denormalize(Lambda, X, Y, Z):
    """
    By undoing the normalization of the factors this function makes it unnecessary the use of the diagonal tensor
    Lambda. This is useful when one wants the CPD described only by the triplet (X, Y, Z).
    """

    R = Lambda.size
    X_new = zeros(X.shape)
    Y_new = zeros(Y.shape)
    Z_new = zeros(Z.shape)
    for r in range(R):
        if Lambda[r] >= 0:
            a = Lambda[r]**(1/3)
            X_new[:, r] = a*X[:, r]
            Y_new[:, r] = a*Y[:, r]
            Z_new[:, r] = a*Z[:, r]
        else:
            a = (-Lambda[r])**(1/3)
            X_new[:, r] = -a*X[:, r]
            Y_new[:, r] = a*Y[:, r]
            Z_new[:, r] = a*Z[:, r]
            
    return X_new, Y_new, Z_new


@njit(nogil=True)
def equalize(X, Y, Z, R):
    """ 
    After a Gauss-Newton iteration we have an approximated CPD with factors X_r ⊗ Y_r ⊗ Z_r. They may have very
    different magnitudes and this can have effect on the convergence rate. To improve this we try to equalize their
    magnitudes by introducing scalars a, b, c such that X_r ⊗ Y_r ⊗ Z_r = (a*X_r) ⊗ (b*Y_r) ⊗ (c*Z_r) and
    |a*X_r| = |b*Y_r| = |c*Z_r|. Notice that we must have a*b*c = 1.
    
    To find good values for a, b, c, we can search for critical points of the function 
    f(a,b,c) = (|a*X_r|-|b*Y_r|)^2 + (|a*X_r|-|c*Z_r|)^2 + (|b*Y_r|-|c*Z_r|)^2.
    Using Lagrange multipliers we find the solution 
        a = (|X_r|*|Y_r|*|Z_r|)^(1/3)/|X_r|,
        b = (|X_r|*|Y_r|*|Z_r|)^(1/3)/|Y_r|,
        c = (|X_r|*|Y_r|*|Z_r|)^(1/3)/|Z_r|.
    We can see that this solution satisfy the conditions mentioned.
    """
    
    for r in range(R):
        X_nr = norm(X[:, r])
        Y_nr = norm(Y[:, r])
        Z_nr = norm(Z[:, r])
        if (X_nr != 0) and (Y_nr != 0) and (Z_nr != 0):
            numerator = (X_nr*Y_nr*Z_nr)**(1/3)
            X[:, r] = (numerator/X_nr)*X[:, r]
            Y[:, r] = (numerator/Y_nr)*Y[:, r]
            Z[:, r] = (numerator/Z_nr)*Z[:, r]
            
    return X, Y, Z


@njit(nogil=True)
def transform(X, Y, Z, a, b, factor, symm, c):
    """
    Depending on the choice of the user, this function can project the entries of X, Y, Z in a given interval (this is 
    very useful with we have constraints at out disposal), it can make the corresponding tensor symmetric or
    non-negative. It is advisable to transform the tensor so that its entries have mean zero and variance 1, this way
    choosing low=-1 and upp=1 works the best way possible. We also remark that it is always better to choose low and upp
    such that low = -upp.
    
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
        B = log( (b-a)/eps - 1 )/( factor*(b-a)/2 - eps )
        A = -B*(a+b)/2
        X = a + (b-a) * 1/( 1 + exp(-A-B*X) )
        Y = a + (b-a) * 1/( 1 + exp(-A-B*Y) )
        Z = a + (b-a) * 1/( 1 + exp(-A-B*Z) )
        
    if symm:
        X = (X+Y+Z)/3
        Y = X
        Z = X

    if c > 0:
        X = c * (1/norm(X)) * X
        Y = c * (1/norm(Y)) * Y
        Z = c * (1/norm(Z)) * Z
    
    return X, Y, Z


@njit(nogil=True, parallel=True)
def vec(M, Bv, num_rows, R):
    """ 
    Take a matrix M with shape (num_rows, R) and stack vertically its columns to form the matrix Bv = vec(M) with shape
    (num_rows*R,).
    """
    
    for r in prange(R):
        Bv[r*num_rows:(r+1)*num_rows] = M[:, r]
        
    return Bv


@njit(nogil=True, parallel=True)
def vect(M, Bv, num_cols, R):
    """ 
    Take a matrix M with shape (R, num_cols) and stack vertically its rows to form the matrix Bv = vec(M) with shape
    (num_cols*R,).
    """
    
    for r in prange(R):
        Bv[r*num_cols:(r+1)*num_cols] = M[r, :]
        
    return Bv


def inflate(T, R, dims):
    """
    Let T be a tensor of shape dims. If rank > dims[l], this function increases T dimensions such that each new
    dimension satisfies new_dims[l] = R. The new entries are all random number very close to zero.
    """

    L = len(dims)
    new_dims = zeros(L, int64)
    slices = []

    for l in range(L):
        slices.append(slice(dims[l]))
        if dims[l] >= R:
            new_dims[l] = dims[l]
        else:
            new_dims[l] = R
            
    new_T = 1e-12*randn(*new_dims)
    new_T[tuple(slices)] = T
    
    return new_T


def deflate(factors, orig_dims, inflate_status):
    """
    If the tensor was inflated, this function restores the factors to their original shape by truncating them.
    """

    if not inflate_status:
        return factors

    else:
        L = len(orig_dims) 
        factors = [ factors[l][:orig_dims[l], :] for l in range(L) ]
        return factors
