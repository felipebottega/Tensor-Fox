"""
 Multilinear Algebra Module
 ==========================
 All relevant functions of the classical multilinear algebra are described and implemented in this module.
"""

# Python modules
import numpy as np
from numpy import dot, zeros, empty, float64
from numpy.linalg import norm
from numba import njit, prange

# Tensor Fox modules
import Auxiliar as aux
import Compression as cmpr
import Conversion as cnv


def multilin_mult_cpd(U, W, dims):
    """    
    Performs the multilinear multiplication (U[0],...,U[L-1])*(W[0], ..., W[L-1])*I = (U[0]*W[0],...,U[L-1]*W[L-1])*I, 
    where I.shape = dims = (W[0].shape[1],...,W[L-1].shape[1]) are the size of the columns of the W's. 
    """

    L = len(dims)
    # dims_out are the dimensions of the output tensor.
    dims_out = [] 
    W_new = []
    
    for l in range(L):
        W_new.append( dot(U[l], W[l]) )
        dims_out.append(W_new[l].shape[0])
    
    S = zeros(dims_out)
    S = cnv.cpd2tens(S, W_new, dims_out)
    return S


def multilin_mult(U, T1, dims):
    """    
    Performs the multilinear multiplication (U[0],...,U[L-1])*T, where dims = T.shape. We need the first unfolding T1 of 
    T to start the computations.
    """

    L = len(dims)
    # dims_out are the dimensions of the output tensor.
    dims_out = list(dims)
    
    unfolding1 = T1    
    for l in range(L):
        unfolding2 = dot(U[l], unfolding1)
        # Update the current dimension of dims_out.
        dims_out[l] = U[l].shape[0]
        S = cnv.foldback(unfolding2, l+1, dims_out)
        if l < L-1:            
            unfolding1 = cnv.unfold(S, l+2, S.shape)
        else:
            return S
        

def multirank_approx(T, r1, r2, r3, options):
    """
    This function computes an approximation of T with multilinear rank = (r1,r2,r3). Truncation the central tensor of
    the MLSVD doesn't gives the best low multirank approximation, but gives very good approximations.
    
    Inputs
    ------
    T: 3-D float ndarray
    r1, r2, r3: int
        (r1,r2,r3) is the desired low multilinear rank.
        
    Outputs
    -------
    T_approx: 3-D float ndarray
        The approximating tensor with multilinear rank = (r1,r2,r3).
    """
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = norm(T)
    
    # Compute the MLSVD of T.
    options = aux.complete_options(options)
    r = min(m, n, p)
    S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = cmpr.mlsvd(T, Tsize, r, options)
    U1 = U1[:, 0:r1]
    U2 = U2[:, 0:r2]
    U3 = U3[:, 0:r3]
    
    # Truncate S to a smaller shape (r1,r2,r3) and construct the tensor T_approx = (U1,U2,U3)*S.
    S = S[0:r1, 0:r2, 0:r3]
    U = [U1, U2, U3]
    dims = (r1, r2, r3)   
    S1 = cnv.unfold(S, 1, dims)         
    T_approx = multilin_mult(U, S, S1)
    
    return T_approx


@njit(nogil=True)
def kronecker(A, B):
    """
    Computes the Kronecker product between A and B. We must have M.shape = (a1*b1, a2*b2), where A.shape = (a1, a2) and 
    B.shape = (b1, b2). 
    """

    a1, a2 = A.shape
    b1, b2 = B.shape
    M = empty((a1*b1, a2*b2), dtype=float64)
    
    for i in range(0, a1):
        for j in range(0, a2):
            M[i*b1:(i+1)*b1, j*b2:(j+1)*b2] = A[i, j]*B

    return M 


@njit(nogil=True, parallel=True)
def khatri_rao(A, B, M):
    """
    Computes the Khatri-Rao product between A and B. We must have M.shape = (a1*b1, a2), where A.shape = (a1, a2) and 
    B.shape = (b1, b2), with a2 == b2. 
    """

    a1, a2 = A.shape
    b1, b2 = B.shape
    
    for i in prange(0, a1):
        M[i*b1:(i+1)*b1, :] = khatri_rao_inner_computations(A, B, M, i, b1, b2)

    return M 


@njit(nogil=True)
def khatri_rao_inner_computations(A, B, M, i, b1, b2):
    """
    Computes A[i, :]*B.
    """

    for k in range(b1):
        for j in range(b2):
            M[i*b1 + k, j] = A[i, j]*B[k, j]

    return M[i*b1:(i+1)*b1, :]


@njit(nogil=True, parallel=True)
def hadamard(A, B, M, r):
    """
    Computes M = A * B, where * is the Hadamard product. Since all Hadamard products in this context are between r x r 
    matrices, we assume this without verifications.
    """
    
    for i in prange(r):
        M[i, :] = A[i, :]*B[i, :]

    return M 
