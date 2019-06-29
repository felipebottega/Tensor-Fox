"""
 Multilinear Algebra Module
 ==========================
 All relevant functions of the classical multilinear algebra are described and implemented in this module.
"""

# Python modules
import numpy as np
from numpy import dot, zeros, empty, float64, array, sort, ceil, prod
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
        S = empty(dims_out)
        S = cnv.foldback(S, unfolding2, l+1, dims_out)
        if l < L-1:            
            unfolding1 = cnv.unfold(S, l+2, S.shape)
        else:
            return S
        

def multirank_approx(T, multi_rank, options):
    """
    This function computes an approximation of T with multilinear rank = multi_rank. Truncation the central tensor of
    the MLSVD doesn't gives the best low multirank approximation, but gives very good approximations.
    
    Inputs
    ------
    T: L-D float ndarray
    multi_rank: list of int
        The desired low multilinear rank.
        
    Outputs
    -------
    T_approx: L-D float ndarray
        The approximating tensor with multilinear rank = multi_rank.
    """
    
    # Compute dimensions and norm of T.
    dims = T.shape
    sorted_dims = sort(array(dims))
    L = len(dims)
    Tsize = norm(T)
    
    # Compute truncated MLSVD of T.
    options = aux.complete_options(options)
    options.display = 0
    options.trunc_dims = multi_rank
    R_gen = int(ceil( prod(sorted_dims)/(np.sum(sorted_dims) - L + 1) ))
    S, U, UT, sigmas = cmpr.mlsvd(T, Tsize, R_gen, options)

    # Construct the corresponding tensor T_approx.
    S1 = cnv.unfold(S, 1, multi_rank)
    T_approx = multilin_mult(U, S1, multi_rank)
    
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
def hadamard(A, B, M, R):
    """
    Computes M = A * B, where * is the Hadamard product. Since all Hadamard products in this context are between R x R
    matrices, we assume this without verifications.
    """
    
    for r in prange(R):
        M[r, :] = A[r, :]*B[r, :]

    return M 
