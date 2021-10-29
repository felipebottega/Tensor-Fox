"""
 Conversion Module
 =================
 This module cares about converting objects into another objects.
"""

# Python modules
import numpy as np
from numpy import empty, array, zeros, arange, prod, int64, uint64, dot, sign, float64, nonzero
from numpy.linalg import norm
from numpy.random import randn
from numba import njit
from scipy.sparse import coo_matrix

# Tensor Fox modules
import TensorFox.Critical as crt
import TensorFox.MultilinearAlgebra as mlinalg


def x2cpd(x, factors, eq=True):
    """
    Given the point x (the flattened CPD), this function breaks it in parts to form the factors of the CPD.
    
    Inputs
    ------
    x: float 1-D array
    factors: list of 2-D arrays
        Factors matrices to be updated.
        
    Outputs
    -------
    factors: list of 2-D arrays
    """ 

    R = factors[0].shape[1]
    L = len(factors)

    s = 0
    for l in range(L):
        dim = factors[l].shape[0]
        for r in range(R):
            factors[l][:, r] = x[s: s+dim]
            s += dim
    
    if eq:        
        factors = equalize(factors, R)
          
    return factors


def cpd2tens(factors):
    """
    Converts the factor matrices to tensor in coordinate format using a Khatri-Rao product formula.

    Inputs
    ------
    factors: list of 2-D arrays
        The factor matrices.

    Outputs
    ------
    T_approx: float L-D array
        Tensor (factors[0],...,factors[L-1])*I in coordinate format. 
    """

    L = len(factors)
    dims = [factors[l].shape[0] for l in range(L)]
    T_approx = empty(dims)
    M = factors[1]
   
    for l in range(2, L):
        N = empty((M.shape[0]*factors[l].shape[0], M.shape[1]))
        M = mlinalg.khatri_rao(factors[l], M, N)

    T1_approx = dot(factors[0], M.T)
    T_approx = foldback(T_approx, T1_approx, 1)

    return T_approx


def cpd2unfold1(T1_approx, factors):
    """
    Converts the factor matrices to the first unfolding of the corresponding tensor.

    Inputs
    ------
    factors: list of 2-D arrays
        The factor matrices.

    Outputs
    ------
    T1_approx: float 2-D array
        First unfolding of T_approx, where T_approx is (factors[0],...,factors[L-1])*I in coordinate format.
    """

    L = len(factors)
    M = factors[1]
   
    for l in range(2, L):
        N = empty((M.shape[0]*factors[l].shape[0], M.shape[1]))
        M = mlinalg.khatri_rao(factors[l], M, N)

    dot(factors[0], M.T, out=T1_approx)

    return T1_approx


def sparse2dense(data, idxs, dims):
    """
    Given the variables defining a sparse tensor, this function computes its dense representation.

    Inputs
    ------
    data: float 1-D arrays
        data[i] is the nonzero value of the tensor at index idxs[i, :].
    idxs: int 2-D array
        Let nnz be the number of nonzero entries of the tensor. Then idxs is an array
        of shape (nnz, L) such that idxs[i, :] is the index of the i-th nonzero entry.
    dims: list or tuple
        The dimensions (shape) of the tensor.

    Outputs
    -------
    T_dense: L-D array
        Dense representation of the tensor.
    """

    data = array(data)
    idxs = array(idxs)
    T1 = sparse_unfold(data, idxs, dims, 1)
    T1_dense = T1.toarray()
    T_dense = empty(dims, dtype=float64)
    T_dense = foldback(T_dense, T1_dense, 1)

    return T_dense
    
    
def dense2sparse(T):
    """
    Given a dense tensor, this function computes its sparse representation.
    
    Inputs
    -------
    T_dense: L-D array
        Dense representation of the tensor.

    Outputs
    ------
    data: float 1-D arrays
        data[i] is the nonzero value of the tensor at index idxs[i, :].
    idxs: int 2-D array
        Let nnz be the number of nonzero entries of the tensor. Then idxs is an array
        of shape (nnz, L) such that idxs[i, :] is the index of the i-th nonzero entry.
    dims: list or tuple
        The dimensions (shape) of the tensor.
    """
    
    dims = T.shape
    L = len(dims)
    nnz = nonzero(T)
    idxs = []
    data = []

    for i in range(len(nnz[0])):
        idxs.append([nnz[l][i] for l in range(L)])
        data.append(T[tuple(idxs[i])])

    return [array(data), array(idxs), list(dims)]


def unfold(T, mode):
    """
    Computes any unfolding of a tensor up to order L = 12. 
    
    Inputs
    ------
    T: float L-D array
       The mode we are interested in. Note that 1 <= mode <= L.
        
    Outputs
    -------
    Tl: 2-D array with Fortran order
        The requested unfolding of T.
    """
 
    dims = T.shape
    L = len(dims)
    Tl = empty((dims[mode-1], int(prod(dims, dtype=uint64))//dims[mode-1]), order='F')
    func_name = "unfold" + str(mode) + "_order" + str(L)
    Tl = getattr(crt, func_name)(T, Tl, tuple(dims))

    return Tl


def unfold_C(T, mode):
    """
    Computes any unfolding of a tensor up to order L = 12. 
    
    Inputs
    ------
    T: float L-D array
       The mode we are interested in. Note that 1 <= mode <= L.
        
    Outputs
    -------
    Tl: 2-D array with C order
        The requested unfolding of T.
    """
 
    dims = T.shape
    L = len(dims)
    Tl = empty((dims[mode-1], int(prod(dims, dtype=uint64))//dims[mode-1]))
    func_name = "unfold" + str(mode) + "_order" + str(L)
    Tl = getattr(crt, func_name)(T, Tl, tuple(dims))

    return Tl


def sparse_unfold(data, idxs, dims, mode):
    """
    Computes any unfolding of a sparse L-th order tensor. 
    
    Inputs
    ------
    data: float 1-D arrays
        data[i] is the nonzero value of the tensor at index idxs[i, :].
    idxs: int 2-D array 
        Let nnz be the number of nonzero entries of the tensor. Then idxs is an array
        of shape (nnz, L) such that idxs[i, :] is the index of the i-th nonzero entry.
    dims: list or tuple
        The dimensions (shape) of the tensor.
    mode: int
        The mode we are interested in. Note that 1 <= mode <= L.
        
    Outputs
    -------
    Tl: csr matrix
        Sparse representation (in compressed sparse row format) of the requested unfolding.
    """
    
    L = len(dims)
    idx = list(arange(L))
    idx.remove(mode-1)
    K = [0 for l in range(L)]

    c = 0
    for l in range(L):
        if l == mode-1:
            K[l] = 0
        else:
            s = 1
            for ll in range(c):
                s *= int(dims[idx[ll]])
            K[l] = s
            c += 1
    
    rows = idxs[:, mode-1]
    cols = np.sum(K * idxs, axis=1, dtype=uint64)
    Tl = coo_matrix((data, (rows, cols)), shape=(dims[mode-1], int(prod(dims, dtype=uint64))//dims[mode-1]))
    Tl = Tl.tocsr()

    return Tl


def foldback(T, Tl, mode):
    """
    Computes the tensor with dimension dims given an unfolding with its mode. 
    """
 
    dims = T.shape
    L = len(dims)
    func_name = "foldback" + str(mode) + "_order" + str(L)
    T = getattr(crt, func_name)(T, Tl, tuple(dims))
    
    return T


def normalize(factors):
    """ 
    Normalize the columns of the factors to have unit column norm and scale Lambda accordingly. This function returns 
    Lambda and the normalized factors. 
    """

    R = factors[0].shape[1]
    Lambda = zeros(R)
    L = len(factors)
    new_factors = factors.copy()
    
    for r in range(R):
        norms = zeros(L)
        for l in range(L):
            W = factors[l]
            norms[l] = norm(W[:, r])
            W[:, r] = W[:, r]/norms[l]
            # Update factor matrix.
            new_factors[l] = W
        Lambda[r] = prod(norms)
        
    return Lambda, new_factors


def denormalize(Lambda, factors):
    """
    By undoing the normalization of the factors this function makes it unnecessary the use of the diagonal tensor
    Lambda. 
    """

    R = Lambda.size
    L = len(factors)
    new_factors = [zeros(factors[l].shape) for l in range(L)]

    for r in range(R):
        a = abs(Lambda[r])**(1/L)
        new_factors[0][:, r] = sign(Lambda[r]) * a * factors[0][:, r]
        for l in range(1, L):
            new_factors[l][:, r] = a * factors[l][:, r]

    return new_factors


def equalize(factors, R):
    """ 
    Let W[0], ..., W[L-1] = factors. After a Gauss-Newton iteration we have an approximated CPD with factors 
    W[0]_r ⊗ ... ⊗ W[L-1]_r. They may have very different magnitudes and this can have effect on the convergence 
    rate. To improve this we try to equalize their magnitudes by introducing scalars a_0, ..., a_{L-1} such that 
    W[0]_r ⊗ ... ⊗ W[L-1]_r = (a_0*W[0]_r) ⊗ ... ⊗ (a_{L-1}*W[L-1]_r) and |a_0*W[0]_r| = ... = |a_{L-1}*W[L-1]_r|. 
    Note that we must have a_0*...*a_{L-1} = 1.
    
    To find good values for the a_l's, we can search for critical points of the function 
    f(a_0,...,a_{L-1}) = sum_{i, j=0...L-1} (|a_i*W[i]_r|-|a_j*W[j]_r|)^2 .
    Using Lagrange multipliers we find the solution 
        a_0 = (|W[0]_r|*...*|W[L-1]_r|)^(1/L)/|W[0]_r|,
        ...
        a_L = (|W[0]_r|*...*|W[L-1]_r|)^(1/L)/|W[L-1]_r|.
    We can see that this solution satisfy the conditions mentioned.
    """
    
    L = len(factors)
    
    for r in range(R):
        norm_r = array([norm(factors[l][:, r]) for l in range(L)])
        if prod(norm_r) != 0.0:
            numerator = prod(norm_r)**(1/L)
            for l in range(L):
                factors[l][:, r] = (numerator/norm_r[l]) * factors[l][:, r]
            
    return factors


def change_sign(factors):
    """
    After the CPD is computed it may be interesting that each vector of a rank one term is as positive as possible, in
    the sense that its mean is positive. If two vectors in the same rank one term have negative mean, then we can
    multiply both by -1 without changing the tensor.
    """
    
    L = len(factors)
    R = factors[0].shape[1]
    
    for r in range(R):
        parity = 0
        for l in range(L):
            if factors[l][:, r].mean() < 0:
                factors[l][:, r] *= -1
                parity += 1
        if parity%2 == 1:
            factors[0] *= -1
            
    return factors


def transform(factors, symm, factors_norm):
    """
    The parameter symm indicates that the objective tensor is symmetric, so the program forces this symmetry over the
    factor matrices.
    The parameter factors_norm forces the factor matrices to always have the same prescribed norm, which is the value
    factors_norm.
    
    Inputs
    ------
    factors: list of 2D arrays
    symm: bool
    factors_norm: float
        
    Outputs
    -------
    factors: list of 2D arrays
    """ 

    L = len(factors)
        
    if symm:
        s = factors[0]
        for l in range(1, L):
            s += factors[l]
        factors[0] = s/L
        for l in range(1, L):
            factors[l] = factors[0]

    if factors_norm > 0:
        for l in range(L):
            factors[l] = factors_norm * (1/norm(factors[l])) * factors[l]
    
    return factors


@njit(nogil=True)
def vec(M, Bv, num_rows, R):
    """ 
    Take a matrix M with shape (num_rows, R) and stack vertically its columns to form the matrix Bv = vec(M) with shape
    (num_rows*R,).
    """
    
    for r in range(R):
        Bv[r*num_rows:(r+1)*num_rows] = M[:, r]
        
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
