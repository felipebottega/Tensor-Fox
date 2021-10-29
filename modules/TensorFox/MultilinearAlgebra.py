"""
 Multilinear Algebra Module
 ==========================
 All relevant functions of the classical multilinear algebra are described and implemented in this module.

 References
 ==========

 - C. J. Hillar, and L.-H. Lim. Most tensor problems are NP-hard, Journal of the ACM, 60(6):45:1-45:39, November 2013.
   ISSN 0004-5411. doi: 10.1145/2512329.

 - P. Breiding and N. Vannieuwenhoven, A Riemannian Trust Region Method for the Canonical Tensor Rank Approximation
   Problem, SIAM J. Optim., 28(3), 2435-2465.

 - P. Breiding and N. Vannieuwenhoven, The Condition Number of Join Decompositions, arXiv:1611.08117v3 (2018).
"""

# Python modules
import numpy as np
from numpy import dot, zeros, empty, prod, uint64, float64, array, sort, ceil, identity, argmax, inf, sqrt, arange
from numpy.linalg import norm, svd
import numpy.matlib
import scipy as scp
from numba import njit, prange

# Tensor Fox modules
import TensorFox.Auxiliar as aux
import TensorFox.Compression as cmpr
import TensorFox.Conversion as cnv
import TensorFox.Critical as crt


def multilin_mult_cpd(U, W, dims):
    """    
    Performs the multilinear multiplication (U[0],...,U[L-1])*(W[0], ..., W[L-1])*I = (U[0]*W[0],...,U[L-1]*W[L-1])*I, 
    where I.shape = dims = (W[0].shape[1],...,W[L-1].shape[1]) are the size of the columns of the W's.

    Inputs
    ------
    U: list of 2-D arrays
    W: list of 2-D arrays
    dims: list of ints

    Outputs
    -------
    S: float array
        S is the resulting multidimensional of the mentioned multiplication.
    """

    L = len(dims)
    # dims_out are the dimensions of the output tensor.
    dims_out = [] 
    W_new = []
    
    for l in range(L):
        W_new.append( dot(U[l], W[l]) )
        dims_out.append(W_new[l].shape[0])
    
    S = cnv.cpd2tens(W_new)
    return S


def multilin_mult(U, T1, dims):
    """    
    Performs the multilinear multiplication (U[0]^T,...,U[L-1]^T)*T, where U[i]^T is the transpose of U[i], 
    dims = T.shape. We need the first unfolding T1 of T to start the computations.

    Inputs
    ------
    U: list of 2-D arrays
    T1: 2-D array
        First unfolding of T.
    dims: list of ints
        Dimension of T.

    Outputs
    -------
    S: float array
        S is the resulting multidimensional of the multilinear multiplication (U[0]^T,...,U[L-1]^T)*T.
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
        S = cnv.foldback(S, unfolding2, l+1)
        if l < L-1:            
            unfolding1 = cnv.unfold(S, l+2)
        else:
            return S


def sparse_multilin_mult(U, data, idxs, dims):
    """
    Performs the multilinear multiplication (U[0]^T,...,U[L-1]^T)*T, where U[i]^T is the transpose of U[i], 
    dims = T.shape and T is sparse. The first unfolding T1 of T is given as a csr matrix.

    Inputs
    ------
    U: list of 2-D arrays
    data: float 1-D arrays
        data[i] is the nonzero value of the tensor at index idxs[i, :].
    idxs: int 2-D array
        Let nnz be the number of nonzero entries of the tensor. Then idxs is an array of shape (nnz, L) such that
        idxs[i, :] is the index of the i-th nonzero entry.
    dims: list or tuple
        The dimensions (shape) of the tensor T.

    Outputs
    -------
    S: float array
        S is the resulting multidimensional array of the multilinear multiplication (U[0]^T,...,U[L-1]^T)*T.
    """

    L = len(dims)
    # dims_out are the dimensions of the output tensor S.
    dims_out = [U[l].shape[0] for l in range(L)]
    S = empty(dims_out, dtype=float64)
    func_name = "sparse_multilin_mult_order" + str(L)
        
    # Generate a different version of U to deal with sparse index accesses.
    nnz = len(data)    
    U_tmp = [U[l][:, idxs[:, l]] for l in range(L)]
    data_list = [array(data) for i in range(dims_out[0])]
            
    # Run the multiplication function.
    try: 
        S = getattr(crt, func_name)(U_tmp, data_list, S, dims_out)
    except:
        # Change arrays order to be compatible with Numba function.
        U_tmp = [array(U[l][:, idxs[:, l]], order='C') for l in range(L)]
        data_list = [array(data, order='A') for i in range(dims_out[0])]        
        S = getattr(crt, func_name)(U_tmp, data_list, S, dims_out)
    
    # Free memory.
    U_tmp = []

    return S


def multirank_approx(T, multi_rank, options):
    """
    This function computes an approximation of T with multilinear rank = multi_rank. Truncation the core tensor of the
    MLSVD doesn't gives the best low multirank approximation, but gives very good approximations.
    
    Inputs
    ------
    T: float array
    multi_rank: list of int
        The desired low multilinear rank.
        
    Outputs
    -------
    T_approx: float array
        The approximating tensor with multilinear rank = multi_rank.
    """
    
    # Compute dimensions and norm of T.
    dims = T.shape
    sorted_dims = sort(array(dims))
    L = len(dims)
    Tsize = norm(T)
    
    # Compute truncated MLSVD of T.
    options = aux.make_options(options)
    options.display = 0
    options.trunc_dims = multi_rank
    R_gen = int(ceil( int(prod(sorted_dims, dtype=uint64))/(np.sum(sorted_dims) - L + 1) ))
    S, U, UT, sigmas = cmpr.mlsvd(T, Tsize, R_gen, options)

    # Construct the corresponding tensor T_approx.
    S1 = cnv.unfold(S, 1)
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
    
    for i in range(a1):
        for j in range(a2):
            M[i*b1: (i+1)*b1, j*b2: (j+1)*b2] = A[i, j]*B

    return M


@njit(nogil=True, parallel=True)
def khatri_rao(A, B, M):
    """
    Computes the Khatri-Rao product between A and B. We must have M.shape = (a1*b1, a2), where A.shape = (a1, a2) and 
    B.shape = (b1, b2), with a2 == b2. This function makes the computation of A ⊙ B row by row, starting at the top.
    """

    a1, a2 = A.shape
    b1, b2 = B.shape
    
    for i in prange(a1):
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

    return M[i*b1: (i+1)*b1, :]


@njit(nogil=True)
def hadamard(A, B, M):
    """
    Computes M = A * B, where * is the Hadamard product. Since all Hadamard products in this context are between R x R
    matrices, we assume this without verifications.
    """

    R = A.shape[0]
    
    for r in range(R):
        for rr in range(R):
            M[r, rr] = A[r, rr]*B[r, rr]

    return M


def cond(factors):
    """
    Computes the geometric condition number of 'factors', where factors is a list with the factor matrices of some CPD.
    Warning: this function requires a lot of memory.
    """

    def trd_jacobian(A):
        """
        Computes the derivative of the map S^r -> s_r(S).
        """

        L = len(A)
        dims = array([A[i].shape[0] for i in range(L)])
        N = 1 + np.sum(dims - 1)
        r = A[0].shape[1]
        Pi = int(prod(dims, dtype=uint64))
        Sigma = 1 + np.sum(dims - 1)

        idxSet = zeros((L - 1, r))
        for k in range(1, L):
            idxSet[k - 1, :] = argmax(np.abs(A[k]), 0)

        J = zeros((Pi, r * Sigma))
        off = 0
        for i in range(r):
            F = [np.matlib.repmat(scp.linalg.orth(A[j][:, i].reshape(A[j].shape[0], 1)), 1, dims[0]) for j in range(L)]
            F[0] = identity(dims[0])
            F_inverse = [F[i] for i in reversed(range(L))]
            G = khatri_rao_factors(F_inverse)
            J[:, off: off + dims[0]] = G
            off = off + dims[0]

            for k in range(1, L):
                F = [np.matlib.repmat(scp.linalg.orth(A[j][:, i].reshape(A[j].shape[0], 1)), 1, dims[k] - 1) for j in
                     range(L)]
                I = identity(dims[k])
                ri = int(idxSet[k - 1, i])
                cols1 = [i for i in range(ri)]
                cols2 = [i for i in range(ri + 1, dims[k])]
                F[k] = I[:, cols1 + cols2]
                F_inverse = [F[i] for i in reversed(range(L))]
                G = khatri_rao_factors(F_inverse)
                J[:, off: off + dims[k] - 1] = G
                off = off + dims[k] - 1

            u, s, vt = svd(J[:, off - N + 1:off], full_matrices=False)
            J[:, off - N + 1:off] = u

        return J

    L = len(factors)
    dims = array([factors[l].shape[0] for l in range(L)])
    R = factors[0].shape[1]

    # Verify if current rank is bigger than the generic rank of the space.
    P = int(prod(dims, dtype=uint64))
    S = R * (1 + np.sum(dims - 1))
    if P < S:
        J = []
        cN = inf
        return cN, J

    J = trd_jacobian(factors)
    u, s, vt = svd(J, full_matrices=False)
    cN = s[-1] ** (-1)

    return cN


def khatri_rao_factors(factors):
    """
    Computes the Khatri-Rao products W^(1) ⊙ W^(2) ⊙ ... ⊙ W^(L) between the factor matrices.
    """

    L = len(factors)
    A = factors[0]
    for l in range(1, L):
        m, R = factors[l].shape
        B = zeros((m * A.shape[0], R))
        B = khatri_rao(A, factors[l], B)
        A = B

    return B


def compute_error(T, Tsize, S1, U, dims):
    """
    Compute relative error between T and (U_1,...,U_L)*S, where dims is the shape of S. In the case T is sparse, we 
    should pass S instead of the unfolding S1.
    """

    # T is sparse.
    if type(T) == list:
        S = S1
        L = len(U)
        UT = [U[l].T for l in range(L)]
        data, idxs, Tdims = T
        T_compress = sparse_multilin_mult(UT, data, idxs, Tdims)
        error = norm(T_compress - S) / Tsize
    # T is dense.
    else:
        T_compress = multilin_mult(U, S1, dims)
        error = norm(T - T_compress)/Tsize

    return error


def rank1_terms_list(factors):
    """
    Compute each rank 1 term, as a multidimensional array, of the CPD. Let T be the corresponding the tensor, in
    coordinates, of thr CPD given by factors, and let rank1_terms = [T_1, T_2, ..., T_R] be the output of this function.
    Then we have that T_1 + T_2 + ... + T_R = T.

    Inputs
    ------
    factors: list of float 2-D ndarrays with shape (dims[i], R) each
        The CPD factors of some tensor.

    Outputs
    -------
    rank1_terms: list of float ndarrays
        Each tensor rank1_terms[r] is the r-th rank-1 term of the given CPD.
    """

    R = factors[0].shape[1]
    L = len(factors)
    rank1_terms = []

    for r in range(R):
        vectors = []
        for l in range(L):
            # vectors[l] = [w_r^(1),w_r^(2),...,w_r^(L)], which represents w_r^(1) ⊗ w_r^(2) ⊗ ... ⊗ w_r^(L).
            v = factors[l][:, r]
            vectors.append(v.reshape(v.size, 1))
        term = cnv.cpd2tens(vectors)
        rank1_terms.append(term)

    return rank1_terms


@njit(nogil=True, parallel=True)
def rank1(X, Y, Z, m, n, R, k):
    """
    Computes the k-th slice of each rank 1 term of the CPD given by X, Y, Z.  By doing this for all R terms we have a
    tensor with R slices, each one representing a rank-1 term of the original CPD.

    Inputs
    ------
    X, Y, Z: 2-D float array
        The CPD factors of some third order tensor.
    m, n, p, R: int
    k: int
        Slice we want to compute.

    Outputs
    -------
    rank1_slices: 3-D float array
        Each matrix rank1_slices[:, :, l] is the k-th slice associated with the l-th factor in the CPD of some tensor.
    """

    # Each frontal slice of rank1_slices is the coordinate representation of a
    # rank one term of the CPD given by (X,Y,Z)*Lambda.
    rank1_slices = zeros((m, n, R), dtype=float64)

    for r in prange(R):
        for i in range(m):
            for j in range(n):
                rank1_slices[i, j, r] = X[i, r] * Y[j, r] * Z[k, r]

    return rank1_slices


def forward_error(orig_factors, approx_factors):
    """
    Let T = T_1 + T_2 + ... + T_R be the decomposition of T as sum of rank-1 terms and let
    T_approx = T_approx_1 + T_approx_2 + ... + T_approx_R be the decomposition of T_approx as sum of R terms. Supposedly
    T_approx is obtained after the cpd function call. The ordering of the rank-1 terms of T_approx can be permuted freely
    without changing the tensor. While |cpd2tens(T) - cpd2tens(T_approx)| is the backward error of the CPD computation 
    problem, we have that min_s sqrt( |T_1 - T_approx_s(1)|^2 + ... + |T_R - T_approx_s(R)|^2 ) is the forward error of 
    the problem, where s is an element of the permutation group S_R.

    Inputs
    ------
    orig_factors: list of arrays
        The elements of the list are the factor matrices of the original tensor.
    approx_factors: list of arrays
        The elements of the list are the factor matrices of the approximated tensor.

    Outputs
    -------
    best_error: float
        Relative error of the best permutation found.
    best_factors: list
        List with the new factor matrices of the best permutation.
    best_s: list
        The indexes of the best permutation found.
    """

    R = orig_factors[0].shape[1]
    L = len(orig_factors)
    orig_rank1 = rank1_terms_list(orig_factors)
    approx_rank1 = rank1_terms_list(approx_factors)

    best_forward_error, s = search_forward_error(tuple(orig_rank1), tuple(approx_rank1), R)

    # Rearrange approx_factors with the best permutation found.
    new_factors = [approx_factors[l][:, s] for l in range(L)]

    return best_forward_error, new_factors, s


def search_forward_error(orig_rank1, approx_rank1, R):
    """
    Auxiliary function for the function forward_error.
    """
    
    best_error = 0
    best_s = arange(R)
    idx = []
    for r in range(R):
        f_error = inf
        for rr in range(len(approx_rank1)):
            if (rr not in idx) and (norm(orig_rank1[r] - approx_rank1[rr]) < f_error):
                best_s[r] = rr
                f_error = norm(orig_rank1[r] - approx_rank1[rr])
        idx.append(best_s[r])
        best_error += f_error**2
    best_error = sqrt(best_error)

    return best_error, best_s


def slow_sparse_dot(A):
    """
    This function computes dot(A, A.T), where A is a sparse csr matrix. The function compute_svd calls this product when
    the sparse_dot_mkl and scipy dot fails to perform the product, usually due to memory limitations. They tend to
    explode the memory for too large column sizes, whereas this functions perform well for large column sizes but it
    will explode for large row sizes.
    """

    n = A.shape[0]
    out_arr = np.zeros((n, n), dtype=A.dtype)

    for i in range(n):
        for j in range(n):
            idxs = set(A[i, :].indices).intersection(A[j, :].indices)
            out_arr[i, j] = sum([A[i, k] * A[j, k] for k in idxs])

    return out_arr
