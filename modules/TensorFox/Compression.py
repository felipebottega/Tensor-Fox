"""
 Compression Module
 ==================
 This module is responsible for all routines related to the compression of tensors, which amounts to computing its
 MLSVD and truncating it.

 References
 ==========

 - L. De Lathauwer, B. De Moor, and J. Vandewalle, A Multilinear Singular Value Decomposition, SIAM J. Matrix Anal.
   Appl., 21 (2000), pp. 1253-1278.

 - N. Vannieuwenhoven, R. Vandebril, and K. Meerbergen, A new truncation strategy for the higher-order singular value
   decomposition, SIAM J. Sci. Comput. 34 (2012), no. 2, A1027-A1052.
   
 - https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition#Computation

 - N. Halko, P. Martinsson, and J. Tropp. Finding structure with randomness: probabilistic algorithms for constructing
   approximate matrix decompositions (2009).

 - S. Voronin and P.Martinsson. RSVDPACK: Subroutines for computing partial singular value decompositions via
   randomized sampling on single core, multi core, and GPU architectures (2015).
"""

# Python modules
import numpy as np
from numpy import identity, ones, empty, array, uint64, float32, float64, copy, sqrt, prod, dot, ndarray
from numpy.linalg import norm
from sklearn.utils.extmath import randomized_svd as rand_svd
from scipy.sparse import coo_matrix
import sys

# Tensor Fox modules
import TensorFox.Auxiliar as aux
import TensorFox.Conversion as cnv
import TensorFox.MultilinearAlgebra as mlinalg


def mlsvd(T, Tsize, R, options):
    """
    This function computes a truncated MLSVD of tensors of any order. The output is such that T = (U_1,...,U_L)*S, and
    UT is the list of the transposes of U.
    The parameter n_iter of the randomized SVD is set to 2. It is only good to increase this value when the tensor has
    much noise. Still this issue is addressed by the low rank CPD approximation, so n_iter=2 is enough.
    We remark that if T is given as a sparse tensor, the component 'data' must be a float32.

    Inputs
    ------
    T: float array
        Objective tensor in coordinates.
    Tsize: float
        Frobenius norm of T.
    R: int
        An upper bound for the multilinear rank of T. Normally one will use the rank of T.
    options: class with the parameters previously defined.

    Outputs
    -------
    S: float array
        Core tensor of the MLSVD.
    U: list of float 2-D arrays
        List with truncated matrices of the original U.
    T1: float 2-D arrays
        First unfolding of T.
    sigmas: list of float 1-D arrays
        List with truncated arrays of the original sigmas.
    """

    # INITIALIZE RELEVANT VARIABLES.

    sigmas = []
    U = []
    
    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        data, idxs, dims = T
        if type(idxs) != ndarray:
            idxs = array(idxs, dtype=uint64)
    else:
        dims = T.shape
    L = len(dims) 

    # Set options.
    options = aux.make_options(options, L)
    trunc_dims = options.trunc_dims
    display = options.display
    mlsvd_method = options.mlsvd_method
    tol_mlsvd = options.tol_mlsvd
    mkl_dot = options.mkl_dot
    if type(tol_mlsvd) == list:
        if L > 3:
            tol_mlsvd = tol_mlsvd[0]
        else:
            tol_mlsvd = tol_mlsvd[1]
    gpu = options.gpu
    if gpu:
        import pycuda.gpuarray as gpuarray
        import pycuda.autoinit
        from skcuda import linalg, rlinalg

    # tol_mlsvd = -1 means no truncation and no compression, that is, the original tensor.
    if tol_mlsvd == -1:
        T1 = cnv.unfold(T, 1)
        U = [identity(dims[l]) for l in range(L)]
        sigmas = [ones(dims[l]) for l in range(L)]
        if display > 2 or display < -1:
            return T, U, T1, sigmas, 0.0
        else:
            return T, U, T1, sigmas
    
    # T is sparse.        
    elif type(T) == list:
        for l in range(L):
            Tl = cnv.sparse_unfold(data, idxs, dims, l+1)
            if l == 0:
                T1 = cnv.sparse_unfold(data, idxs, dims, l+1)
            mlsvd_method = 'sparse'
            U, sigmas, Vlt, dim = compute_svd(Tl, U, sigmas, dims, R, mlsvd_method, tol_mlsvd, gpu, mkl_dot, L, l)
            
        # Compute (U_1^T,...,U_L^T)*T = S.
        new_dims = [U[l].shape[1] for l in range(L)]
        UT = [U[l].T for l in range(L)]
        S = mlinalg.sparse_multilin_mult(UT, data, idxs, new_dims)

    # Compute MLSVD base on sequentially truncated method.
    elif mlsvd_method == 'seq':
        S_dims = copy(dims)
        S = T
        for l in range(L):
            Sl = cnv.unfold(S, l+1)
            if l == 0:
                T1 = cnv.unfold_C(S, l+1)
            U, sigmas, Vlt, dim = compute_svd(Sl, U, sigmas, dims, R, mlsvd_method, tol_mlsvd, gpu, mkl_dot, L, l)

            # Compute l-th unfolding of S truncated at the l-th mode.
            Sl = (Vlt.T * sigmas[-1]).T
            S_dims[l] = dim
            S = empty(S_dims, dtype=float64)
            S = cnv.foldback(S, Sl, l+1)

    # Compute MLSVD based on classic method.
    elif mlsvd_method == 'classic':
        for l in range(L):
            Tl = cnv.unfold(T, l+1)
            if l == 0:
                T1 = cnv.unfold_C(T, l+1)
            U, sigmas, Vlt, dim = compute_svd(Tl, U, sigmas, dims, R, mlsvd_method, tol_mlsvd, gpu, mkl_dot, L, l)

        # Compute (U_1^T,...,U_L^T)*T = S.
        UT = [U[l].T for l in range(L)]
        S = mlinalg.multilin_mult(UT, T1, dims)

    # Specific truncation is given by the user.
    if type(trunc_dims) == list:
        slices = []
        for l in range(L):
            slices.append(slice(0, trunc_dims[l]))
            if trunc_dims[l] > U[l].shape[1]:
                print('trunc_dims[', l, '] =', trunc_dims[l], 'and U[', l, '].shape =', U[l].shape)
                sys.exit('Must have trunc_dims[l] <= min(dims[l], R) for all mode l=1...' + str(L))
            U[l] = U[l][:, :trunc_dims[l]]
        S = S[tuple(slices)]

    # Compute error of compressed tensor.
    if display > 2 or display < -1:
        if type(T) == list:
            best_error = mlinalg.compute_error(T, Tsize, S, U, dims)
        else:
            S1 = cnv.unfold(S, 1)
            best_error = mlinalg.compute_error(T, Tsize, S1, U, S.shape)
        return S, U, T1, sigmas, best_error

    return S, U, T1, sigmas


def compute_svd(Tl, U, sigmas, dims, R, mlsvd_method, tol_mlsvd, gpu, mkl_dot, L, l):
    """
    Subroutine of the function mlsvd. This function performs the SVD of a given unfolding.
    """
    
    low_rank = min(R, dims[l])

    if gpu:
        if mlsvd_method == 'gpu' or mlsvd_method == 'sparse':
            tmp = array(dot(Tl, Tl.T), dtype=float32, order='F')
            Tl_gpu = gpuarray.to_gpu(tmp)
            Ul, sigma_l, Vlt = rlinalg.rsvd(Tl_gpu, k=low_rank, p=10, q=2, method='standard')
            sigma_l = sqrt(sigma_l)
        else:
            tmp = array(Tl.T, dtype=float32)
            Tl_gpu = gpuarray.to_gpu(tmp)
            Ul, sigma_l, Vlt = rlinalg.rsvd(Tl_gpu, k=low_rank, p=10, q=2, method='standard')

    else:
        if mlsvd_method == 'sparse':
            if mkl_dot:
                try:
                    from sparse_dot_mkl import dot_product_mkl
                except:
                    print('Module sparse_dot_mkl could not be imported. Using standard scipy dot function instead.')
                    mkl_dot = False
                if mkl_dot:
                    TlT = Tl.T
                    Tl = dot_product_mkl(Tl, TlT, copy=False)
                else:
                    Tl = Tl.dot(Tl.T)                    
            else:  
                Tl = Tl.dot(Tl.T) 
            # The function rand_svd works better with float64 types. This conversion is only necessary for sparse tensors.
            Tl = Tl.astype(float64, copy=False)
            Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_oversamples=10, n_iter=2, power_iteration_normalizer='none')
            sigma_l = sqrt(sigma_l)
        else:
            Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_oversamples=10, n_iter=2, power_iteration_normalizer='none')

    # Truncate more based on energy.
    Ul, sigma_l, Vlt, dim = clean_compression(Ul, sigma_l, Vlt, tol_mlsvd, L)
    sigmas.append(sigma_l)
    U.append(Ul)

    return U, sigmas, Vlt, dim


def clean_compression(U, sigma, Vt, tol_mlsvd, L):
    """
    This function try different threshold values to truncate the mlsvd. The conditions to accept a truncation are
    defined by the parameter level. Higher level means harder constraints, which translates to bigger tensors after the
    truncation.

    Inputs
    ------
    U, sigma, Vt: float arrays
        Arrays of some SVD of the form M = U * diag(sigma) * V^T.
    tol_mlsvd: float
        Tolerance criterion for the truncation. The idea is to obtain a truncation (U_1,...,U_L)*S such that
        |T - (U_1,...,U_L)*S| / |T| < tol_mlsvd.
    L: int
        Number of modes of the original tensor.

    Outputs
    -------
    U, sigma, Vt: float ndarrays
        Arrays of the SVD after truncation.
    dim: int
        New dimension size after truncation. 
    """    

    # INITIALIZE RELEVANT VARIABLES.
    eps = tol_mlsvd/L

    # COMPUTE TRUNCATION.
    sigma_len = len(sigma)
    for i in range(1, sigma_len):
        Tl_trunc_error = np.sum(sigma[i:]**2)/np.sum(sigma**2)
        if Tl_trunc_error < eps:
            sigma = sigma[:i]
            U = U[:, :i]
            Vt = Vt[:i, :]
            break
            
    # Size of truncation.
    dim = sigma.size

    return U, sigma, Vt, dim


def test_truncation(T, trunc_list, display=True, n_iter=2, power_iteration_normalizer='none'):
    """
    This function test one or several possible truncations for the MLSVD of T, showing the  error of the truncations. It
    is possible to accomplish the same results calling the function mlsvd with display=3 but this is not advisable since
    each call recomputes the same unfolding SVD's.
    The variable trunc_list must be a list of truncations. Even if it is only one truncation, it must be a list with one
    truncation only.
    """

    # Set the main variables about T.
    dims = T.shape
    L = len(dims)
    Tsize = norm(T)

    # Transform list into array and get the maximum for each dimension.
    max_trunc_dims = np.max(array(trunc_list), axis=0)

    # Compute truncated SVD of all unfoldings of T.
    sigmas = []
    U = []
    T1 = empty((dims[0], int(prod(dims, dtype=uint64)) // dims[0]), dtype=float64)
    for l in range(L):
        Tl = cnv.unfold(T, l+1)
        if l == 0:
            T1 = cnv.unfold_C(T, l+1)
        low_rank = min(dims[l], max_trunc_dims[l])
        Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer)
        sigmas.append(sigma_l)
        U.append(Ul)

    # Save errors in a list.
    trunc_error = []

    # Truncated MLSVD.
    for trunc in trunc_list:
        # S, U and UT truncated.
        current_dims = trunc
        current_U = []
        current_sigmas = []
        for l in range(L):
            current_U.append(U[l][:, :current_dims[l]])
            current_sigmas.append(sigmas[l][:current_dims[l]])
        current_UT = [current_U[l].T for l in range(L)]
        S = mlinalg.multilin_mult(current_UT, T1, dims)

        # Error of truncation.
        S1 = cnv.unfold(S, 1)
        current_error = mlinalg.compute_error(T, Tsize, S1, current_U, current_dims)
        trunc_error.append(current_error)

        # Display results.
        if display:
            print('Truncation:', current_dims)
            print('Error:', current_error)
            print()

    return trunc_error


def sparse_dot(Tl):
    """
    Given a csr sparse matrix Tl, this function computes the product Tl * Tl.T. The result is also given as a csr sparse
    matrix. This function is used only when the direct method Tl.dot(Tl.T) fails. This happens when the number of
    columns of Tl is very large. The dot method is faster but its memory consumption varies with the number of columns,
    whereas this function is slower but requires less memory usage (it varies with the nonzero entries of Tl).
    """

    m = Tl.shape[0]
    B = Tl.tocsr()
    C = Tl.tocsr().T

    rows, cols, datas = [], [], []

    # Store the sparse arrays in a list to speed-up the computations.
    bi = [B[i, :] for i in range(B.shape[0])]
    cjt = [C[:, j].T for j in range(C.shape[1])]

    # Compute lower trinagular (not the diagonal) entries and assign to its transpose entries.
    for i in range(m):
        for j in range(i):
            tmp = bi[i].multiply(cjt[j]).data
            if len(tmp) > 0:
                tmpsum = tmp.sum()
                rows.append(i)
                cols.append(j)
                datas.append(tmpsum)
                rows.append(j)
                cols.append(i)
                datas.append(tmpsum)
     
    # Compute the diagonal entries.
    for i in range(m):
        tmp = bi[i].multiply(cjt[i]).data
        if len(tmp) > 0:
            tmpsum = tmp.sum()
            rows.append(i)
            cols.append(i)
            datas.append(tmpsum)

    out = coo_matrix((datas, (rows, cols)), shape=(m, m))
    out = out.tocsr()

    return out
