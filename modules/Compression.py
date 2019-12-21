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
"""

# Python modules
import numpy as np
from numpy import identity, ones, empty, array, prod, float64, copy
from numpy.linalg import norm
from sklearn.utils.extmath import randomized_svd as rand_svd
import sys

# Tensor Fox modules
import Auxiliar as aux
import Conversion as cnv
import MultilinearAlgebra as mlinalg


def mlsvd(T, Tsize, R, options):
    """
    This function computes a truncated MLSVD of tensors of any order. The output is such that T = (U_1,...,U_L)*S, and
    UT is the list of the transposes of U.
    The parameter n_iter of the randomized SVD is set to 2. It is only good to increase this value when the tensor has
    much noise. Still this issue is addressed by the low rank CPD approximation, so n_iter=2 is enough.

    Inputs
    ------
    T: float array
        Objective tensor in coordinates.
    Tsize: float
        Frobenius norm of T.
    R: int
        The desired rank of the approximating tensor.
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
    
    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        data, idxs, dims = T
    else:
        dims = T.shape
    L = len(dims) 

    # Set options.
    options = aux.make_options(options, L)
    trunc_dims = options.trunc_dims
    display = options.display
    mlsvd_method = options.mlsvd_method
    tol_mlsvd = options.tol_mlsvd
    if type(tol_mlsvd) == list:
        if L > 3:
            tol_mlsvd = tol_mlsvd[0]
        else:
            tol_mlsvd = tol_mlsvd[1]
            
    if type(T) == list:
        sigmas = []
        U = []
        for l in range(L):
            Tl = cnv.sparse_unfold(data, idxs, dims, l+1)
            if l == 0:
                T1 = Tl.copy()
            Tl = Tl.dot(Tl.T)
            low_rank = min(R, dims[l])
            Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_oversamples=10, n_iter=2, power_iteration_normalizer='none')

            # Truncate more based on energy.
            Ul, sigma_l, Vlt, dim = clean_compression(Ul, sigma_l, Vlt, tol_mlsvd, L)
            sigmas.append(sigma_l)
            U.append(Ul)

        # Compute (U_1^T,...,U_L^T)*T = S.
        new_dims = [U[l].shape[1] for l in range(L)]
        UT = [U[l].T for l in range(L)]
        S = mlinalg.sparse_multilin_mult(UT, data, idxs, new_dims)

        # Compute error of compressed tensor.
        if display > 2 or display < -1:
            S1 = cnv.unfold(S, 1)
            T1_dense = T1.toarray()
            T_dense = empty(dims, dtype=float64)
            T_dense = cnv.foldback(T_dense, T1_dense, 1)
            best_error = mlinalg.compute_error(T_dense, Tsize, S1, U, S.shape)
            return S, U, T1, sigmas, best_error
        else:
            return S, U, T1, sigmas
        

    # tol_mlsvd = -1 means no truncation and no compression, in other words, the original tensor.
    if tol_mlsvd == -1:
        T1 = cnv.unfold(T, 1)
        U = [ identity(dims[l]) for l in range(L) ]
        sigmas = [ ones(dims[l]) for l in range(L) ]
        if display > 2 or display < -1:
            return T, U, T1, sigmas, 0.0
        else:
            return T, U, T1, sigmas

    # Compute MLSVD base on sequentially truncated method.
    if mlsvd_method == 'seq':
        sigmas = []
        U = []
        S_dims = copy(dims)
        S = copy(T)
        for l in range(L):
            Sl = cnv.unfold(S, l+1)
            if l == 0:
                T1 = copy(Sl)
            low_rank = min(R, dims[l])
            Ul, sigma_l, Vlt = rand_svd(Sl, low_rank, n_oversamples=10, n_iter=2, power_iteration_normalizer='none')

            # Truncate more based on energy.
            Ul, sigma_l, Vlt, dim = clean_compression(Ul, sigma_l, Vlt, tol_mlsvd, L)
            sigmas.append(sigma_l)
            U.append(Ul)

            # Compute l-th unfolding of S truncated at the l-th mode.
            Sl = (Vlt.T * sigma_l).T  
            S_dims[l] = dim
            S = empty(S_dims, float64)
            S = cnv.foldback(S, Sl, l+1)
        
    # Compute MLSVD based on classic method.
    elif mlsvd_method == 'classic':
        sigmas = []
        U = []
        T1 = empty((dims[0], prod(dims) // dims[0]), dtype=float64)
        for l in range(L):
            Tl = cnv.unfold(T, l+1)
            if l == 0:
                T1 = copy(Tl)
            low_rank = min(R, dims[l])
            Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_oversamples=10, n_iter=2, power_iteration_normalizer='none')

            # Truncate more based on energy.
            Ul, sigma_l, Vlt, dim = clean_compression(Ul, sigma_l, Vlt, tol_mlsvd, L)
            sigmas.append(sigma_l)
            U.append(Ul)

        # Compute (U_1^T,...,U_L^T)*T = S.
        UT = [U[l].T for l in range(L)]
        S = mlinalg.multilin_mult(UT, T1, dims)
        
    else:
        sys.exit('Wrong MLSVD method name. Must be seq or classic.')

    # tol_mlsvd = 0 means to not truncate the compression, we use the core tensor if the MLSVD without truncating it.
    if tol_mlsvd == 0:
        UT = [U[l].T for l in range(L)]
        S = mlinalg.multilin_mult(UT, T1, dims)
        new_dims = [min(R, dims[l]) for l in range(L)]
        if display > 2 or display < -1:
            S1 = cnv.unfold(S, 1)
            best_error = mlinalg.compute_error(T, Tsize, S1, U, new_dims)
            return S, U, T1, sigmas, best_error
        else:
            return S, U, T1, sigmas

    # TRUNCATE SVD'S OF UNFOLDINGS

    # Specific truncation is given by the user.
    if type(trunc_dims) == list:
        best_U = []
        for l in range(L):
            if trunc_dims[l] > U[l].shape[1]:
                print('trunc_dims[', l, '] =', trunc_dims[l], 'and U[', l, '].shape =', U[l].shape)
                sys.exit('Must have trunc_dims[l] <= min(dims[l], R) for all mode l=1...' + str(L))
            best_U.append( U[l][:, :trunc_dims[l]] )
        best_UT = [best_U[l].T for l in range(L)]
        S = mlinalg.multilin_mult(best_UT, T1, dims)
        
        if display > 2 or display < -1:
            S1 = cnv.unfold(S, 1)
            best_error = mlinalg.compute_error(T, Tsize, S1, best_U, trunc_dims)
            return S, best_U, T1, sigmas, best_error
        else:
            return S, best_U, T1, sigmas

    # Compute error of compressed tensor.
    if display > 2 or display < -1:
        S1 = cnv.unfold(S, 1)
        best_error = mlinalg.compute_error(T, Tsize, S1, U, S.shape)
        return S, U, T1, sigmas, best_error

    return S, U, T1, sigmas


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
    T1 = empty((dims[0], prod(dims) // dims[0]), dtype=float64)
    for l in range(L):
        Tl = cnv.unfold(T, l+1)
        if l == 0:
            T1 = copy(Tl)
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
