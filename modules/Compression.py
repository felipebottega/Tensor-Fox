"""
 Compression Module
 ==================
 This module is responsible for all routines related to the compression of tensors, which ammounts to computing its
MLSVD and truncating it.
"""

# Python modules
import numpy as np
from numpy import eye, zeros, ones, empty, prod, float64, int64, copy, dot, sort, argsort
from sklearn.utils.extmath import randomized_svd as rand_svd
from decimal import Decimal

# Tensor Fox modules
import Auxiliar as aux
import Conversion as cnv
import MultilinearAlgebra as mlinalg


def mlsvd(T, Tsize, r, options):
    """
    This function computes the full MLSVD of tensors of any order. The output is such that T = (U_1,...,U_L)*S, and UT
    is the list of the transposes of U.
    """

    # INITIALIZE RELEVANT VARIABLES.    

    # Set the main variables about T.
    dims = T.shape
    L = len(dims)

    # Extract all variable from the class of options.
    trunc_dims = options.trunc_dims
    display = options.display
    mlsvd_tol = options.mlsvd_tol
    if type(mlsvd_tol) == list:
        if L > 3:
            mlsvd_tol = mlsvd_tol[0]
        else:
            mlsvd_tol = mlsvd_tol[1]

    # mlsvd_tol = -1 means no truncation and no compression, in other words, the original tensor.
    if mlsvd_tol == -1:
        U = [ eye(dims[l]) for l in range(L) ]
        UT = U
        sigmas = [ ones(dims[l]) for l in range(L) ]
        if display > 2 or display < -1:
            return T, U, UT, sigmas, 0.0
        else:
            return T, U, UT, sigmas

    # Compute truncated SVD of all unfoldings of T.
    sigmas = []
    U = []
    UT = []
    T1 = empty((dims[0], prod(dims) // dims[0]), dtype=float64)
    for l in range(L):
        Tl = cnv.unfold(T, l+1, dims)
        if l == 0:
            T1 = copy(Tl)
        low_rank = min(r, dims[l])
        Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_iter=2, power_iteration_normalizer='none')
        sigmas.append(sigma_l)
        U.append(Ul)
        UT.append(Ul.T)

    # mlsvd_tol = 0 means to not truncate the compression, we use the central tensor if the MLSVD without truncating it.
    if mlsvd_tol == 0:
        S = mlinalg.multilin_mult(UT, T1, dims)
        if display > 2 or display < -1:
            S1 = cnv.unfold(S, 1, dims)
            best_error = aux.compute_error(T, Tsize, S, S1, U, dims)
            return S, U, UT, sigmas, best_error
        else:
            return S, U, UT, sigmas

    # TRUNCATE SVD'S OF UNFOLDINGS

    # Specific truncation is given by the user.
    if type(trunc_dims) == list:
        best_U = []
        best_UT = []
        for l in range(L):
            best_U.append( U[l][:, :trunc_dims[l]] )
            best_UT.append( UT[l][:trunc_dims[l], :] )
        S = mlinalg.multilin_mult(best_UT, T1, dims)
        if display > 2 or display < -1:
            S1 = cnv.unfold(S, 1, trunc_dims)
            best_error = aux.compute_error(T, Tsize, S, S1, best_U, trunc_dims)
            return S, best_U, best_UT, sigmas, best_error
        else:
            return S, best_U, best_UT, sigmas

    # Clean SVD's, because the original SVD factors may have unnecessary information due to noise or numerical error.
    U, UT, sigmas = clean_compression(sigmas, U, UT, mlsvd_tol)

    # Compute (U_1^T,...,U_L^T)*T = S.
    S = mlinalg.multilin_mult(UT, T1, dims)

    # Compute error of compressed tensor.
    if display > 2 or display < -1:
        S1 = cnv.unfold(S, 1, S.shape)
        best_error = aux.compute_error(T, Tsize, S, S1, U, S.shape)
        return S, U, UT, sigmas, best_error

    return S, U, UT, sigmas


def clean_compression(sigmas, U, UT, mlsvd_tol):
    """
    This function try different threshold values to truncate the mlsvd. The conditions to accept a truncation are
    defined by the parameter level. Higher level means harder constraints, which translates to bigger tensors after the
    truncation.

    Inputs
    ------
    T: float 3-D ndarray
    T1: float 2-D ndarray
        First unfolding of T. This matrix is used compute the multilinear multiplication (U1^T, U2^T, U3^T)*T = S.
    Tsize: float
    S: float 3-D ndarray
        Central tensor obtained by the mlsvd.
    sigma1, sigma2, sigma3: float 1-D ndarrays
        Each one of these array is an ordered list (ascendent) with the singular values of the respective unfolding.
    U1, U2, U3: float 2-D ndarrays
    m, n, p, r: int
    level: 0, 1, 2, 3
        0 means the stopping conditions are very weak, while 3  means the stopping conditions are very hard.
    stage: 1, 2
        1 means we are at the first stage of cleaning. At this stage we can stop the program for a specific condition.
        After that the function is called again to improve the truncation. The second time we have stage == 2, so the
        mentioned condition won't be verified anymore.

    Outputs
    -------
    best_S: float 3-D ndarray
        Best truncation of S obtained.
    best_energy: float
        Energy that best_S retained with respect to S.
    best_R1, best_R2, best_R3: int
        Dimensions of best_S.
    best_U1, best_U2, best_U3: float 2-D ndarrays
        Truncated versions of U1, U2, U3.
    best_sigma1, best_sigma2, best_sigma3: float 1-D ndarrays
        Truncated versions of sigma1, sigma2, sigma3.
    mlsvd_stop: 0,1,2,3,4,5,6 or 7
    situation: str
        1) situation == 'random' means the function stopped with random truncation
        2) situation == 'ok' means the function stopped normally, without random truncation
    """

    # INITIALIZE RELEVANT VARIABLES.
    L = len(sigmas)
    eps = mlsvd_tol/L

    # COMPUTE TRUNCATION FOR EACH MODE.
    for l in range(L):
        sigma_l_len = len(sigmas[l])
        for i in range(1, sigma_l_len):
            Tl_trunc_error = np.sum(sigmas[l][i:]**2)/np.sum(sigmas[l]**2)
            if Tl_trunc_error < eps:
                sigmas[l] = sigmas[l][:i]
                U[l] = U[l][:, :i]
                UT[l] = UT[l][:i, :]
                break

    return U, UT, sigmas


def test_truncation(T, r, trunc_list, display=True, n_iter=2, power_iteration_normalizer='none'):
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
    Tsize = np.linalg.norm(T)

    # Compute truncated SVD of all unfoldings of T.
    sigmas = []
    U = []
    UT = []
    T1 = empty((dims[0], prod(dims) // dims[0]), dtype=float64)
    for l in range(L):
        Tl = cnv.unfold(T, l + 1, dims)
        if l == 0:
            T1 = copy(Tl)
        low_rank = min(r, dims[l])
        Ul, sigma_l, Vlt = rand_svd(Tl, low_rank, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer)
        sigmas.append(sigma_l)
        U.append(Ul)
        UT.append(Ul.T)

    # Save errors in a list.
    trunc_error = []
    
    # Truncated MLSVD.
    for trunc in trunc_list:
        # S, U and UT truncated.
        current_dims = trunc
        current_U = []
        current_UT = []
        current_sigmas = []
        for l in range(L):
            current_U.append(U[l][:, :current_dims[l]])
            current_UT.append(UT[l][:current_dims[l], :])
            current_sigmas.append(sigmas[l][:current_dims[l]])
        S = mlinalg.multilin_mult(current_UT, T1, dims)
        
        # Error of truncation.
        S1 = cnv.unfold(S, 1, current_dims)
        current_error = aux.compute_error(T, Tsize, S, S1, current_U, current_dims)
        trunc_error.append(current_error)

        # Display results.
        if display:
            print('Truncation:', current_dims)
            print('Error:', current_error)
            print()

    return trunc_error
