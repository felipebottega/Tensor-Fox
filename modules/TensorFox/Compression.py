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
import sys
import numpy as np
from numpy import identity, ones, empty, array, uint64, float32, float64, copy, sqrt, prod, dot, ndarray, argmax, newaxis, sign
from numpy.linalg import norm
#from sklearn.utils.extmath import randomized_svd as rand_svd
from scipy import sparse
from scipy.linalg import qr, svd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Tensor Fox modules
import TensorFox.Auxiliar as aux
import TensorFox.Conversion as cnv
import TensorFox.MultilinearAlgebra as mlinalg


# try to import dot_product_mkl. It will be used in intermediate computations.
try:
    from sparse_dot_mkl import dot_product_mkl
except:
    print('Module sparse_dot_mkl could not be imported. Standard scipy dot will be used for sparse matrix multiplications.\nFor more information see https://github.com/felipebottega/Tensor-Fox/blob/master/README.md#sparse-dot-mkl-requirements.', file=sys.stderr)


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
        data = array(data, dtype=float64)
        idxs = array(idxs, dtype=uint64)
        dims = array(dims)
    else:
        dims = T.shape
    L = len(dims) 

    # Set options.
    options = aux.make_options(options)
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
            if display != 0:
                print('    Compressing unfolding mode', l+1)
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
            if display != 0:
                print('    Compressing unfolding mode', l+1)
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
            if display != 0:
                print('    Compressing unfolding mode', l+1)
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
                Tl = sparse_dot_mkl_call(Tl, mkl_dot)
            else:  
                Tl = sparse_dot_calls(Tl)

            Ul, sigma_l, Vlt = randomized_svd(Tl, low_rank, mkl_dot, n_oversamples=10, n_iter=2)
            sigma_l = sqrt(sigma_l)

        else:
            Ul, sigma_l, Vlt = randomized_svd(Tl, low_rank, mkl_dot, n_oversamples=10, n_iter=2)

    # Truncate more based on energy.
    Ul, sigma_l, Vlt, dim = clean_compression(Ul, sigma_l, Vlt, tol_mlsvd, L)
    sigmas.append(sigma_l)
    U.append(Ul)

    return U, sigmas, Vlt, dim
    
    
def sparse_dot_mkl_call(Tl, mkl_dot):
    """
    This function tries to compute the product dot(Tl, Tl.T) with the method from the package sparse_dot_mkl. If this
    method fails, the program calls the function sparse_dot_calls. Only the function compute_svd calls this function.
    """
    
    # Defines environment variable and make import.
    try:
        # Set MKL interface layer to int64 before importing the package. This handles bigger tensors.
        import os
        os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
        from sparse_dot_mkl import dot_product_mkl
    except:
        mkl_dot = False
        
    if mkl_dot:
        try:
            TlT = Tl.T
            Tl = dot_product_mkl(Tl, TlT, copy=False, dense=True)
        except Exception as e:
            print('        ' + str(e) + '. Using standard scipy dot.', file=sys.stderr)
            Tl = sparse_dot_calls(Tl)
    else:
        Tl = sparse_dot_calls(Tl)
        
    return Tl
    
    
def sparse_dot_calls(Tl):
    """
    This function tries to compute the product dot(Tl, Tl.T) with 2 methods: standard Scipy dot and a naive method 
    written in this project, called slow_sparse_dot. The former is faster but consumes a lot of memory, whereas the
    latter is slower but consumes less memory. Only the function compute_svd calls this function.
    """

    try:
        Tl = Tl.dot(Tl.T)
    except Exception as e:
        try:
            print('        ' + str(e) + '. Using slow sparse dot.')
            Tl = mlinalg.slow_sparse_dot(Tl)
        except Exception as e:
            sys.exit('        ' + str(e))
            
    return Tl


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
    
    
def safe_sparse_dot(a, b, mkl_dot):
    """
    Dot product that handle the sparse matrix case correctly

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if a and b are sparse.
    """
    
    if mkl_dot:
        try:
            from sparse_dot_mkl import dot_product_mkl
        except:
            mkl_dot = False
        if mkl_dot:
            if (sparse.issparse(a) and a.getformat() == 'csr') or (sparse.issparse(b) and b.getformat() == 'csr'):
                ret = dot_product_mkl(a, b, copy=False, dense=True)
            else:
                ret = a @ b
        else:
            ret = a @ b
    else:
        ret = a @ b

    return ret


def randomized_range_finder(A, mkl_dot, size, n_iter, random_state=None):
    """
    Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix

    size : integer
        Size of the return array

    n_iter : integer
        Number of power iterations used to stabilize the result

        .. versionadded:: 0.18

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    
    random_state = np.random.mtrand._rand

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)
    
    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        Q = safe_sparse_dot(A, Q, mkl_dot)
        Q = safe_sparse_dot(A.T, Q, mkl_dot)
        
    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr(safe_sparse_dot(A, Q, mkl_dot), mode='economic')
    
    return Q


def randomized_svd(M, n_components, mkl_dot, n_oversamples=10, n_iter='auto', transpose='auto', flip_sign=True, random_state=0):
    """
    Computes a truncated randomized SVD

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.

        .. versionchanged:: 0.18

    power_iteration_normalizer : 'auto' (default), 'QR', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger). The 'auto' mode applies no normalization 
        if `n_iter` <= 2.

        .. versionadded:: 0.18

    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    
    random_state = np.random.mtrand._rand
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. 
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T
    
    Q = randomized_range_finder(M, mkl_dot, size=n_random, n_iter=n_iter, random_state=random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M, mkl_dot)
    
    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = svd(B, full_matrices=False)
    
    del B
    U = dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]


def svd_flip(u, v, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = argmax(np.abs(u), axis=0)
        signs = sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = argmax(np.abs(v), axis=1)
        signs = sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, newaxis]
    return u, v


def test_truncation(T, trunc_list, mkl_dot=True, display=True, n_iter=2):
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
        Ul, sigma_l, Vlt = randomized_svd(Tl, low_rank, mkl_dot, n_iter=n_iter)
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
    