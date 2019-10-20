"""
 Initialization Module
 ===================
 As we mentioned in the main module *Tensor Fox*, the module *Construction* is responsible for constructing the more 
complicated objects necessary to make computations. Between these objects we have the array of residuals, the derivative 
of the residuals, the starting points to begin the iterations and so on. 
"""

# Python modules
import numpy as np
from numpy import dot, empty, zeros, ones, float64, int64, arange, sqrt, inf, argmin, array
from numpy.linalg import norm
from numpy.random import randn, randint
import sys
from numba import jit, njit

# Tensor Fox modules
import Auxiliar as aux
import Conversion as cnv
import MultilinearAlgebra as mlinalg


def starting_point(T, Tsize, S, U1, U2, U3, R, R1, R2, R3, ordering, options):
    """
    This function generates a starting point to begin the iterations of the Gauss-Newton method. There are three
    options:
        - list: the user may give a list [X,Y,Z] with the three arrays to use as starting point.
        - 'random': each entry of X, Y, Z are generated by the normal distribution with mean 0 and variance 1.
        - 'smart_random': generates a random starting point with a method based on the MLSVD which always guarantee a 
           small relative error. Check the function 'smart_random' for more details about this method.
        - 'smart': works similar as the smart_random method, but this one is deterministic and generates the best rank-R
          approximation based on the MLSVD.
    
    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
    S: float 3-D ndarray with shape (R1, R2, R3)
    U1: float 2-D ndarrays with shape (R1, R)
    U2: float 2-D ndarrays with shape (R2, R)
    U3: float 2-D ndarrays with shape (R3, R)
    R, R1, R2, R3: int
    initialization: string or list
       Method of initialization. The five methods were described above.
    symm: bool
    display: int
    
    Outputs
    -------
    X: float 2-D ndarray of shape (R1, R)
    Y: float 2-D ndarray of shape (R2, R)
    Z: float 2-D ndarray of shape (R3, R)
    rel_error: float
        Relative error associate to the starting point. More precisely, it is the relative error between T and 
        (U1,U2,U3)*S_init, where S_init = (X,Y,Z)*I.
    """

    # Extract all variable from the class of options.
    initialization = options.initialization
    low, upp, factor = options.constraints
    c = options.factors_norm
    symm = options.symm
    display = options.display

    if type(initialization) == list:
        X = initialization[ordering[0]]
        Y = initialization[ordering[1]]
        Z = initialization[ordering[2]]
        X = dot(U1.T, X)
        Y = dot(U2.T, Y)
        Z = dot(U3.T, Z)

    elif initialization == 'random':
        X = randn(R1, R)
        Y = randn(R2, R)
        Z = randn(R3, R)

    elif initialization == 'smart_random':
        X, Y, Z = smart_random(S, R, R1, R2, R3)

    elif initialization == 'smart':
        X, Y, Z = smart(S, R, R1, R2, R3)

    else:
        sys.exit('Error with init parameter.')

    # Depending on the tensor, the factors X, Y, Z may have null entries. We want to
    # avoid that. The solution is to introduce very small random noise. 
    X, Y, Z = clean_zeros(S, X, Y, Z)

    # Make all factors balanced.
    X, Y, Z = cnv.equalize((X, Y, Z), R)

    # Apply additional transformations if requested.
    X, Y, Z = cnv.transform(X, Y, Z, low, upp, factor, symm, c)

    if display > 2 or display < -1:
        # Computation of relative error associated with the starting point given.
        S_init = empty((R1, R2, R3), dtype=float64)
        S_init = cnv.cpd2tens(S_init, [X, Y, Z], (R1, R2, R3))
        S1_init = cnv.unfold(S_init, 1, (R1, R2, R3))
        rel_error = mlinalg.compute_error(T, Tsize, S1_init, [U1, U2, U3], (R1, R2, R3))
        return X, Y, Z, rel_error

    return X, Y, Z


def smart_random(S, R, R1, R2, R3):
    """
    This function generates 1 + int(sqrt(R1*R2*R3)) samples of random possible initializations. The closest to S_trunc
    is saved. This method draws R points in S_trunc and generates a tensor with rank <= R from them. The distribution
    is such that it tries to maximize the energy of the sampled tensor, so the error is minimized. Although we are
    using the variables named as R1, R2, R3, remember they refer to R1_trunc, R2_trunc, R3_trunc at the main function.
    Since this function depends on the energy, it only makes sense using it when the original tensor can be compressed.
    If this is not the case, avoid using this function.
    
    Inputs
    ------
    S: 3-D float ndarray
    R: int
    R1, R2, R3: int
        The dimensions of the truncated tensor S.
    samples: int
        The number of tensors drawn randomly. Default is 100.
        
    Outputs
    -------
    X: float 2-D ndarray of shape (R1, R)
    Y: float 2-D ndarray of shape (R2, R)
    Z: float 2-D ndarray of shape (R3, R)
    """

    # Initialize auxiliary values and arrays.
    samples = 1 + int(sqrt(R1 * R2 * R3))
    best_error = inf
    Ssize = norm(S)

    # Start search for a good initial point.
    for sample in range(samples):
        X, Y, Z = smart_sample(S, R, R1, R2, R3)
        S_init = empty((R1, R2, R3), dtype=float64)
        S_init = cnv.cpd2tens(S_init, [X, Y, Z], (R1, R2, R3))
        rel_error = norm(S - S_init) / Ssize
        if rel_error < best_error:
            best_error = rel_error
            best_X, best_Y, best_Z = X, Y, Z

    return best_X, best_Y, best_Z


def smart_sample(S, R, R1, R2, R3):
    """
    We consider a distribution that gives more probability to smaller coordinates. This is because these are associated 
    with more energy. We choose a random number c1 in the integer interval [0, R1 + (R1-1) + (R1-2) + ... + 1]. 
    If 0 <= c1 < R1, we choose i = 1, if R1 <= c1 < R1 + (R1-1), we choose i = 2, and so on. The same goes for the other
    coordinates.
    Let S_{i_r,j_r,k_r}, r = 1...R, be the points chosen by this method. With them we form the tensor
    S_init = sum_{r=1}^R S_{i_r,j_r,k_r} e_{i_r} ⊗ e_{j_r} ⊗ e_{k_r}, which should be close to S_trunc.
    
    Inputs
    ------
    S: 3-D float ndarray
    r: int
    R1, R2, R3: int
    
    Ouputs
    ------
    X: float 2-D ndarray of shape (R1, R)
    Y: float 2-D ndarray of shape (R2, R)
    Z: float 2-D ndarray of shape (R3, R)
    """

    # Initialize arrays to construct initial approximate CPD.
    X = zeros((R1, R), dtype=float64)
    Y = zeros((R2, R), dtype=float64)
    Z = zeros((R3, R), dtype=float64)
    # Construct the upper bounds of the intervals.
    arr1 = R1 * ones(R1, dtype=int64) - arange(R1)
    arr2 = R2 * ones(R2, dtype=int64) - arange(R2)
    arr3 = R3 * ones(R3, dtype=int64) - arange(R3)
    high1 = np.sum(arr1)
    high2 = np.sum(arr2)
    high3 = np.sum(arr3)

    # Arrays with all random choices.
    C1 = randint(high1, size=R)
    C2 = randint(high2, size=R)
    C3 = randint(high3, size=R)

    # Update arrays based on the choices made.
    for r in range(R):
        X[:, r], Y[:, r], Z[:, r] = assign_values(S, X, Y, Z, R1, R2, R3, C1, C2, C3, arr1, arr2, arr3, r)

    return X, Y, Z


@jit(nogil=True)
def assign_values(S, X, Y, Z, R1, R2, R3, C1, C2, C3, arr1, arr2, arr3, r):
    """
    For each r = 1...R, this function constructs l-th one rank term in the CPD of the initialization tensor, which is of
    the form S[i,j,k]*e_i ⊗ e_j ⊗ e_k for some i,j,k choose through the random distribution described earlier.
    """

    for i in range(R1):
        if (np.sum(arr1[0:i]) <= C1[r]) and (C1[r] < np.sum(arr1[0:i + 1])):
            X[i, r] = 1
            break
    for j in range(R2):
        if (np.sum(arr2[0:j]) <= C2[r]) and (C2[r] < np.sum(arr2[0:j + 1])):
            Y[j, r] = 1
            break
    for k in range(R3):
        if (np.sum(arr3[0:k]) <= C3[r]) and (C3[r] < np.sum(arr3[0:k + 1])):
            Z[k, r] = 1
            break

    X[i, r] = S[i, j, k]

    return X[:, r], Y[:, r], Z[:, r]


@njit(nogil=True)
def smart(S, R, R1, R2, R3):
    """
    Construct a truncated version of S with the r entries with higher energy. Let S_{i_l,j_l,k_l}, r = 1...R, be the
    points chosen by this method. With them we form the tensor 
    S_init = sum_{r=1}^r S_{i_r,j_r,k_r} e_{i_r} ⊗ e_{j_r} ⊗ e_{k_r}, which should be close to S_trunc.
    
    Inputs
    ------
    S: 3-D float ndarray
    R: int
    R1, R2, R3: int
        The dimensions of the truncated tensor S.
            
    Outputs
    -------
    X: float 2-D ndarray of shape (R1, R)
    Y: float 2-D ndarray of shape (R2, R)
    Z: float 2-D ndarray of shape (R3, R)
    """

    # Find the entries of S with higher energy.
    largest = zeros(R, dtype=float64)
    indexes = zeros((R, 3), dtype=int64)
    for i in range(R1):
        for j in range(R2):
            for k in range(R3):
                if np.abs(S[i, j, k]) > np.min(np.abs(largest)):
                    idx = argmin(np.abs(largest))
                    largest[idx] = S[i, j, k]
                    indexes[idx, :] = array([i, j, k])

    # Initialize the factors X, Y, Z.
    X = zeros((R1, R), dtype=float64)
    Y = zeros((R2, R), dtype=float64)
    Z = zeros((R3, R), dtype=float64)

    # Use the entries computed previously to generates the factors X, Y, Z.
    for r in range(R):
        i, j, k = indexes[r, :]
        X[i, r] = largest[r]
        Y[j, r] = 1
        Z[k, r] = 1

    return X, Y, Z


@jit(nogil=True)
def clean_zeros(T, X, Y, Z):
    """
    Any null entry is redefined to be a small random number.
    """

    m, n, p = X.shape[0], Y.shape[0], Z.shape[0]
    R = X.shape[1]

    # Initialize the factors X, Y, Z with small noises to avoid null entries.
    for i in range(m):
        for r in range(R):
            if X[i, r] == 0.0:
                X[i, r] = 1e-16 * randn()
    for j in range(n):
        for r in range(R):
            if Y[j, r] == 0.0:
                Y[j, r] = 1e-16 * randn()
    for k in range(p):
        for r in range(R):
            if Z[k, r] == 0.0:
                Z[k, r] = 1e-16 * randn()

    return X, Y, Z
