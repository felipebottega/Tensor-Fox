"""
Auxiliar Module
 
 This module is composed by minor functions, designed to work on very specific tasks. Some of them may be useful for the user but most of them are just some piece of another (and more important) function. Below we list all funtions presented in this module.
 
 - cond
 
 - consistency
 
 - line_search
 
 - multilin_mult
 
 - multirank_approx
 
 - refine
 
 - tens2matlab
 
 - _sym_ortho
 
 - update_damp
""" 


import numpy as np
import sys
import scipy.io
from scipy import sparse 
from scipy.sparse.linalg import svds
from numba import jit, njit, prange
import TensorFox as tf
import Construction as cnst
import Conversion as cnv
import Critical as crt


def cond(T, r):
    """
    The condition number of T with respect to some rank r is defined as
    k(T) = 1/sigma_min(Dres), where Dres is the derivative computed at
    (Lambda,X,Y,Z), the CPD for T of rank r. "sigma_min" stands for the
    least singular value of Dres. If this value is zero, we define
    k(T) = infinity.
    
    Inputs
    ------
    T: float 3-D ndarray
    r: int
    """
    
    # Compute dimensions of T.
    m, n, p = T.shape
    
    # Test consistency of dimensions and rank.
    consistency(r, m, n, p) 
    
    # Initialize row and column indexes for sparse matrix creation.
    Lambda, X, Y, Z, T_approx, rel_err, step_sizes_trunc, step_sizes_ref, errors_trunc, errors_ref = tf.cpd(T,r)
    res = np.zeros(m*n*p, dtype = np.float64)
    data, row, col, datat_id, colt = cnst.residual_derivative_structure(r, m, n, p)
    res = cnst.residual(res, T, Lambda, X, Y, Z, r, m, n, p)
    data = cnst.residual_derivative(data, Lambda, X, Y, Z, r, m, n, p)
    Dres = sparse.csr_matrix((data, (row, col)), shape=(m*n*p, r + r*(m+n+p)))
    
    U, S, Vt = svds(Dres, k=2, which='SM', maxiter = 1000)
    
    # In the case all values of S are nan, we assume Dres is singular.
    if np.prod(np.isnan(S)) == 1:
        return np.inf
    else:
        cond = 1/min(S[~np.isnan(S)])
        return cond


def consistency(r, m, n, p):
    # If some dimension is equal to 1, the user may just use classical SVD with numpy.
    # We won't consider these situations here. 
    if (m == 1) or (n == 1) or (p == 1):
        sys.exit('At least one dimension is equal to 1. This situation not supported by Tensor Fox.')
        
    # Consistency of rank value.
    if type(r) != int:
        sys.exit('Invalid rank value.')
        
    # Check valid interval for rank.
    if (r > min(m*n, m*p, n*p)) or (r < 1):
        value = str(min(m*n, m*p, n*p))
        msg = 'Rank r must be satisfy 1 <= r <= min(m*n, m*p, n*p) = ' + value + '.'
        sys.exit(msg)
        
    return


def line_search(T, T_aux, Lambda, X, Y, Z, x, y, r):
    """
    We use a very simple line search after each Gauss-Newton iteration to try
    to improve the accuracy of the minimum just computed with the lsmr function.
    Since y is a descent direction for the error function, we test several points
    x + alpha*y. Depending on the current relative error of the CPD, we can try 
    two strategies:
        1) test alpha varying between 0.01 and 1 when the error is bigger than 1
        2) test alpha varying between 1.01 and 10 when the error is smaller than 1
        
    Basically, we try to shorten the step if the error is too big, and we try to 
    stretch the step if the error is small. After several tests, we keep the best
    result.
    
    This is a temporary function, for it is costly, so I want to improve the results
    with a cheaper algorithm in the future.
    
    Inputs
    ------
    T: float 3-D ndarray
    x: float 1-D ndarray
    y: float 1-D ndarray
        y has the same shape as x. It is a small step in some direction. This step is
    computed with the lsmr function, and it is the principal part of the Gauss-Newton
    method.
    r: int
    
    Outputs
    -------
    best_alpha: float
        best_alpha is a positive number such that x + alpha*y gives the best CPD in the
    neighborhood of x + y.    
    """
    
    # Compute dimensions of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
            
    # Set initial conditions. 
    Lambda, X, Y, Z = cnv.x2CPD(x + y, Lambda, X, Y, Z, r, m, n, p)
    T_aux = cnv.CPD2tens(T_aux, Lambda, X, Y, Z, r)
    best_error = np.linalg.norm(T - T_aux)
    alpha = 1.0
    best_alpha = 1.0
    
    # Declare arrays.
    errors = np.zeros(100, dtype = np.float64)
    alphas = np.zeros(100, dtype = np.float64)
    
    if best_error/Tsize > 1.0:
        low = 0.01
        high = 1
    else:
        low = 1.01
        high = 10.0
    
    # Test different size steps with line search to improve the new point.
    for i in range(0,100):
        alpha = low + (high - low)*i/100 
        Lambda, X, Y, Z = cnv.x2CPD(x + alpha*y, Lambda, X, Y, Z, r, m, n, p)
        T_aux = cnv.CPD2tens(T_aux, Lambda, X, Y, Z, r)
        errors[i] = np.linalg.norm(T - T_aux)
        alphas[i] = alpha    
        # Choose the step y associated with the smallest error.
        if errors[i] < best_error:
                best_error = errors[i]
                best_alpha = alphas[i]
       
    return best_alpha


@jit(nogil=True,cache=True)
def multilin_mult(T, L, M, N, m, n, p):
    """
    This function computes (L,M,N)*T, the multilinear multiplication between
    (L,M,N) and T.
    
    Inputs
    ------
    T: float 3-D ndarray
    L: float 2-D ndarray with m columns
    M: float 2-D ndarray with n columns
    N: float 2-D ndarray with p columns
    m, n, p: int
        The dimensions of T.
    """
    
    # Test for consistency.
    if (L.shape[1] != m) or (M.shape[1] != n) or (N.shape[1] != p):
        sys.exit("Wrong shape given in multilinear multiplication function.")
        return 
    
    # Define new tensors.
    d1 = L.shape[0]
    d2 = M.shape[0]
    d3 = N.shape[0]
    LT = np.zeros((d1,n,p), dtype = np.float64)
    LMT = np.zeros((d1,d2,p), dtype = np.float64)
    LMNT = np.zeros((d1,d2,d3), dtype = np.float64)
    
    # Compute unfoldings and update the new tensors accordingly
    T1 = cnv.unfold(T ,m, n, p, 1)
    T1_new = np.dot(L,T1)
    for k in range(0,p):
        for j in range(0,n):
            LT[:,j,k] = T1_new[:,k*n + j]
     
    T2 = cnv.unfold(LT, d1, n, p, 2)
    T2_new = np.dot(M,T2)
    for k in range(0,p):
        for i in range(0,d1):
            LMT[i,:,k] = T2_new[:,k*d1 + i]
    
    T3 = cnv.unfold(LMT, d1, d2, p, 3)
    T3_new = np.dot(N,T3)
    for j in range(0,d2):
        for i in range(0,d1):
            LMNT[i,j,:] = T3_new[:,j*d1 + i]
            
    return LMNT


def multirank_approx(T, r1, r2, r3):
    """
    This function computes an approximation of T with multilinear rank = (r1,r2,r3).
    Truncation the central tensor of the HOSVD doesn't gives the best low multirank
    approximation, but gives very good approximations. 
    
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
    
    # Compute dimensions of T.
    m, n, p = T.shape
    
    # Compute the HOSVD of T.
    S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = tf.hosvd(T)
    U1 = U1[:,0:r1]
    U2 = U2[:,0:r2]
    U3 = U3[:,0:r3]
    
    # Trncate S to a smaller shape (r1,r2,r3) and construct the tensor T_approx = (U1,U2,U3)*S.
    S = S[0:r1,0:r2,0:r3]                
    T_approx = multilin_mult(S, U1, U2, U3, r1, r2, r3)
    
    return T_approx


def refine(S, Lambda, X, Y, Z, r, R1_trunc, R2_trunc, R3_trunc, maxiter=200, tol=1e-4, display='none'):
    """
    After the cpd function computes a CPD for T using the truncated S, this function 
    puts the CPD Lambda, X, Y, Z in the same space of S (no truncation anymore) and try
    to uses the previous solution as starting point to a new call to dGN. The idea is to
    obtain an even better solution from the previous one. This is considered a refinement 
    stage. Since the space now is bigger and we already have a reasonable solution (hopefully),
    the stopping criteria as more loose, so the program may stop earlier.
    
    Inputs
    ------
    S: float 3-D ndarray
        S is the central tensor, computed with the HOSVD of T. It has shape (R1,R2,R3).
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,R1_trunc)
    Y: float 2-D ndarray of shape (n,R2_trunc)
    Z: float 2-D ndarray of shape (p,R3_trunc)
    r: int
    R1_trunc, R2_trunc, R3_trunc: int
    maxiter: int
    tol: float
    display: string
    
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (R1,r)
    Y: float 2-D ndarray of shape (R2,r)
    Z: float 2-D ndarray of shape (R3,r)
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Absolute error of the computed approximating tensor at each iteration. 
    """
    
    R1, R2, R3 = S.shape
    
    # Consider larger versions of X, Y, Z filled with zeros at the end. 
    # They are made to fit S in the computations.
    X_aug = np.zeros((R1,r), np.float64)
    Y_aug = np.zeros((R2,r), np.float64)
    Z_aug = np.zeros((R3,r), np.float64)    
    X_aug[0:R1_trunc,:] = X
    Y_aug[0:R2_trunc,:] = Y
    Z_aug[0:R3_trunc,:] = Z
                
    x, S_approx, step_sizes, errors = tf.dGN(S, Lambda, X_aug, Y_aug, Z_aug, r, maxiter=maxiter, tol=tol, display=display)
    
    Lambda, X, Y, Z = cnv.x2CPD(x, Lambda, X, Y, Z, r, R1, R2, R3)
       
    return Lambda, X, Y, Z, step_sizes, errors


def tens2matlab(T):
    """This function constructs the unfolding matrix T1 of T and creates a matlab file 
    containing T1. When in matlab, we can just open this file and the matrix T_unf will
    be created automatically. To transform it in a tensor m x n x p, first initialize 
    m,n,p with the correct values, then run the following code:
    
    T = zeros(m,n,p);
    s = 1;
    for k=1:p
       for j=1:n 
           T(:,j,k) = T_unf(:,s);
           s = s+1;
       end
    end
    
    After that we have the same tensor to work with in matlab. 
    """
    
    # Compute dimensions of T.
    m, n, p = T.shape
    
    # Unfold the tensor.
    T1 = cnv.unfold(T, m, n, p, mode=1)
    
    # Save the unfolding in matlab format.
    scipy.io.savemat('T1.mat', dict(T_unf=T1))
    
    return


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    
    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py). This function is useful
    for the LSMR function.
    
    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    
    if b == 0:
        return np.sign(a), 0, np.abs(a)
    elif a == 0:
        return 0, np.sign(b), np.abs(b)
    elif np.abs(b) > np.abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def update_damp(damp, v, old_error, error, old_residualnorm, residualnorm):
    """Update rule of the damping parameter for the dGN function."""
 
    g = 2*(old_error - error)/(old_residualnorm - residualnorm)
    if g > 0:
        damp = damp*max(1/3, 1-(2*g-1)**3)
        v = 2
    else:
        damp = damp*v
        v = 2*v

    return damp, v


def normalize(Lambda, X, Y, Z, r):
    """ Normalize the columns of X, Y, Z and scale Lambda accordingly."""
    
    for l in range(0,r):
        xn = np.linalg.norm(X[:,l])
        yn = np.linalg.norm(Y[:,l])
        zn = np.linalg.norm(Z[:,l])
        Lambda[l] = xn*yn*zn*Lambda[l]
        X[:,l] = X[:,l]/xn
        Y[:,l] = Y[:,l]/yn
        Z[:,l] = Z[:,l]/zn
        
    return Lambda, X, Y, Z