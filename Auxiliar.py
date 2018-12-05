"""
Auxiliar Module
 
 This module is composed by minor functions, designed to work on very specific tasks. Some of them may be useful for the user but most of them are just some piece of another (and more important) function. Below we list all funtions presented in this module.
 
 - consistency
 
 - line_search

 - cpd_error

 - bissection 
 
 - multilin_mult
 
 - multirank_approx
 
 - refine
 
 - tens2matlab
 
 - _sym_ortho
 
 - update_damp

 - normalize

 - equalize
""" 


import numpy as np
import sys
import scipy.io
from numba import jit, njit, prange
import TensorFox as tf
import Construction as cnst
import Conversion as cnv


def consistency(r, m, n, p):
    """ This function checks some invalid cases before anything is done in the program. """

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


@njit(nogil=True)
def line_search(T, T_aux, X, Y, Z, X_low, Y_low, Z_low, X_high, Y_high, Z_high, x, y, m, n, p, r):
    """
    We use a very simple line search after each Gauss-Newton iteration to try
    to improve the accuracy of the minimum just computed with the lsmr function.
    Since y is a descent direction for the error function, the program search
    for the optimal point using a bissection strategy.
    x + alpha*y. 
    
    Inputs
    ------
    T, T_aux: float 3-D ndarray
    X, Y, Z, X_low, Y_low, Z_low, X_high, Y_high, Z_high: float 2-D ndarray
    x, y: float 1-D ndarray
        y has the same shape as x. It is a small step in some direction. This step is
    computed with the lsmr function, and it is the principal part of the Gauss-Newton
    method.
    m, n, p, r: int
    
    Outputs
    -------
    best_alpha: float
        best_alpha is a positive number such that x + alpha*y gives the best CPD in the
    neighborhood of x + y. 
    best_error: float
        Is the absolute error |T - T_approx(alpha)|, where T_approx(alpha) is the tensor
    obtained from x + alpha*y for the best alpha found.   
    """
    
    # Initial values.
    best_error = cpd_error(T, T_aux, x + y, X_high, Y_high, Z_high, m, n, p, r)
    best_alpha = 1.0
    alpha_low = 0.01
    alpha_high = 3.0
    
    # Compute error for alpha = 0.01.
    error_low = cpd_error(T, T_aux, x + alpha_low*y, X_low, Y_low, Z_low, m, n, p, r)
        
    if error_low < best_error:
        cuts = 7
        alpha_low = 0.01
        alpha_high = best_alpha
        error_high = best_error
                
    else:
        # Compute error for alpha = 3.
        error_high = cpd_error(T, T_aux, x + alpha_high*y, X_high, Y_high, Z_high, m, n, p, r)
        alpha_low = best_alpha
        error_low = best_error
                
        if best_error < error_high:
            cuts = 7
                                    
        else:
            # First make sure the errors associated with the points bewtween best_alpha and 3 are greater. 
            cuts = 3
            best_alpha, best_error, alpha_low, alpha_high, error_low, error_high = bissection(alpha_low, alpha_high, error_low, error_high, cuts, T, T_aux, X_low, Y_low, Z_low, X_high, Y_high, Z_high, x, y, m, n, p, r)
            # If best_alpha = 3, then we should look for further values of alpha. We set alpha_low = 3 and alpha_high = 30.
            if best_alpha == 3.0:
                cuts = 6
                alpha_low = alpha_high
                error_low = error_high
                alpha_high = 30.0
                # Compute error for alpha = 3.
                error_high = cpd_error(T, T_aux, x + alpha_high*y, X_high, Y_high, Z_high, m, n, p, r)
            # If best_alpha < 3, then we try to refine the value of alpha by looking more this interval.
            else:
                cuts = 4
                                           
    best_alpha, best_error, alpha_low, alpha_high, error_low, error_high = bissection(alpha_low, alpha_high, error_low, error_high, cuts, T, T_aux, X_low, Y_low, Z_low, X_high, Y_high, Z_high, x, y, m, n, p, r)  
                
    return best_alpha, best_error


@njit(nogil=True)
def cpd_error(T, T_aux, x, X_lh, Y_lh, Z_lh, m, n, p, r):
    """ 
    This function computes the error between the objective tensor T and the CPD
    associated to x. The 'lh' index stands for 'low or high', because this function
    computes the error in both cases.
    """
    
    X_lh, Y_lh, Z_lh = cnv.x2CPD(x, X_lh, Y_lh, Z_lh, m, n, p, r)
    T_aux = cnv.CPD2tens(T_aux, X_lh, Y_lh, Z_lh, m, n, p, r)
    error = np.sqrt(np.sum((T - T_aux)**2))
    
    return error


@njit(nogil=True)
def bissection(alpha_low, alpha_high, error_low, error_high, cuts, T, T_aux, X_low, Y_low, Z_low, X_high, Y_high, Z_high, x, y, m, n, p, r):
    """
    This function tries to find the best alpha between alpha_low and alpha_high using the bissection method. 

    Inputs
    ------
    alpha_low, alpha_high, error_low_ error_high: float
        error_low is the error associated with x + alpha_low*y. The same goes for error_high.
    cuts: int
        Number of cuts to be used in this bissection. It can vary depending on alpha_low and alpha_high.
    T, T_aux: float 3-D ndarray
    X_low, Y_low, Z_low, X_high, Y_high, Z_high: float 2-D array
        Matrices of the CPD associated with x + alpha*y, where 'alpha' is alpha_low or alpha_high.
    x, y: float
    m, n, p, r: int

    Outputs
    -------
    best_alpha, best_error, alpha_low, alpha_high, error_low, error_high: float
        best_alpha is in fact the best parameter alpha found, while best_error is the error associated
    with this parameter. alpha_low is the last least value of alpha in the bissection method, while
    alpha_high is the last greater value of alpha in the bissection method.  
    """    

    for i in range(0, cuts):
        if error_low < error_high:
            best_alpha = alpha_low
            best_error = error_low
            if i == cuts-1:
                break  
            alpha_high = (alpha_low + alpha_high)/2    
            error_high = cpd_error(T, T_aux, x + alpha_high*y, X_high, Y_high, Z_high, m, n, p, r)
        else:
            best_alpha = alpha_high
            best_error = error_high
            if i == cuts-1:
                break  
            alpha_low = (alpha_low + alpha_high)/2
            error_low = cpd_error(T, T_aux, x + alpha_low*y, X_low, Y_low, Z_low, m, n, p, r)
            
    return best_alpha, best_error, alpha_low, alpha_high, error_low, error_high


@jit(nogil=True)
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

    Outputs
    -------
    LMNT: float 3-D ndarray
        LMNT is the result of the multilinear multiplication (L,M,N)*T.
    """
    
    # Test for consistency.
    if (L.shape[1] != m) or (M.shape[1] != n) or (N.shape[1] != p):
        sys.exit("Wrong shape given in multilinear multiplication function.")
        return 
    
    # Define new tensors.
    d1 = L.shape[0]
    d2 = M.shape[0]
    d3 = N.shape[0]
    LT = np.zeros((d1, n, p), dtype = np.float64)
    LMT = np.zeros((d1, d2, p), dtype = np.float64)
    LMNT = np.zeros((d1, d2, d3), dtype = np.float64)
    
    # Compute unfoldings and update the new tensors accordingly
    T1 = cnv.unfold(T ,m, n, p, 1)
    T1_new = np.dot(L, T1)
    for k in range(0,p):
        for j in range(0,n):
            LT[:,j,k] = T1_new[:, k*n + j]
     
    T2 = cnv.unfold(LT, d1, n, p, 2)
    T2_new = np.dot(M, T2)
    for k in range(0,p):
        for i in range(0,d1):
            LMT[i,:,k] = T2_new[:, k*d1 + i]
    
    T3 = cnv.unfold(LMT, d1, d2, p, 3)
    T3_new = np.dot(N, T3)
    for j in range(0,d2):
        for i in range(0,d1):
            LMNT[i,j,:] = T3_new[:, j*d1 + i]
            
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


def refine(S, X, Y, Z, r, R1_trunc, R2_trunc, R3_trunc, maxiter=200, cg_lower=2, cg_upper=10, tol=1e-6, display='none'):
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
    X_aug = np.zeros((R1, r), np.float64)
    Y_aug = np.zeros((R2, r), np.float64)
    Z_aug = np.zeros((R3, r), np.float64)    
    X_aug[0:R1_trunc,:] = X
    Y_aug[0:R2_trunc,:] = Y
    Z_aug[0:R3_trunc,:] = Z
                
    x, step_sizes, errors, stop = tf.dGN(S, X_aug, Y_aug, Z_aug, r, maxiter=maxiter, cg_lower=cg_lower, cg_upper=cg_upper, tol=tol, display=display)
    
    X, Y, Z = cnv.x2CPD(x, X, Y, Z, R1, R2, R3, r)
       
    return X, Y, Z, step_sizes, errors, stop


def tens2matlab(T):
    """ 
    This function constructs the unfolding matrix T1 of T and creates a matlab file 
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


@njit(nogil=True)
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


@njit(nogil=True)
def update_damp(damp, old_error, error, residualnorm):
    """ Update rule of the damping parameter for the dGN function. """
 
    g = 2*(old_error - error)/(old_error - residualnorm)
        
    if g < 0.75:
        damp = damp/2
    elif g > 0.9:
        damp = 1.5*damp
    
    return damp


def normalize(X, Y, Z, r):
    """ Normalize the columns of X, Y, Z and scale Lambda accordingly. This function returns
    Lambda, X, Y, Z, where (X,Y,Z)*Lambda is a normalized CPD. """
    Lambda = np.zeros(r, dtype = np.float64)
    
    for l in range(0,r):
        xn = np.linalg.norm(X[:,l])
        yn = np.linalg.norm(Y[:,l])
        zn = np.linalg.norm(Z[:,l])
        Lambda[l] = xn*yn*zn
        X[:,l] = X[:,l]/xn
        Y[:,l] = Y[:,l]/yn
        Z[:,l] = Z[:,l]/zn
        
    return Lambda, X, Y, Z


@njit(nogil=True, parallel=True)
def equalize(X, Y, Z, r):
    """ 
    After a Gauss-Newton iteration we have an approximated CPD with factors 
    X_l ⊗ Y_l ⊗ Z_l. They may have very differen sizes and this can have effect
    on the convergence rate. To improve this we try to equalize their sizes by 
    introducing scalars a, b, c such that X_l ⊗ Y_l ⊗ Z_l = (a*X_l) ⊗ (b*Y_l) ⊗ (c*Z_l)
    and |a*X_l| = |b*Y_l| = |c*Z_l|. Notice that we must have a*b*c = 1.
    
    To find good values for a, b, c, we can search for critical points of the function 
    f(a,b,c) = (|a*X_l|-|b*Y_l|)^2 + (|a*X_l|-|c*Z_l|)^2 + (|b*Y_l|-|c*Z_l|)^2.
    Using Lagrange multipliers we find the solution 
        a = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|X_l|,
        b = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|Y_l|,
        c = (|X_l|*|Y_l|*|Z_l|)^(1/3)/|Z_l|.
    
    We can see that this solution satisfy the conditions mentioned above.
    """
    
    for l in prange(0, r):
        X_nr = np.linalg.norm(X[:,l])
        Y_nr = np.linalg.norm(Y[:,l])
        Z_nr = np.linalg.norm(Z[:,l])
        if (X_nr != 0) and (Y_nr != 0) and (Z_nr != 0) :
            numerator = (X_nr*Y_nr*Z_nr)**(1/3)
            X[:,l] = (numerator/X_nr)*X[:,l]
            Y[:,l] = (numerator/Y_nr)*Y[:,l]
            Z[:,l] = (numerator/Z_nr)*Z[:,l] 
            
    return X, Y, Z


def sort_dims(T, m, n, p):
    """
    "m = 0", "n = 1", "p = 2"
    """

    if m >= n and n >= p:
        ordering = [0,1,2]
        return T, ordering
  
    elif p >= n and n >= m:
        ordering = [2,1,0]
       
    elif n >= p and p >= m:
        ordering = [1,2,0]

    elif m >= p and p >= n:
        ordering = [0,2,1]

    elif n >= m and m >= p:
        ordering = [1,0,2]

    elif p >= m and m >= n:
        ordering = [2,0,1]
 
    # Define m_s, n_s, p_s such that T_sorted.shape == m_s, n_s, p_s.
    m_s, n_s, p_s = T.shape[ordering[0]], T.shape[ordering[1]], T.shape[ordering[2]]
    T_sorted = np.zeros((m_s, n_s, p_s), dtype = np.float64)
    # In the function sort_T, inv_sort is such that T_sorted[inv_sort[i,j,k]] == T[i,j,k].
    inv_sort = np.argsort(ordering)
    T_sorted = sort_T(T, T_sorted, ordering, inv_sort, m_s, n_s, p_s)      

    return T_sorted, ordering
   

@njit(nogil=True)
def sort_T(T, T_sorted, ordering, inv_sort, m, n, p):
    # id receives the current triple (i,j,k) at each iteration.
    idx = np.array([0,0,0])
    
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
               idx[0], idx[1], idx[2] = i, j, k
               T_sorted[i,j,k] = T[idx[inv_sort[0]], idx[inv_sort[1]], idx[inv_sort[2]]]
                              
    return T_sorted
        

def unsort_dims(X, Y, Z, U1, U2, U3, ordering):
    if ordering == [0,1,2]:
        return X, Y, Z, U1, U2, U3,

    elif ordering == [0,2,1]:        
        return X, Z, Y, U1, U3, U2

    elif ordering == [1,0,2]:        
        return Y, X, Z, U2, U1, U3

    elif ordering == [1,2,0]:        
        return Y, Z, X, U2, U3, U1

    elif ordering == [2,0,1]:        
        return Z, X, Y, U3, U1, U2

    elif ordering == [2,1,0]:        
        return Z, Y, X, U3, U2, U1


@jit(nogil=True)
def search_compression1(T, Tsize, S1, S2, S3, U1, U2, U3, m, n, p, r):
    """
    This function try different threshold values to truncate the hosvd. For
    eps = 0.99 initially and updating with the rule eps = eps^2, we forget the singular
    values smaller than eps*max(S1, S2, S3). If the truncation has a relative error 
    smaller than 1e-6, then we consider this truncation is good enough. 

    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
    S1, S2, S3: float 1-D ndarrays
        Each one of these array is an ordered list (ascendent) with the singular values of the respective
    unfolding.
    U1, U2, U3: float 2-D ndarrays
    m, n, p, r: int   

    Outputs
    -------
    best_err: float
        Relative error of the best truncation founded.
    best_R1, best_R2, best_R3: int
        Dimensions of the truncation.
    best_sigma1, best_sigma2, best_sigma3: float 1-D ndarrays
        Truncated versions od S1, S2, S3.
    best_U1, best_U2, best_U3: float 2-D ndarrays
        Truncated versions of U1, U2, U3.
    status: str
        'fail 1': At the first iteration the truncation is the original tensor. This means the tensor can't
    be truncated (at each iteration the truncation increases).
        'fail 2': At some iteration (after the first one) the truncation is the original tensor. For the same 
    reasons as before we don't need to keep iterating. 
        'success': The function found a truncation with relative error less than 1e-6. 
    """

    # Initialize the best results in the case none of the truncations work.
    best_R1, best_R2, best_R3 = m, n, p
    best_U1 = U1
    best_U2 = U2
    best_U3 = U3
    best_sigma1 = S1
    best_sigma2 = S2
    best_sigma3 = S3
       
    # Initialize relevant constants.
    S_energy = np.sum(best_sigma1**2) + np.sum(best_sigma2**2) + np.sum(best_sigma3**2)
    best_energy = 0.9
    const_max = max(S1[-1], S2[-1], S3[-1])
    eps = 0.99
    tol = 1
    best_err = 1
    status = 'fail 2'
   
    while tol > 1e-16:        
        # Updates.
        tol = eps*const_max
        sigma1, R1, U1_new = update_truncation1(S1, U1, tol)
        sigma2, R2, U2_new = update_truncation1(S2, U2, tol)
        sigma3, R3, U3_new = update_truncation1(S3, U3, tol) 
                                
        # Stopping conditions.
        if (R1, R2, R3) == (m, n, p):
            # If this proccess is unable to truncate at the very first iteration, we can conclude that the tensor has no compression at all.
            if eps == 0.99:
                status = 'fail 1'
                return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3, status
            else:
                status = 'fail 2'
            return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3, status

        # Update eps.
        eps = eps**2

        # Compute energy of compressed tensor.
        total_energy = 100*(np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2))/S_energy                            
                
        if total_energy > best_energy and r < min(R1*R2, R1*R3, R2*R3):
            # Compute error of compressed tensor.
            S = np.zeros((R1,R2,R3))
            S = multilin_mult(T, U1_new.transpose(), U2_new.transpose(), U3_new.transpose(), m, n, p)
            T_compress = multilin_mult(S, U1_new, U2_new, U3_new, R1, R2, R3)
            rel_err = np.linalg.norm(T - T_compress)/Tsize
            
            # Update best results only if the relative error of the compression is smaller than 1e-2.
            if rel_err < 1e-1:
                best_err = rel_err
                best_energy = total_energy
                best_R1, best_R2, best_R3 = R1, R2, R3
                best_sigma1 = sigma1
                best_sigma2 = sigma2
                best_sigma3 = sigma3
                best_U1 = U1_new
                best_U2 = U2_new
                best_U3 = U3_new
            
            # Stopping condition. 
            if best_err < 1e-3 or best_energy > 99.9:
                status = 'success'
                return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3, status 
                               
    return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3, status 


@jit(nogil=True)
def update_truncation1(S, U, tol):
    """
    This function ia a subroutine for the search_compression1 function. It computes the truncations
    of S and U, given some tolerance tol.
    """
    sigma = S[S > tol]
    if np.sum(sigma) == 0:
        sigma = S
        R = U.shape[1]  
    else:
        R = sigma.size
        U = U[:, U.shape[1]-R:]
                                    
    return sigma, R, U 


@jit(nogil=True)
def search_compression2(T, Tsize, S1, S2, S3, U1, U2, U3, m, n, p, r):
    """
    This function has the same inputs and outputs as the function search_compression1, with 
    the exception of the output status, which is not necessary here.

    This function try different values for the dimensions to truncate. For each dimension
    dim, the program evaluates four values, obtained by dividing the interval [2, dim] in
    four (almost) equal parts.
    In this part we will accept less precise truncations, since the previous search function 
    already tried the more precise ones. Therefore, if the truncation has a relative error 
    smaller than 0.5, then we consider this truncation is good enough.     
    """

    best_R1, best_R2, best_R3 = m, n, p
    best_U1 = U1
    best_U2 = U2
    best_U3 = U3
    best_sigma1 = S1
    best_sigma2 = S2
    best_sigma3 = S3
    S_energy = np.sum(best_sigma1**2) + np.sum(best_sigma2**2) + np.sum(best_sigma3**2)
    best_energy = 0.9
    best_err = 1
    idx_list1 = [2, 1 + int(0.33*m), 1 + int(0.66*m), m]
    idx_list2 = [2, 1 + int(0.33*n), 1 + int(0.66*n), n]
    idx_list3 = [2, 1 + int(0.33*p), 1 + int(0.66*p), p]
    
    for i in idx_list1:
        for j in idx_list2:
            for k in idx_list3:
                # Updates.
                sigma1, R1, U1_temp = update_truncation2(S1, U1, m, i)
                sigma2, R2, U2_temp = update_truncation2(S2, U2, n, j)
                sigma3, R3, U3_temp = update_truncation2(S3, U3, p, k)   

                # First stopping condition.  
                if (R1, R2, R3) == (m, n, p):
                    return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3    

                # Compute energy of compressed tensor.
                total_energy = 100*(np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2))/S_energy          
        
                if total_energy > best_energy and r < min(R1*R2, R1*R3, R2*R3):
                    # Compute error of compressed tensor.
                    S = np.zeros((R1,R2,R3))
                    S = multilin_mult(T, U1_temp.transpose(), U2_temp.transpose(), U3_temp.transpose(), m, n, p)
                    T_compress = multilin_mult(S, U1_temp, U2_temp, U3_temp, R1, R2, R3)
                    rel_err = np.linalg.norm(T - T_compress)/Tsize

                    # Update best results only if the relative error of the compression is smaller than 0.5.
                    if rel_err < 0.5:
                        best_err = rel_err
                        best_energy = total_energy
                        best_R1, best_R2, best_R3 = R1, R2, R3
                        best_sigma1 = sigma1
                        best_sigma2 = sigma2
                        best_sigma3 = sigma3
                        best_U1 = U1_temp
                        best_U2 = U2_temp
                        best_U3 = U3_temp
                    
                    # Second stopping condition.
                    if best_err < 0.25:
                        return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3

    return best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3


@jit(nogil=True)
def update_truncation2(S, U, dim, idx):
    """
    This function ia a subroutine for the search_compression1 function. It computes the truncations
    of S and U, given some index idx (which will be in the interval [2,dim]).
    """
    sigma = S[dim-idx:]
    R = sigma.size
    U = U[:, dim-R:]
                
    return sigma, R, U


def unfoldings_svd(T1, T2, T3, m, n, p):
    """
    Computes SVD's of all unfoldings Ti, taking in account the size of the matrix.
    """

    # SVD of unfoldings.
    if m < n*p:
        S1, U1 = np.linalg.eigh( np.dot(T1, T1.transpose()) )
        # Clean noise and compute the actual singular values.
        S1[S1 <= 0] = 0
        S1 = np.sqrt(S1)
    else:
        u1, s1, v1h = np.linalg.svd(T1.transpose(), full_matrices=False)
        S1 = np.sort(s1)
        new_col_order = np.argsort(s1)
        v1h = v1h.transpose()
        U1 = v1h[:, new_col_order]
        S1[S1 <= 0] = 0

    if n < m*p:
        S2, U2 = np.linalg.eigh( np.dot(T2, T2.transpose()) )
        S2[S2 <= 0] = 0
        S2 = np.sqrt(S2)
    else:
        u2, s2, v2h = np.linalg.svd(T2.transpose(), full_matrices=False)
        S2 = np.sort(s2)
        new_col_order = np.argsort(s2)
        v2h = v2h.transpose()
        U2 = v2h[:, new_col_order]
        S2[S2 <= 0] = 0

    if p < m*n:
        S3, U3 = np.linalg.eigh( np.dot(T3, T3.transpose()) )
        S3[S3 <= 0] = 0
        S3 = np.sqrt(S3)
    else:
        u3, s3, v3h = np.linalg.svd(T3.transpose(), full_matrices=False)
        S3 = np.sort(s3)
        new_col_order = np.argsort(s3)
        v3h = v3h.transpose()
        U3 = v3h[:, new_col_order]
        S3[S3 <= 0] = 0

    return S1, S2, S3, U1, U2, U3
