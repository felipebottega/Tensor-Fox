"""
Auxiliar Module
 
 This module is composed by minor functions, designed to work on very specific tasks. Some of them may be useful for the user but most of them are just some piece of another (and more important) function. Below we list all funtions presented in this module.
 
 - consistency
 
 - multilin_mult
 
 - multirank_approx
 
 - tens2matlab
 
 - update_damp

 - normalize

 - denormalize

 - equalize

 - sort_dims

 - sort_T

 - unsort_dims

 - clean_compression

 - update_compression

 - compute_error

 - compute_energy

 - check_jump

 - set_constraints

 - generate_cuts

 - unfoldings_svd

 - make_info
""" 


import numpy as np
import sys
import scipy.io
from numba import jit, njit, prange
import TensorFox as tfx
import Construction as cnst
import Conversion as cnv
import Critical as crt


def consistency(r, m, n, p, symm):
    """ 
    This function checks some invalid cases before anything is done in the program. 
    """

    # If some dimension is equal to 1, the user may just use classical SVD with numpy.
    # We won't address these situations here. 
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

    if symm:
        if (m != n) or (m != p) or (n != p):
            msg = 'Symmetric tensors must have equal dimensions.'
            sys.exit(msg)
        
    return


@njit(nogil=True)
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
    
    # Define new tensors.
    d1, d2, d3 = L.shape[0], M.shape[0], N.shape[0]
    LT = np.zeros((d1, n, p), dtype = np.float64)
    LMT = np.zeros((d1, d2, p), dtype = np.float64)
    LMNT = np.zeros((d1, d2, d3), dtype = np.float64)
        
    # Compute unfoldings and update the new tensors accordingly
    T1 = cnv.unfold(T ,m, n, p, 1)
    T1_new = np.dot(L, T1)
    for k in range(0, p):
        for j in range(0, n):
            LT[:,j,k] = T1_new[:, k*n + j]
     
    T2 = cnv.unfold(LT, d1, n, p, 2)
    T2_new = np.dot(M, T2)
    for k in range(0, p):
        for i in range(0, d1):
            LMT[i,:,k] = T2_new[:, k*d1 + i]
    
    T3 = cnv.unfold(LMT, d1, d2, p, 3)
    T3_new = np.dot(N, T3)
    for j in range(0, d2):
        for i in range(0, d1):
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
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
    
    # Compute the HOSVD of T.
    trunc_dims = 0
    level = 1
    display = 0
    r = min(m, n, p)
    S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = tfx.hosvd(T, Tsize, r, trunc_dims, level, display)
    U1 = U1[:,0:r1]
    U2 = U2[:,0:r2]
    U3 = U3[:,0:r3]
    
    # Trncate S to a smaller shape (r1,r2,r3) and construct the tensor T_approx = (U1,U2,U3)*S.
    S = S[0:r1,0:r2,0:r3]                
    T_approx = multilin_mult(S, U1, U2, U3, r1, r2, r3)
    
    return T_approx


def tens2matlab(T):
    """ 
    This function creates a matlab file containing T and its dimensions. 
    """
    
    # Compute dimensions of T.
    m, n, p = T.shape
    
    # Save the unfolding in matlab format.
    scipy.io.savemat('T_data.mat', dict(T=T, m=m, n=n, p=p))
    
    return


@njit(nogil=True)
def update_damp(damp, old_error, error, residualnorm):
    """ 
    Update rule of the damping parameter for the dGN function. 
    """
 
    g = 2*(old_error - error)/(old_error - residualnorm)
        
    if g < 0.75:
        damp = damp/2
    elif g > 0.9:
        damp = 1.5*damp
    
    return damp


def normalize(X, Y, Z, r):
    """ 
    Normalize the columns of X, Y, Z to have unit column norm and scale Lambda accordingly.
    This function returns Lambda, X, Y, Z, where (X,Y,Z)*Lambda is a normalized CPD. 
    """

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


def denormalize(Lambda, X, Y, Z):
    """
    By undoing the normalization of the factors this function makes it unnecessary the use
    of the diagonal tensor Lambda. This is useful when one wants the CPD described only by
    the triplet (X, Y, Z).
    """

    R = Lambda.size
    X_new = np.zeros(X.shape)
    Y_new = np.zeros(Y.shape)
    Z_new = np.zeros(Z.shape)
    for r in range(R):
        if Lambda[r] >= 0:
            a = Lambda[r]**(1/3)
            X_new[:,r] = a*X[:,r]
            Y_new[:,r] = a*Y[:,r]
            Z_new[:,r] = a*Z[:,r]
        else:
            a = (-Lambda[r])**(1/3)
            X_new[:,r] = -a*X[:,r]
            Y_new[:,r] = a*Y[:,r]
            Z_new[:,r] = a*Z[:,r]
            
    return X_new, Y_new, Z_new


@njit(nogil=True)
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
    Consider the following identifications.
        "m = 0", "n = 1", "p = 2"
    We will use them to reorder the dimensions of the tensor, in such a way
    that we have m_new >= n_new >= p_new.
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
    """
    Subroutine of the function sort_dims. Here the program deals with
    the computationally costly part, which is the assignment of values
    to the new tensor.
    """

    # id receives the current triple (i,j,k) at each iteration.
    idx = np.array([0,0,0])
    
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                idx[0], idx[1], idx[2] = i, j, k
                T_sorted[i,j,k] = T[idx[inv_sort[0]], idx[inv_sort[1]], idx[inv_sort[2]]]
                              
    return T_sorted
        

def unsort_dims(X, Y, Z, U1, U2, U3, ordering):
    """
    Put the CPD factors and orthogonal transformations to the 
    original ordering of dimensions.
    """

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


def clean_compression(T, Tsize, S, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage):
    """
    This function try different threshold values to truncate the hosvd. The conditions to accept
    a truncation are defined by the parameter level. Higher level means harder constraints, which
    translates to bigger tensors after the truncation.

    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
    S: float 3-D ndarray
        Central tensor obtained by the hosvd.
    sigma1, sigma2, sigma3: float 1-D ndarrays
        Each one of these array is an ordered list (ascendent) with the singular values of the respective
    unfolding.
    U1, U2, U3: float 2-D ndarrays
    m, n, p, r: int   
    level: 0, 1, 2, 3
        0 means the stopping conditions are very weak, while 3  means the stopping conditions are very 
    hard. 
    stage: 1, 2
        1 means we are at the first stage of cleaning. At this stage we can stop the program for a specific
    condition. After that the function is called again to improve the truncation. The second time we have 
    stage == 2, so the mentioned condition won't be verified anymore.

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
    hosvd_stop: 0,1,2,3,4,5,6 or 7 
    situation: str
        There are three possibilities.
        1) situation == 'random' means the function stopped with random truncation (hosvd_stop == 4) 
        2) situation == 'overfit' means the function stopped with because of overfit (hosvd_stop == 5) 
        3) situation == 'ok' means the function stopped normally, without random truncation or overfit     
    """

    # Define constraints.
    energy_tol = set_constraints(m, n, p, level)

    # Initialize the best results in the case none of the truncations work.
    best_R1, best_R2, best_R3 = sigma1.size, sigma2.size, sigma3.size
    best_S = S[:best_R1, :best_R2, :best_R3]
    best_U1, best_U2, best_U3 = U1[:, :best_R1], U2[:, :best_R2], U3[:, :best_R3] 
    best_sigma1, best_sigma2, best_sigma3 = sigma1[:best_R1], sigma2[:best_R2], sigma3[:best_R3]
          
    # Initialize relevant constants.
    count = 0
    num_cuts = 1 + int((m+n+p)/3)
    S_energy = np.sum(best_sigma1**2) + np.sum(best_sigma2**2) + np.sum(best_sigma3**2)
    R1_old, R2_old, R3_old = m, n, p
    best_energy = 1
    situation = 'ok'
    sigma1_settled, sigma2_settled, sigma3_settled = False, False, False

    # In the first stage we fix any dimension too small compared to the others.
    if stage == 1:
        if 500*n < m:
            sigma2_settled, sigma3_settled = True, True
            sigma2_new, sigma3_new = best_sigma2, best_sigma3
            R2_new, R3_new = best_R2, best_R3
            U2_new, U3_new = best_U2, best_U3
        elif 500*p < n:
            sigma3_settled = True
            sigma3_new = best_sigma3
            R3_new = best_R3
            U3_new = best_U3
        
    # Create arrays with the points where to truncate in each array of singular values. These
    # 'points of cut' are generated randomly.   
    cut1, cut2, cut3 = generate_cuts(sigma1, sigma2, sigma3, num_cuts, r)
       
    # START ITERATIONS.

    for i in range(num_cuts):        
        # Updates.
        if sigma1_settled == False:
            sigma1_new, R1_new, U1_new = update_compression(sigma1, U1, sigma1[cut1[i]])
        if sigma2_settled == False:
            sigma2_new, R2_new, U2_new = update_compression(sigma2, U2, sigma2[cut2[i]])
        if sigma3_settled == False:
            sigma3_new, R3_new, U3_new = update_compression(sigma3, U3, sigma3[cut3[i]]) 
        
        # If the difference between two consecutive singular values is very large, we keep the
        # current value for the rest of the iterations. 
        if i > 1:      
            sigma1_new, R1_new, U1_new, sigma1_settled = check_jump(sigma1_new, R1_new, U1_new, sigma1, U1, cut1, sigma1_settled, i)             
            sigma2_new, R2_new, U2_new, sigma2_settled = check_jump(sigma2_new, R2_new, U2_new, sigma2, U2, cut2, sigma2_settled, i)            
            sigma3_new, R3_new, U3_new, sigma3_settled = check_jump(sigma3_new, R3_new, U3_new, sigma3, U3, cut3, sigma3_settled, i)   
                    
        # Compute energy of compressed tensor.
        total_energy = compute_energy(S_energy, sigma1_new, sigma2_new, sigma3_new) 
                    
        # STOPPING CONDITIONS.

        # If all three cuts stopped iterating, we are done.
        if sigma1_settled == True and sigma2_settled == True and sigma3_settled == True:
             hosvd_stop = 2
             best_energy = total_energy
             best_R1, best_R2, best_R3 = R1_new, R2_new, R3_new
             best_sigma1, best_sigma2, best_sigma3 = sigma1_new, sigma2_new, sigma3_new
             best_U1, best_U2, best_U3 = U1_new, U2_new, U3_new
             best_S = multilin_mult(T, best_U1.transpose(), best_U2.transpose(), best_U3.transpose(), m, n, p)
             return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation 

        # If the proccess is unable to compress at the very first iteration, then the tensor 
        # singular values are equal or almost equal. In this case we can't truncate.
        if (R1_new, R2_new, R3_new) == (m, n, p) and i == 0:  
            hosvd_stop = 3
            best_R1, best_R2, best_R3 = m, n, p
            best_U1, best_U2, best_U3 = np.eye(m), np.eye(n), np.eye(p)
            best_sigma1, best_sigma2, best_sigma3 = np.ones(m), np.ones(n), np.ones(p)
            return T, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation              
            
        # If no truncation was detected through almost all iterations, we may assume this tensor is random or
        # has a lot of random noise (although it is also possible that the energy stop condition is too restrictive
        # for this case). When this happens the program just choose some small truncation to work with. A good
        # idea is to suppose the chosen rank is correct and use it to estimate the truncation size.
        if stage == 1 and i == num_cuts-int(num_cuts/10): 
            hosvd_stop = 4
            situation = 'random'
            val1, val2, val3 = max(1, min(m, r) - 1), max(1, min(n, r) - 1), max(1, min(p, r) - 1)
            if sigma1_settled == False:
                best_sigma1, best_R1, best_U1 = update_compression(sigma1, U1, sigma1[val1])
            if sigma2_settled == False:
                best_sigma2, best_R2, best_U2 = update_compression(sigma2, U2, sigma2[val2])
            if sigma3_settled == False:
                best_sigma3, best_R3, best_U3 = update_compression(sigma3, U3, sigma3[val3]) 
            best_S = multilin_mult(T, best_U1.transpose(), best_U2.transpose(), best_U3.transpose(), m, n, p)
            best_energy = compute_energy(S_energy, best_sigma1, best_sigma2, best_sigma3) 
            return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation  
        
        # Stop the program due to a probable overfit. In this case we stop and return the previous valid values.
        if total_energy > 99.99 and r > min(R1_new*R2_new, R1_new*R3_new, R2_new*R3_new) and R1_new + R2_new + R3_new > 3:
            hosvd_stop = 5
            situation = 'overfit'
            best_energy = compute_energy(S_energy, best_sigma1, best_sigma2, best_sigma3)
            return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation
        
        # CHECK QUALITY OF TRUNCATION.
        
        # If no overfit occurred, check the quality of the compression.
        if total_energy > best_energy and r <= min(R1_new*R2_new, R1_new*R3_new, R2_new*R3_new) and R1_new + R2_new + R3_new > 3:                                   
            # Update best results.
            best_energy = total_energy
            best_R1, best_R2, best_R3 = R1_new, R2_new, R3_new
            best_sigma1, best_sigma2, best_sigma3 = sigma1_new, sigma2_new, sigma3_new
            best_U1, best_U2, best_U3 = U1_new, U2_new, U3_new
                                                                 
            # Inner stopping condition.

            if best_energy > energy_tol:
                hosvd_stop = 6
                best_S = multilin_mult(T, best_U1.transpose(), best_U2.transpose(), best_U3.transpose(), m, n, p)
                return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation 
    
        # Keep values of all dimensions to compare in the next iteration.
        R1_old, R2_old, R3_old = R1_new, R2_new, R3_new

    hosvd_stop = 7
    best_S = S[:best_R1, :best_R2, :best_R3]
    best_U1, best_U2, best_U3 = U1[:, :best_R1], U2[:, :best_R2], U3[:, :best_R3] 
    best_sigma1, best_sigma2, best_sigma3 = sigma1[:best_R1], sigma2[:best_R2], sigma3[:best_R3]
    return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, hosvd_stop, situation


def update_compression(S, U, tol):
    """
    This function ia a subroutine for the search_compression1 function. It computes the 
    compressions of S and U, given some tolerance tol.
    """
    sigma = S[S >= tol]
    if np.sum(sigma) == 0:
        sigma = S
        R = U.shape[1]  
    else:
        R = sigma.size
        U = U[:, :R]
                                    
    return sigma, R, U 


def compute_error(T, Tsize, S, R1, R2, R3, U1, U2, U3):
    """
    Compute relative error between T and (U1,U2,U3)*S.
    """

    T_compress = multilin_mult(S, U1, U2, U3, R1, R2, R3)
    rel_error = np.linalg.norm(T - T_compress)/Tsize 
    return rel_error


def compute_energy(S_energy, sigma1, sigma2, sigma3):
    """
    Compute energy of compressed tensor.
    """

    best_energy = 100*(np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2))/S_energy   
    return best_energy


def check_jump(sigma_new, R_new, U_new, sigma, U, cut, sigma_settled, i):
    """
    Search for big jumps between the singular values of the unfoldings. If the difference 
    between two consecutive sigmas is very large, we keep the current value for the rest 
    of the iterations.
    """

    if (sigma[cut[i-1]] > 1e4*sigma[cut[i]]) and (sigma_settled == False):
        best_sigma, best_R, best_U = update_compression(sigma, U, sigma[cut[i]])
        sigma_settled = True
        return best_sigma, best_R, best_U, sigma_settled
    else:
        return sigma_new, R_new, U_new, sigma_settled

def set_constraints(m, n, p, level):
    """
    The level parameter is 0, 1, 2, 3 or 4. The larger is this value, the bigger is the
    threshold value of the energy to stop truncating. Small level values means small
    truncations, and big level values means bigger truncations. In particular, level = 4
    means no truncation at all.
    """

    val = m*n*p

    # Truncation is small.
    if level == 0:
        if val <= 1e5:
            energy_tol = 99.9
        if 1e5 < val and val <= 1e6:
           energy_tol = 99
        if 1e6 < val and val <= 1e7:
            energy_tol = 98
        if 1e7 < val:
            energy_tol = 95
    
    # Normal truncation.
    if level == 1:
        if val <= 1e5:
            energy_tol = 99.9999999
        if 1e5 < val and val <= 1e6:
            energy_tol = 99.999
        if 1e6 < val and val <= 1e7:
            energy_tol = 99.9
        if 1e7 < val:
            energy_tol = 99

    # Truncation is large.
    if level == 2:
        if val <= 1e5:
            energy_tol = 99.999999999
        if 1e5 < val and val <= 1e6:
            energy_tol = 99.99999
        if 1e6 < val and val <= 1e7:
            energy_tol = 99.999
        if 1e7 < val:
            energy_tol = 99.9

    # Truncation is almost equal or equal to the original HOSVD.
    if level == 3:
        energy_tol = 99.999999999

    return energy_tol


def generate_cuts(sigma1, sigma2, sigma3, num_cuts, r):
    """
    At iteration i of the function clean_compression, we will truncate the sigmas by considering
    only the singular values bigger than cut[i]. Each cut is a random number between 0 and 100.
    This means we will take num_cut points of each array of singular values.
    """

    sigma1 = sigma1[:min(r, sigma1.size)]
    sigma2 = sigma2[:min(r, sigma2.size)]
    sigma3 = sigma3[:min(r, sigma3.size)]

    if r > 1:
        cut1 = np.random.randint(1,sigma1.size, size=num_cuts)
        cut1 = np.sort(cut1)
        cut2 = np.random.randint(1,sigma2.size, size=num_cuts)
        cut2 = np.sort(cut2)
        cut3 = np.random.randint(1,sigma3.size, size=num_cuts)
        cut3 = np.sort(cut3)
    else:
        cut1 = np.ones(num_cuts, dtype = np.int64)
        cut2 = np.ones(num_cuts, dtype = np.int64)
        cut3 = np.ones(num_cuts, dtype = np.int64)
    
    return cut1, cut2, cut3 


def unfoldings_svd(T1, T2, T3, m, n, p):
    """
    Computes SVD's of all unfoldings, taking in account the sizes of the matrix in
    in order to make computations faster.
    """

    # SVD of unfoldings.
    if m < n*p:
        sigma1, U1 = np.linalg.eigh( np.dot(T1, T1.transpose()) )
        # Clean noise and compute the actual singular values.
        sigma1[sigma1 <= 0] = 0
        sigma1 = np.sqrt(sigma1)
        Sigma1 = -np.sort(-sigma1)
        new_col_order = np.argsort(-sigma1)
        U1 = U1[:, new_col_order]
    else:
        u1, sigma1, v1h = np.linalg.svd(T1.transpose(), full_matrices=False)
        sigma1[sigma1 <= 0] = 0
        Sigma1 = -np.sort(-sigma1)
        new_col_order = np.argsort(-sigma1)
        v1h = v1h.transpose()
        U1 = v1h[:, new_col_order]

    if n < m*p:
        sigma2, U2 = np.linalg.eigh( np.dot(T2, T2.transpose()) )
        sigma2[sigma2 <= 0] = 0
        sigma2 = np.sqrt(sigma2)
        Sigma2 = -np.sort(-sigma2)
        new_col_order = np.argsort(-sigma2)
        U2 = U2[:, new_col_order]
    else:
        u2, sigma2, v2h = np.linalg.svd(T2.transpose(), full_matrices=False)
        sigma2[sigma2 <= 0] = 0
        Sigma2 = -np.sort(-sigma2)
        new_col_order = np.argsort(-sigma2)
        v2h = v2h.transpose()
        U2 = v2h[:, new_col_order]

    if p < m*n:
        sigma3, U3 = np.linalg.eigh( np.dot(T3, T3.transpose()) )
        sigma3[sigma3 <= 0] = 0
        sigma3 = np.sqrt(sigma3)
        Sigma3 = -np.sort(-sigma3)
        new_col_order = np.argsort(-sigma3)
        U3 = U3[:, new_col_order]
    else:
        u3, sigma3, v3h = np.linalg.svd(T3.transpose(), full_matrices=False)
        sigma3[sigma3 <= 0] = 0
        Sigma3 = -np.sort(-sigma3)
        new_col_order = np.argsort(-sigma3)
        v3h = v3h.transpose()
        U3 = v3h[:, new_col_order]

    return Sigma1, Sigma2, Sigma3, U1, U2, U3


def make_info(T_orig, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, gradients_main, gradients_refine, hosvd_stop, stop_main, stop_refine):
    class info:
        rel_error = np.linalg.norm(T_orig - T_approx)/Tsize
        step_sizes = [step_sizes_main, step_sizes_refine]
        errors = [errors_main, errors_refine]
        errors_diff = [np.concatenate(([1], np.abs(errors_main[0:-1] - errors_main[1:]))), np.concatenate(([1], np.abs(errors_refine[0:-1] - errors_refine[1:])))]
        gradients = [gradients_main, gradients_refine]
        stop = [hosvd_stop, stop_main, stop_refine]
        num_steps = np.size(step_sizes_main) + np.size(step_sizes_refine)
        accuracy = max(0, 100*(1 - rel_error))

        def stop_msg(self):
            # hosvd_stop message
            print('HOSVD stop:')
            if self.stop[0] == 0:
                print('0 - Truncation was given manually by the user.')
            if self.stop[0] == 1:
                print('1 - User choose level = 4 so no truncation was done.')
            if self.stop[0] == 2:
                print('2 - When testing the truncations a big gap between singular values were detected and the program lock the size of the truncation.')
            if self.stop[0] == 3:
                print('3 - The program was unable to compress at the very first iteration. In this case the tensor singular values are equal or almost equal. The program stops the truncation process when this happens.')
            if self.stop[0] == 4:
                print('4 - Tensor probably is random or has a lot of noise.')
            if self.stop[0] == 5:
                print('5 - Overfit was found and the user will have to try again or try a smaller rank.')
            if self.stop[0] == 6:
                print('6 - The energy of the truncation is accepted because it is big enough.')
            if self.stop[0] == 7:
                print('7 - None of the previous conditions were satisfied and we keep the last truncation computed. This condition is only possible at the second stage.')
           
            # stop_main message
            print()
            print('Main stop:')
            if self.stop[1] == 0:
                print('0 - Steps are too small.')
            if self.stop[1] == 1:
                print('1 - The improvement in the relative error is too small.')
            if self.stop[1] == 2:
                print('2 - The gradient is close enough to 0.')
            if self.stop[1] == 3:
                print('3 - The average of the last k relative errors is too small, where k = 1 + int(maxiter/10).')
            if self.stop[1] == 4:
                print('4 - Limit of iterations was been reached.')

            # stop_refine message
            print()
            print('Refinement stop:')
            if self.stop[2] == 0:
                print('0 - Steps are too small.')
            if self.stop[2] == 1:
                print('1 - The improvement in the relative error is too small.')
            if self.stop[2] == 2:
                print('2 - The gradient is close enough to 0.')
            if self.stop[2] == 3:
                print('3 - The average of the last k relative errors is too small, where k = 1 + int(maxiter/10).')
            if self.stop[2] == 4:
                print('4 - Limit of iterations was been reached.')
            if self.stop[2] == 5:
                print('5 - No refinement was performed.')

            msg = ' '
            return msg

    return info
