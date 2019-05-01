"""
 Compression Module
 ==================
 This module is responsible for all routines related to the compression of tensors, which ammounts to computing its MLSVD 
and truncating it.
"""

# Python modules
import numpy as np
from numpy import eye, ones, empty, prod, float64, int64, copy, dot, sort, argsort, sqrt
from numpy.linalg import eigh, svd
from numpy.random import randint
import numpy as np
import sys

# Tensor Fox modules
import Auxiliar as aux
import Conversion as cnv
import Critical as crt
import MultilinearAlgebra as mlinalg


def mlsvd(T, Tsize, r, options):
    """
    This function computes the full MLSVD of tensors of any order. The output is such that T = (U_1,...,U_L)*S, and UT is 
    the list of the transposes of U.
    """

    # INITIALIZE RELEVANT VARIABLES.

    # Extract all variable from the class of options.
    trunc_dims = options.trunc_dims
    level = options.level
    display = options.display

    # Set the other variables.
    dims = T.shape
    L = len(dims)

    if L == 3:
        return trimlsvd(T, Tsize, r, trunc_dims, level, display)

    U = []
    UT = []
    T1 = empty((dims[0], prod(dims)), dtype = float64)

    # Compute the SVD of all unfolding of T.
    for l in range(L):
        Tl = cnv.unfold(T, l+1, dims)
        if l == 0:
            T1 = copy(Tl)
        if Tl.shape[0] < prod(Tl.shape[1:]):
            sigma_l, Ul = eigh( dot(Tl, Tl.T) )
            U.append(Ul)
            UT.append(Ul.T)
        else:
            Ul, sigma_l, Vlh = svd(Tl.T, full_matrices=False)
            U.append(Ul)
            UT.append(Ul.T)
    
    # Compute (U_1^T,...,U_L^T)*T = S
    S = mlinalg.high_multilin_mult(UT, T, T1, dims)

    return U, UT, S


def trimlsvd(T, Tsize, r, trunc_dims, level, display): 
    """
    This function computes the High order singular value decomposition (MLSVD) of a tensor T. This decomposition is given 
    by T = (U1,U2,U3)*S, where U1, U2, U3 are orthogonal matrices, S is the central tensor and * is the multilinear 
    multiplication. The MLSVD is a particular case of the Tucker decomposition.

    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
        Norm of T.
    r: int
        Desired rank.
    trunc_dims: 0 or list of ints
    level: 0,1,2,3,4 or 5
    display: string         

    Outputs
    -------
    S: float 3-D ndarray
        The central tensor (possibly truncated).
    best_energy: float
        It is a value between 0 and 100 indicating how much of the energy was preserverd after truncating the central 
        tensor. If no truncation occurs (best_energy == 100), then we are working with the original central tensor, which 
        has the same size as T.
    R1, R2, R3: int
        S.shape = (R1, R2, R3)
    U1, U2, U3: float 2-D ndarrays
        U1.shape = (m, R1), U2.shape = (n, R2), U3.shape = (p, R3)
    sigma1, sigma2, sigma3: float 1-D arrays
        Each one of these array is an ordered list with the singular values of the respective unfolding of T. We have that 
        sigma1.size = R1, sigma2.size = R2, sigma3.size = R3.
    mlsvd_stop: int
        It is a integer between 0 and 7, indicating how the compression was obtained. Below we summarize the possible 
        situations.
        0: truncation is given manually by the user with trunc_dims.
        1: user choose level = 4 so there is no truncation to be done.
        2: when testing the truncations a big gap between singular values were detected and the program lock the size of 
        the truncation. 
        3: the program was unable to compress at the very first iteration. In this case the tensor singular values are 
        equal or almost equal. We stop the truncation process when this happens.
        4: when no truncation were detected in 90 % of the attempts we suppose the tensor is random or has a lot of noise. 
        In this case we take a small truncation based on the rank r. This verification is made only made at the first stage.
        5: the energy of the truncation is accepted for it is big enough (this 'big' depends on the level choice and the 
        size of the tensor).
        6: none of the previous conditions were satisfied and we keep the last truncation computed. This condition is only 
        possible at the second stage.
        7:  user choosed level = 4, i.e., to work with the compressed tensor without truncating it.
    """    

    # Compute dimensions of T.
    m, n, p = T.shape
    dims = (m, n, p)

    # Level = 5 means no truncation and no compression, in other words, the original tensor.
    if level == 5:
        mlsvd_stop = 1
        U1, U2, U3 = eye(m), eye(n), eye(p)
        sigma1, sigma2, sigma3 = ones(m), ones(n), ones(p)
        if display > 2:
            return T, 100, m, n, p, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, 0.0
        else:
            return T, 100, m, n, p, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop
        
    # Compute all unfoldings of T and its SVD's. 
    T1 = cnv.unfold(T, 1, dims)
    T2 = cnv.unfold(T, 2, dims)
    T3 = cnv.unfold(T, 3, dims)
    sigma1, sigma2, sigma3, U1, U2, U3 = unfoldings_svd(T1, T2, T3, m, n, p)

    # Level = 4 means to not truncate the compression, i.e., we use the central tensor if the MLSVD without truncating it.
    if level == 4:
        mlsvd_stop = 7
        R1, R2, R3 = sigma1.size, sigma2.size, sigma3.size
        UT = [U1.T, U2.T, U3.T]
        S = mlinalg.multilin_mult(UT, T, T1, dims) 
        if display > 2:
            S1 = cnv.unfold(S, 1, (R1, R2, R3))
            best_error = aux.compute_error(T, Tsize, S, S1, [U1, U2, U3], (R1, R2, R3))
            return S, 100, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        else:
            return S, 100, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop
                            
    # TRUNCATE SVD'S OF UNFOLDINGS

    # Specific truncation is given by the user.
    if type(trunc_dims) == list:
        mlsvd_stop = 0
        S_energy = np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2)
        R1, R2, R3 = trunc_dims 
        U1, U2, U3 = U1[:, :R1], U2[:, :R2], U3[:, :R3] 
        sigma1, sigma2, sigma3 = sigma1[:R1], sigma2[:R2], sigma3[:R3]
        UT = [U1.T, U2.T, U3.T]
        S = mlinalg.multilin_mult(UT, T, T1, dims)  
        best_energy = compute_energy(S_energy, sigma1, sigma2, sigma3) 
        if display > 2:
            S1 = cnv.unfold(S, 1, (R1, R2, R3))
            best_error = aux.compute_error(T, Tsize, S, S1, [U1, U2, U3], (R1, R2, R3))
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        else:
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # The original SVD factors may have extra information due to noise or numerical error. We clean this SVD by performing
    # a specialized truncation. 
    stage = 1
    S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, situation = clean_compression(T, T1, Tsize, T, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage)
                      
    # TRUNCATE THE TRUNCATION

    # If one of the conditions below is true we don't truncate again.
    if (R1, R2, R3) == dims or situation == 'random' or R1*R2*R3 < 10**4:
        if display > 2:
            S1 = cnv.unfold(S, 1, (R1, R2, R3))
            best_error = aux.compute_error(T, Tsize, S, S1, [U1, U2, U3], (R1, R2, R3))
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        else:
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # Sometimes the first truncation still is too large. To fix this we consider a second truncation over the first one.
    else:
        if level < 3:
            level += 1
        stage = 2
        best_energy2 = 100
        S, best_energy2, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, situation = clean_compression(T, T1, Tsize, S, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage)
        # The second energy is a fraction of the first one. To compare with the original MLSVD we update the energy accordingly.
        best_energy = best_energy*best_energy2/100

    # Compute error of compressed tensor.
    if display > 2:
        S1 = cnv.unfold(S, 1, (R1, R2, R3))
        best_error = aux.compute_error(T, Tsize, S, S1, [U1, U2, U3], (R1, R2, R3))
        return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        
    return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop


def clean_compression(T, T1, Tsize, S, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage):
    """
    This function try different threshold values to truncate the mlsvd. The conditions to accept a truncation are defined 
    by the parameter level. Higher level means harder constraints, which translates to bigger tensors after the truncation.

    Inputs
    ------
    T: float 3-D ndarray
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

    # Define constraints.
    energy_tol = set_constraints(m, n, p, level)

    # Initialize the best results in the case none of the truncations work.
    best_R1, best_R2, best_R3 = sigma1.size, sigma2.size, sigma3.size
    best_S = S[:best_R1, :best_R2, :best_R3]
    best_U1, best_U2, best_U3 = U1[:, :best_R1], U2[:, :best_R2], U3[:, :best_R3] 
    best_sigma1, best_sigma2, best_sigma3 = sigma1[:best_R1], sigma2[:best_R2], sigma3[:best_R3]
          
    # Initialize relevant constants and arrays.
    dims = (m, n, p)
    count = 0
    num_cuts = max(100, int((m+n+p)/3))
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
        
    # Create arrays with the points where to truncate in each array of singular values. 
    # These 'points of cut' are randomly generated.   
    cut1, cut2, cut3 = generate_cuts(sigma1, sigma2, sigma3, num_cuts, r)
       
    # START ITERATIONS.

    for i in range(num_cuts):        
        # UPDATES.

        if sigma1_settled == False:
            sigma1_new, R1_new, U1_new = update_compression(sigma1, U1, sigma1[cut1[i]])
        if sigma2_settled == False:
            sigma2_new, R2_new, U2_new = update_compression(sigma2, U2, sigma2[cut2[i]])
        if sigma3_settled == False:
            sigma3_new, R3_new, U3_new = update_compression(sigma3, U3, sigma3[cut3[i]]) 
        
        # If the difference between two consecutive singular values is very large, we keep the current value for the rest 
        # of the iterations. 
        if i > 1:      
            sigma1_new, R1_new, U1_new, sigma1_settled = check_jump(sigma1_new, R1_new, U1_new, sigma1, U1, cut1, sigma1_settled, i)             
            sigma2_new, R2_new, U2_new, sigma2_settled = check_jump(sigma2_new, R2_new, U2_new, sigma2, U2, cut2, sigma2_settled, i)            
            sigma3_new, R3_new, U3_new, sigma3_settled = check_jump(sigma3_new, R3_new, U3_new, sigma3, U3, cut3, sigma3_settled, i)   
                    
        # Compute energy of compressed tensor.
        total_energy = compute_energy(S_energy, sigma1_new, sigma2_new, sigma3_new) 
                    
        # STOPPING CONDITIONS.

        # If all three cuts stopped iterating, we are done.
        if sigma1_settled == True and sigma2_settled == True and sigma3_settled == True:
            mlsvd_stop = 2
            best_energy = total_energy
            best_R1, best_R2, best_R3 = R1_new, R2_new, R3_new
            best_sigma1, best_sigma2, best_sigma3 = sigma1_new, sigma2_new, sigma3_new
            best_U1, best_U2, best_U3 = U1_new, U2_new, U3_new
            UT = [best_U1.T, best_U2.T, best_U3.T]
            best_S = mlinalg.multilin_mult(UT, T, T1, dims) 
            return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, mlsvd_stop, situation 

        # If the proccess is unable to compress at the very first iteration, then the singular values are all equal or 
        # almost equal. In this case we can't truncate and just stop here.
        if (R1_new, R2_new, R3_new) == dims and i == 0:  
            mlsvd_stop = 3
            best_R1, best_R2, best_R3 = m, n, p
            best_U1, best_U2, best_U3 = eye(m), eye(n), eye(p)
            best_sigma1, best_sigma2, best_sigma3 = ones(m), ones(n), ones(p)
            return T, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, mlsvd_stop, situation              
            
        # If no truncation was detected through almost all iterations, we may assume this tensor is random or has a lot 
        # of random noise (although it is also possible that the energy stop condition is too restrictive for this case). 
        # When this happens the program just choose some small truncation to work with. A good idea is to suppose the 
        # chosen rank is close to correct and use it to estimate the truncation size.
        if stage == 1 and i == num_cuts-int(num_cuts/10): 
            mlsvd_stop = 4
            situation = 'random'
            best_R1, best_R2, best_R3 = min(m, r), min(n, r), min(p, r)
            best_U1, best_U2, best_U3 = U1[:, :best_R1], U2[:, :best_R2], U3[:, :best_R3] 
            best_sigma1, best_sigma2, best_sigma3 = sigma1[:best_R1], sigma2[:best_R2], sigma3[:best_R3]
            best_energy = compute_energy(S_energy, best_sigma1, best_sigma2, best_sigma3) 
            UT = [best_U1.T, best_U2.T, best_U3.T]
            best_S = mlinalg.multilin_mult(UT, T, T1, dims)
            return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, mlsvd_stop, situation  
        
        # CHECK QUALITY OF TRUNCATION.
        
        if total_energy > best_energy:                                   
            # Update best results.
            best_energy = total_energy
            best_R1, best_R2, best_R3 = R1_new, R2_new, R3_new
            best_sigma1, best_sigma2, best_sigma3 = sigma1_new, sigma2_new, sigma3_new
            best_U1, best_U2, best_U3 = U1_new, U2_new, U3_new
                                                                 
            # Check energy of truncation.
            if best_energy > energy_tol:
                mlsvd_stop = 5
                UT = [best_U1.T, best_U2.T, best_U3.T]
                best_S = mlinalg.multilin_mult(UT, T, T1, dims)
                return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, mlsvd_stop, situation 
    
        # Keep values of all dimensions to compare in the next iteration.
        R1_old, R2_old, R3_old = R1_new, R2_new, R3_new
  
    mlsvd_stop = 6
    best_S = S[:best_R1, :best_R2, :best_R3]
    best_U1, best_U2, best_U3 = U1[:, :best_R1], U2[:, :best_R2], U3[:, :best_R3] 
    best_sigma1, best_sigma2, best_sigma3 = sigma1[:best_R1], sigma2[:best_R2], sigma3[:best_R3]
    return best_S, best_energy, best_R1, best_R2, best_R3, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3, mlsvd_stop, situation


def update_compression(S, U, tol):
    """
    This function ia a subroutine for the search_compression1 function. It computes the compressions of S and U, given 
    some tolerance tol.
    """

    sigma = S[S >= tol]
    if np.sum(sigma) == 0:
        sigma = S
        R = U.shape[1]  
    else:
        R = sigma.size
        U = U[:, :R]
                                    
    return sigma, R, U


def compute_energy(S_energy, sigma1, sigma2, sigma3):
    """
    Compute energy of compressed tensor.
    """

    best_energy = 100*(np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2))/S_energy   
    return best_energy


def check_jump(sigma_new, R_new, U_new, sigma, U, cut, sigma_settled, i):
    """
    Search for big jumps between the singular values of the unfoldings. If the difference between two consecutive sigmas 
    is very large, we keep the current value for the rest of the iterations.
    """

    if (sigma[cut[i-1]] > 1e4*sigma[cut[i]]) and (sigma_settled == False):
        best_sigma, best_R, best_U = update_compression(sigma, U, sigma[cut[i]])
        sigma_settled = True
        return best_sigma, best_R, best_U, sigma_settled
    else:
        return sigma_new, R_new, U_new, sigma_settled

def set_constraints(m, n, p, level):
    """
    The level parameter is 0, 1, 2, 3 or 4. The larger is this value, the bigger is the threshold value of the energy to 
    stop truncating. Small level values means small truncations, and big level values means bigger truncations. In 
    particular, level = 4 means no truncation at all.
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
            energy_tol = 100 - 1e-7
        if 1e5 < val and val <= 1e6:
            energy_tol = 100 - 1e-5
        if 1e6 < val and val <= 1e7:
            energy_tol = 100 - 1e-3
        if 1e7 < val:
            energy_tol = 100 - 1e-1

    # Truncation is large.
    if level == 2:
        if val <= 1e5:
            energy_tol = 100 - 1e-9
        if 1e5 < val and val <= 1e6:
            energy_tol = 100 - 1e-7
        if 1e6 < val and val <= 1e7:
            energy_tol = 100 - 1e-5
        if 1e7 < val:
            energy_tol = 100 - 1e-3

    # Truncation is almost equal or equal to the original MLSVD.
    if level == 3:
        energy_tol = 100 - 1e-9

    return energy_tol


def generate_cuts(sigma1, sigma2, sigma3, num_cuts, r):
    """
    At iteration i of the function clean_compression, we will truncate the sigmas by considering only the singular values 
    bigger than cut[i]. Each cut is a random number between 0 and 100. This means we will take num_cut points of each 
    array of singular values.
    """

    sigma1 = sigma1[:min(r, sigma1.size)]
    sigma2 = sigma2[:min(r, sigma2.size)]
    sigma3 = sigma3[:min(r, sigma3.size)]

    if r > 1:
        cut1 = randint(1,sigma1.size, size=num_cuts)
        cut1 = sort(cut1)
        cut2 = randint(1,sigma2.size, size=num_cuts)
        cut2 = sort(cut2)
        cut3 = randint(1,sigma3.size, size=num_cuts)
        cut3 = sort(cut3)
    else:
        cut1 = ones(num_cuts, dtype = int64)
        cut2 = ones(num_cuts, dtype = int64)
        cut3 = ones(num_cuts, dtype = int64)
    
    return cut1, cut2, cut3 


def unfoldings_svd(T1, T2, T3, m, n, p):
    """
    Computes SVD's of all unfoldings, taking in account the sizes of the matrix in in order to make computations faster.
    """

    # SVD of unfoldings.
    if m < n*p:
        sigma1, U1 = eigh( dot(T1, T1.T) )
        # Clean noise and compute the actual singular values.
        sigma1[sigma1 <= 0] = 0
        sigma1 = sqrt(sigma1)
        Sigma1 = -sort(-sigma1)
        new_col_order = argsort(-sigma1)
        U1 = U1[:, new_col_order]
    else:
        u1, sigma1, v1h = svd(T1.T, full_matrices=False)
        sigma1[sigma1 <= 0] = 0
        Sigma1 = -sort(-sigma1)
        new_col_order = argsort(-sigma1)
        v1h = v1h.T
        U1 = v1h[:, new_col_order]

    if n < m*p:
        sigma2, U2 = eigh( dot(T2, T2.T) )
        sigma2[sigma2 <= 0] = 0
        sigma2 = sqrt(sigma2)
        Sigma2 = -sort(-sigma2)
        new_col_order = argsort(-sigma2)
        U2 = U2[:, new_col_order]
    else:
        u2, sigma2, v2h = svd(T2.T, full_matrices=False)
        sigma2[sigma2 <= 0] = 0
        Sigma2 = -sort(-sigma2)
        new_col_order = argsort(-sigma2)
        v2h = v2h.T
        U2 = v2h[:, new_col_order]

    if p < m*n:
        sigma3, U3 = eigh( dot(T3, T3.T) )
        sigma3[sigma3 <= 0] = 0
        sigma3 = sqrt(sigma3)
        Sigma3 = -sort(-sigma3)
        new_col_order = argsort(-sigma3)
        U3 = U3[:, new_col_order]
    else:
        u3, sigma3, v3h = svd(T3.T, full_matrices=False)
        sigma3[sigma3 <= 0] = 0
        Sigma3 = -sort(-sigma3)
        new_col_order = argsort(-sigma3)
        v3h = v3h.T
        U3 = v3h[:, new_col_order]

    return Sigma1, Sigma2, Sigma3, U1, U2, U3
