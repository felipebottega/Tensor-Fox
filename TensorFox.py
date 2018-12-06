"""
General Description
 
 *Tensor Fox* is a vast library of routines related to tensor problems. Since most tensor problems fall in the category of NP-hard problems [1], a great effort was made to make this library as efficient as possible. Some relevant routines and features of Tensor Fox are the following: 
 
 - Canonical polyadic decomposition (CPD)
 
 - High order singular value decomposition (HOSVD)
 
 - Multilinear rank
 
 - Rank estimate
 
 - Rank related statistics, including histograms
 
 - Rank related information about tensors and tensorial spaces
 
 - Convert tensors to matlab format
 
 - High performance with parallelism and GPU computation
 
 The CPD algorithm is based on the Damped Gauss-Newton method (dGN) with help of line search at each iteration in order to accelerate the convergence. We expect to make more improvements soon.
 
 All functions of Tensor Fox are separated in five categories: *Main*, *Construction*, *Conversion*, *Auxiliar*, *Display* and *Critical*. Each category has its own module, this keeps the whole project more organized. The main module is simply called *Tensor Fox*, since this is the module the user will interact to most of the time. The other modules have the same names as their categories. The construction module deals with constructing more complicated objects necessary to make computations with. The conversion module deals with conversions of one format to another. The auxiliar module includes minor function dealing with certain computations which are not so interesting to see at the main module. The display module has functions to display useful information. Finally, the critical module is not really a module, but a set of compiled Cython functions dealing with the more computationally demanding parts. We list all the functions and their respective categories below.
 
 **Main:** 
 
 - cpd
 
 - dGN 
 
 - hosvd
 
 - rank
 
 - stats
 
 - cg
 
 **Construction:**
 
 - residual
 
 - residual_entries
 
 - start_point
 
 - smart_random
 
 - smart_sample
 
 - assign_values
 
 - truncation
 
 - truncate1
 
 - truncate2
 
 **Conversion:**
 
 - x2cpd
 
 - cpd2tens
 
 - unfold

 - _unfold
 
 - foldback
 
 **Auxiliar:**
 
 - consistency
 
 - multilin_mult
 
 - multirank_approx
 
 - refine
 
 - tens2matlab
 
 - _sym_ortho
 
 - update_damp

 - normalize

 - equalize

 - sort_dims

 - sort_T

 - unsort_dims

 - search_compression1

 - update_truncation1

 - search_compression2

 - update_truncation2 
 
 **Display:**
 
 - showtens
 
 - infotens
 
 - infospace
 
 - rank1_plot
 
 - rank1
 
 **Critical:**
 
 - khatri_rao

 - gramians

 - hadamard

 - vec

 - vect

 - prepare_data

 - prepare_data_rmatvec

 - matvec

 - rmatvec

 - regularization

 - precond
 
 
 [1] C. J. Hillar and Lek-Heng Lim. Most Tensor Problems are NP-Hard. Journal of the ACM. 2013.

"""


import numpy as np
import sys
import scipy.io
import time
from decimal import Decimal
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import Construction as cnst
import Conversion as cnv
import Auxiliar as aux
import Display as disp
import Critical as crt


def cpd(T, r, energy=99.9, maxiter=200, cg_lower=2, cg_upper=10, tol=1e-6, init='smart_random', display='none', full_output=False):
    """
    Given a tensor T and a rank R, this function computes one approximated CPD of T 
    with rank R. The result is given in the form [Lambda, X, Y, Z], where Lambda is a 
    vector and X, Y, Z are matrices. They are such that 
    sum_(l=1)^r Lambda[l] * X(l) ⊗ Y(l) ⊗ Z(l), 
    where X(l) denotes the l-th column of X. The same goes for Y and Z.

    Inputs
    ------
    T: float 3-D ndarray
        Objective tensor in coordinates.
    r: int 
        The desired rank of the approximating tensor.
    energy: float
        The energy varies between 0 and 100. For more information about how this parameter
    works, check the functions 'truncation', 'trancate1' and 'truncate2'. Default is 99.9.
    maxiter: int
        Number of maximum iterations allowed in the Gauss-Newton function. Default is 200.
    tol: float
        Tolerance criterium to stop the iteration proccess of the Gauss-Newton function.
    Default is 1e-4.
    init: string
        This options is used to choose the initial point to start the iterations. For more
    information, check the function 'init'. 
    display: string
        This options is used to control how information about the computations are displayed
    on the screen. For more information, check the function 'dGN'.
    
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray with shape (m,r)
    Y: float 2-D ndarray with shape (n,r)
    Z: float 2-D ndarray with shape (p,r)
        Lambda,X,Y,Z are such that (X,Y,Z)*Lambda ~ T.
    T_approx: float 3-D ndarray
        The approximating tensor in coordinates.
    err: float
        The relative error |T - T_approx|/|T| of the approximation computed. 
    step_sizes_trunc: float 1-D ndarray 
        Distance between the computed points at each iteration of the truncation stage.
    step_sizes_ref: float 1-D ndarray 
        Distance between the computed points at each iteration of the refinement stage.
    errors_trunc: float 1-D ndarray 
        Absolute error of the computed approximating tensor at each iteration of the 
    truncation stage.
    errors_ref: float 1-D ndarray 
        Absolute error of the computed approximating tensor at each iteration of the 
    refinement stage.
    """ 
        
    # Compute dimensions and norm of T.
    m_orig, n_orig, p_orig = T.shape
    m, n, p = m_orig, n_orig, p_orig
    T_orig = np.copy(T)
    Tsize = np.linalg.norm(T)
            
    # Test consistency of dimensions and rank.
    aux.consistency(r, m, n, p) 

    # Change ordering of indexes to improve performance if possible.
    T, ordering = aux.sort_dims(T, m, n, p)
    m, n, p = T.shape
    
    # COMPRESSION STAGE
    
    if display != 'none':
        print('-------------------------------------------------------') 
        print('Computing HOSVD of T')
    
    # Compute compressed version of T with the HOSVD. We have that T = (U1,U2,U3)*S.
    S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = hosvd(T, Tsize, r)
    R1, R2, R3 = multi_rank
        
    if display != 'none':
        if (R1, R2, R3) == (m, n, p):
            print('    No compression detected')                        
        else:
            print('    Compression detected')
            print('    Compressing from',T.shape,'to',S.shape)
            # Computation of relative error associated with compression.
            T_compress = aux.multilin_mult(S, U1, U2, U3, R1, R2, R3)
            rel_error = np.linalg.norm(T - T_compress)/Tsize
            a = float('%.4e' % Decimal(rel_error))
            print('    Compression relative error =', a)
            
    # TRUNCATION STAGE       
        
    if display != 'none':
        print('-------------------------------------------------------') 
        print('Computing truncation')
    
    # Truncate S to obtain a small tensor S_trunc with less energy than S, but close to S.
    S_trunc, U1_trunc, U2_trunc, U3_trunc, best_energy, R1_trunc, R2_trunc, R3_trunc, rel_error = cnst.truncation(T, Tsize, S, U1, U2, U3, r, sigma1, sigma2, sigma3, energy)
    
    # Check if the truncation is valid (if truncate too much the problem becomes ill-posed).
    aux.consistency(r, R1_trunc, R2_trunc, R3_trunc) 
     
    if display != 'none':
        
        if (R1_trunc, R2_trunc, R3_trunc) == (R1, R2, R3):
            print('    No truncation detected')             
        else:
            print('    Truncation detected') 
            print('    Truncating from', S.shape, 'to', S_trunc.shape)
            a = float('%.4e' % Decimal(best_energy))
            print(a,'% of the energy was retained')
            a = float('%.4e' % Decimal(rel_error))
            print('    Truncation relative error =', a) 
            
    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start dGN.
    X, Y, Z, rel_error = cnst.start_point(T, Tsize, S_trunc, U1_trunc, U2_trunc, U3_trunc, r, R1_trunc, R2_trunc, R3_trunc, init)      

    if display != 'none':
        print('-------------------------------------------------------')        
        if init == 'fixed':
            print('Type of initialization: fixed')
        elif init == 'random':
            print('Type of initialization: random')
        else:
            print('Type of initialization: smart random')

        a = float('%.4e' % Decimal(rel_error))
        print('    Initial guess relative error =', a)   
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display != 'none':
        print('-------------------------------------------------------') 
        print('Computing truncated CPD of T')

    # Compute the approximated tensor in coordinates with the dGN method.
    x, step_sizes_trunc, errors_trunc, stop_trunc = dGN(S_trunc, X, Y, Z, r, maxiter=maxiter, cg_lower=cg_lower, cg_upper=cg_upper, tol=tol, display=display) 
    
    # Compute CPD of S_trunc, which shoud be close to the CPD of S.
    X, Y, Z = cnv.x2CPD(x, X, Y, Z, R1_trunc, R2_trunc, R3_trunc, r)
    
    # REFINEMENT STAGE
    
    if display != 'none':
        print('-------------------------------------------------------') 
        print('Computing refinement of solution') 
        
    # Refine this CPD to be even closer to the CPD of S.
    X, Y, Z, step_sizes_refine, errors_refine, stop_ref = aux.refine(S, X, Y, Z, r, R1_trunc, R2_trunc, R3_trunc, maxiter=maxiter, cg_lower=cg_lower, cg_upper=cg_upper, tol=tol, display=display)
    
    # FINAL WORKS

    # Go back to the original dimensions of T.
    X, Y, Z, U1_sort, U2_sort, U3_sort = aux.unsort_dims(X, Y, Z, U1, U2, U3, ordering)
       
    # Use the orthogonal transformations to obtain the CPD of T.
    X = np.dot(U1_sort,X)
    Y = np.dot(U2_sort,Y)
    Z = np.dot(U3_sort,Z)
    
    # Compute coordinate representation of the CPD of T.
    T_aux = np.zeros((m_orig, n_orig, p_orig), dtype = np.float64)
    T_approx = cnv.CPD2tens(T_aux, X, Y, Z, m_orig, n_orig, p_orig, r)
        
    # Compute relative error of the approximation.
    rel_error = np.linalg.norm(T_orig - T_approx)/Tsize

    # Normalize X, Y, Z to have column norm equal to 1.
    Lambda, X, Y, Z = aux.normalize(X, Y, Z, r)
    
    # Display final informations.
    if display != 'none':
        print('=======================================================')
        print('Final results')
        print('    Number of steps =',step_sizes_trunc.shape[0] + step_sizes_refine.shape[0])
        print('    Relative error =', rel_error)
        a = float( '%.3e' % Decimal(100*(1 - rel_error)) )
        print('    Accuracy = ', a, '%')
    
    if full_output:
        return Lambda, X, Y, Z, T_approx, rel_error, step_sizes_trunc, step_sizes_refine, errors_trunc, errors_refine, stop_trunc, stop_ref
    else:
        return Lambda, X, Y, Z


def dGN(T, X, Y, Z, r, maxiter=200, cg_lower=2, cg_upper=10, tol=1e-6, display='none'):
    """
    This function uses the Damped Gauss-Newton method to compute an approximation of T 
    with rank r. An initial point to start the iterations must be given. This point is
    described by the ndarrays Lambda, X, Y, Z.
    
    The Damped Gauss-Newton method is iterative, updating a point x at each iteration.
    The last computed x is gives an approximate CPD in flat form, and from this we have
    the components to form the actual CPD. This program also gives some additional 
    information such as the size of the steps (distance between each x computed), the 
    absolute errors between the approximate and target tensor, and the path of solutions 
    (the points x computed at each iteration are saved). 

    Inputs
    ------
    T: float 3-D ndarray
    Lambda: Float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)
    r: int. 
        The desired rank of the approximating tensor.
    maxiter: int
        Number of maximum iterations permitted. Default is 200 iterations.
    tol: float
        Tolerance criterium to stop the iteration proccess. Let S^(k) be the approximating 
    tensor computed at the k-th iteration an x^(k) be the point computed at the k-th 
    iteration. If we have |T - T_approx^(k)|/|T| < tol or |x^(k+1) - x^(k)| < tol, then 
    the program stops. Default is tol = 10**(-6).
    display: string
    
    Outputs
    -------
    x: float 1-D ndarray with r+3*r*n entries 
        Each entry represents the components of the approximating tensor in the CPD form.
    More precisely, x is a flattened version of the CPD, which is given by
    x = [Lambda[1],...,Lambda[r],X[1,1],...,X[m,1],...,X[1,r],...,X[m,r],Y[1,1],...,Z[p,r]].
    T_approx: float 3-D ndarray with m*n*p entries 
        Each entry represents the coordinates of the approximating tensor in coordinate form.
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration. 
    """  
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
    
    # INITIALIZE RELEVANT VARIABLES
    
    # error is the current absolute error of the approximation.
    error = np.inf
    # damp is the damping factos in the damping Gauss-Newton method.
    damp = 2.0
    stop = 3
                
    # INITIALIZE RELEVANT ARRAYS
    
    x = np.concatenate((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
    y = x
    step_sizes = np.zeros(maxiter)
    errors = np.zeros(maxiter)
    # T_aux is an auxiliary array used only to accelerate the CPD2tens function.
    T_aux = np.zeros((m, n, p), dtype = np.float64)
    # res is the array with the residuals (see the residual function for more information).
    res = np.zeros(m*n*p, dtype = np.float64)
    g = np.ones(r*(m+n+p), dtype = np.float64)
    y = np.zeros(r*(m+n+p), dtype = np.float64)

    # Prepare data to use in each Gauss-Newton iteration.
    data = crt.prepare_data(m, n, p, r)    
    data_rmatvec = crt.prepare_data_rmatvec(X, Y, Z, m, n, p, r)
        
    if display == 'full':
        print('    Iteration | Rel Error |  Damp  | #CG iterations ')
    
    # BEGINNING OF GAUSS-NEWTON ITERATIONS
    
    for it in range(0,maxiter):      
        # Update of all residuals at x. 
        res = cnst.residual(res, T, X, Y, Z, r, m, n, p)
                                        
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_error = error
        
        # cg_maxiter is the maximum number of iterations of the Conjugate Gradient. We obtain it randomly.
        cg_maxiter = np.random.randint(cg_lower + int(np.sqrt(it)), cg_upper + it)
                
        # Computation of the Gauss-Newton iteration formula to obtain the new point x + y, where x is the 
        # previous point and y is the new step obtained as the solution of min_y |Ay - b|, with 
        # A = Dres(x) and b = -res(x).         
        y, g, itn, residualnorm = cg(X, Y, Z, data, data_rmatvec, y, g, -res, m, n, p, r, damp, cg_maxiter)       
              
        # Update point obtained by the iteration.         
        x = x + y
             
        # Compute X, Y, Z and error.
        X, Y, Z = cnv.x2CPD(x, X, Y, Z, m, n, p, r)
        T_aux = cnv.CPD2tens(T_aux, X, Y, Z, m, n, p, r)
        error = np.linalg.norm(T - T_aux)
                        
        # Update damp and save relevant information about the current iteration.
        old_damp = damp
        damp = aux.update_damp(damp, old_error, error, residualnorm)
        step_sizes[it] = np.linalg.norm(x - old_x)   
        errors[it] = error
        
        # Show information about current iteration.
        if display == 'full':
            a = float('%.2e' % Decimal(old_damp))
            print('       ',it+1,'    | ', '{0:.5f}'.format(error/Tsize),' | ',a, ' | ', itn+1)
                         
        # After 3 iterations the program starts to verify if the difference between the previous and the current 
        # relative errors are smaller than tol, or if the infinity norm of the derivative g of the error at x is
        # smaller than tol^(1/2).
        if it >= 3:
            errors_diff = np.abs(errors[it] - errors[it-1])/Tsize
            if step_sizes[it] < tol:
                stop = 0
                break
            if errors_diff < tol:
                stop = 1
                break
            elif np.linalg.norm(g, np.inf) < np.sqrt(tol):
                stop = 2
                break
    
    # SAVE LAST COMPUTED INFORMATIONS
    
    step_sizes = step_sizes[0:it+1]
    errors = errors[0:it+1]
    
    return x, step_sizes, errors, stop


def hosvd(T, Tsize, r): 
    """
    This function computes the High order singular value decomposition (HOSVD) of a tensor T.
    This decomposition is given by T = (U1,U2,U3)*S, where U1, U2, U3 are orthogonal matrices,
    S is the central tensor and * is the multilinear multiplication. The HOSVD is a particular
    case of the Tucker decomposition.

    Inputs
    ------
    T: float 3-D ndarray

    Outputs
    -------
    S: float 3-D ndarray
        The central tensor.
    multirank: int 1-D ndarray
        Is the multilinear rank of T (also known as the Tucker rank).
    U1, U2, U3: float 2-D ndarrays
    sigma1, sigma2, sigma3: float 1-D arrays
        Each one of these array is an ordered list with the singular values of the respective 
    unfolding of T.
    """    

    # Compute dimensions of T.
    m, n, p = T.shape
        
    # Compute all unfoldings of T. 
    T1 = cnv.unfold(T, m, n, p, 1)
    T2 = cnv.unfold(T, m, n, p, 2)
    T3 = cnv.unfold(T, m, n, p, 3)
    
    # Compute SVD of unfoldings.
    S1, S2, S3, U1, U2, U3 = aux.unfoldings_svd(T1, T2, T3, m, n, p)
                    
    # Truncate SVD'S OF ALL UNFOLDINGS
    
    # First we try to truncate the SVD's by deleting some small singular values. This approach favors compression
    # with small errors. Any compression with relative error smaller than 1e-6 will be accepted.
    best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3, status = aux.search_compression1(T, Tsize, S1, S2, S3, U1, U2, U3, m, n, p, r)
    
    # If the first search failed, we still can try to construct some artificial truncations. Notice that now we are
    # accepting compressions with large errors. Any compression with relative error smaller than 0.5 will be accepted.
    if status == 'fail 2':
         best_err, best_R1, best_R2, best_R3, best_sigma1, best_sigma2, best_sigma3, best_U1, best_U2, best_U3 = aux.search_compression2(T, Tsize, S1, S2, S3, U1, U2, U3, m, n, p, r)

    
    # CENTRAL TENSOR
    
    # Compute HOSVD of T, which is given by the identity S = (U1^T, U2^T, U3^T)*T.
    # S is the core tensor with size R1 x R2 x R3 and each Ui is an orthogonal matrix.
    multi_rank = np.array([best_R1, best_R2, best_R3])
    S = np.zeros((best_R1, best_R2, best_R3))
    S = aux.multilin_mult(T, best_U1.transpose(), best_U2.transpose(), best_U3.transpose(), m, n, p)
        
    return S, multi_rank, best_U1, best_U2, best_U3, best_sigma1, best_sigma2, best_sigma3


def rank(T, display='full'):
    """
    This function computes several approximations of T for r = 1...min(m*n, m*p, n*p). 
    These computations will be used to determine the (most probable) rank of T. The function 
    also returns an array `errors_per_rank` with the relative errors for each rank computed. 
    It is relevant to say that the rank r computed can also be the `border rank` of T, not the 
    actual rank. 

    The idea is that the minimum of |T - T_approx|, for each rank r, stabilizes when S 
    has the same rank as T. This function also plots the graph of the errors so the user 
    are able to visualize the moment when the error stabilizes.
    
    Inputs
    ------
    T: float 3-D ndarray
    display: string
        There are two options: 'full' (default) and 'none'. The first one shows and
    plot informations while the second shows nothing.
    
    Outputs
    -------
    final_rank: int
        The computed rank of T.
    errors_per_rank: float 1-D ndarray
        The error |T - T_approx| computed for each rank.    
    """
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
        
    # INITIALIZE RELEVANT VARIABLES
    
    # R is an upper bound for the rank.
    R = min(m*n, m*p, n*p)
    
    # INITIALIZE RELEVANT ARRAYS
    
    T_aux = np.zeros(T.shape, dtype = np.float64)
    error_per_rank = np.zeros(R)
    
    
    # Before the relevant loop for r=1...R, we compute the HOSVD of T and truncate it if possible.
    # This is exactly the first part of the cpd function. 

    # START THE PROCCESS OF FINDING THE RANK
    
    print('Start searching for rank')
    print('------------------------------------')
    print('Stops at r =',R,' or less')
    print()

    for r in range(1,R):  
        print('Testing r =',r)
    
        # COMPRESSION STAGE
         
        S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = hosvd(T, Tsize, r)
        R1, R2, R3 = multi_rank
            
        # TRUNCATION STAGE       
        
        S_trunc, U1_trunc, U2_trunc, U3_trunc, best_energy, R1_trunc, R2_trunc, R3_trunc, rel_error = cnst.truncation(T, Tsize, S, U1, U2, U3, r, sigma1, sigma2, sigma3, energy=99)
        
        # Generate starting point.
        X, Y, Z, rel_error = cnst.start_point(T, Tsize, S_trunc, U1_trunc, U2_trunc, U3_trunc, r, R1_trunc, R2_trunc, R3_trunc)      
        
        # Start Gauss-Newton iterations.
        x, step_sizes1, errors1, stop1 = dGN(S_trunc, X, Y, Z, r) 
        
        # Compute CPD of the point obtained.
        X, Y, Z = cnv.x2CPD(x, X, Y, Z, R1_trunc, R2_trunc, R3_trunc, r)
        
        # Refine solution.
        X, Y, Z, step_sizes2, errors2, stop2 = aux.refine(S, X, Y, Z, r, R1_trunc, R2_trunc, R3_trunc)
        
        # Put solution at the original space, where T lies.
        X = np.dot(U1,X)
        Y = np.dot(U2,Y)
        Z = np.dot(U3,Z)
        
        # Compute the solution in coordinates.
        T_approx = cnv.CPD2tens(T_aux, X, Y, Z, m, n, p, r)
    
        # Compute relative error of this approximation.
        err = np.linalg.norm(T - T_approx)/Tsize        
        error_per_rank[r-1] = err
        
        if r > 1:
            # Verification of the stabilization condition.
            if np.abs(error_per_rank[r-1] - error_per_rank[r-2]) < 1e-5:
                break
    
    # SAVE LAST INFORMATIONS
    
    error_per_rank = error_per_rank[0:r] 
    final_rank = np.argmin(error_per_rank)+1
        
    # DISPLAY AND PLOT ALL RESULTS
    
    print('------------------------------------')
    print('Estimated rank(T) =', final_rank)
    print('|T - T_approx|/|T| =', error_per_rank[final_rank - 1])
    
    if display != 'none':
        plt.plot(range(1,r+1), np.log10(error_per_rank))
        plt.plot(final_rank, np.log10(error_per_rank[final_rank - 1]), marker = 'o', color = 'k')
        plt.title('Rank trials')
        plt.xlabel('r')
        plt.ylabel(r'$\log_{10} \|T - S\|/|T|$')
        plt.grid()
        plt.show()
            
    return final_rank, error_per_rank


def stats(T, r, energy=99.9, maxiter=200, cg_lower=2, cg_upper=10, tol=1e-6, num_samples = 100):
    """
    This function makes several calls of the Gauss-Newton function with random initial 
    points. Each call turns into a sample to recorded so we can make statistics lates. 
    By defalt this functions takes 100 samples to analyze. The user may choose the number
    of samples the program makes, but the computational time may be very costly. 
    Also, the user may choose the maximum number of iterations and the tolerance to be 
    used in each Gauss-Newton function.

    The outputs plots with general information about all the trials. These 
    informations are the following:

    * The total time spent in each trial.

    * The number of steps used in each trial.

    * The relative error |T-S|/|T| obtained in each trial.

    Inputs
    ------
    T: float 3-D ndarray
    r: int 
        The desired rank of the approximating tensor.
    maxiter: int
    tol: float
    """
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
    
    # INITIALIZE RELEVANT ARRAYS
    
    times = np.zeros(num_samples)
    steps = np.zeros(num_samples)
    rel_errors = np.zeros(num_samples)
      
    # BEGINNING OF SAMPLING AND COMPUTING
    
    # At each run, the program computes a CPD for T with random guess for initial point.
    k = 1
    for trial in range(1, num_samples+1):            
        start = time.time()
        Lambda, X, Y, Z, T_approx, rel_error, step_sizes_trunc, step_sizes_ref, errors_trunc, errors_ref, stop_trunc, stop_ref = cpd(T, r, energy=energy, maxiter=maxiter, cg_lower=cg_lower, cg_upper=cg_upper, tol=tol, full_output=True)
               
        # Update the vectors with general information.
        times[trial-1] = time.time() - start
        steps[trial-1] = step_sizes_trunc.shape[0] + step_sizes_ref.shape[0]
        rel_errors[trial-1] = rel_error
        
        # Show progress status.
        if trial == int(k/10*num_samples):
            print(100*float(trial/num_samples), '%')
            k += 1
     
    # PLOT HISTOGRAMS
    
    [array,bins,patches] = plt.hist(times, 50)
    plt.xlabel('Seconds')
    plt.ylabel('Quantity')
    plt.title('Histogram of the total time of each trial')
    plt.show()

    [array,bins,patches] = plt.hist(steps, 50)
    plt.xlabel('Number of steps')
    plt.ylabel('Quantity')
    plt.title('Histogram of the number of steps of each trial')
    plt.show()

    [array,bins,patches] = plt.hist(np.log10(rel_errors), 50)
    plt.xlabel(r'$\log_{10} \|T - \tilde{T}\|/\|T\|$')
    plt.ylabel('Quantity')
    plt.title('Histogram of the log10 of the relative error of each trial')
    plt.show()

    return times, steps, rel_errors


@njit(nogil=True)
def cg(X, Y, Z, data, data_rmatvec, y, g, b, m, n, p, r, damp, cg_maxiter):
    """
    Conjugate gradient algorithm specialized to the tensor case.
    """

    # Give names to the arrays.
    Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, M, L, residual_cg, P, Q, z = data
    M_X, M_Y, M_Z, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z = data_rmatvec
    
    # Compute the values of all arrays.
    Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ = crt.gramians(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ)
    N_X, N_Y, N_Z = crt.update_data_rmatvec(X, Y, Z, M_X, M_Y, M_Z, N_X, N_Y, N_Z)
    L = crt.regularization(X, Y, Z, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, m, n, p, r)
    M = crt.precond(X, Y, Z, L, M, damp, m, n, p, r)
    
    y = 0*y
    
    g = crt.rmatvec(X, Y, b, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z, m, n, p, r)
    residual = M*g
    P = residual
    residualnorm = np.dot(residual, residual)
    residualnorm_new = 0.0
    alpha = 0.0
    beta = 0.0
    
    for itn in range(0, cg_maxiter):
        Q = M*P
        z = crt.matvec(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, Q, m, n, p, r) + damp*L*Q
        z = M*z
        alpha = residualnorm/np.dot(P.T, z)
        y += alpha*P
        residual = residual - alpha*z
        residualnorm_new = np.dot(residual, residual)
        beta = residualnorm_new/residualnorm
        residualnorm = residualnorm_new
        P = residual + beta*P
        if residualnorm < 1e-16:
            return M*y, g, itn, residualnorm   
        
    return M*y, g, itn+1, residualnorm


# Below we wrapped some functions which can be useful to the user. By doing this we just need to load the module TensorFox to do all the needed work.


def tens2matlab(T):    
    aux.tens2matlab(T)
    
    return


def CPD2tens(T_aux, X, Y, Z, m, n, p, r):
    T_aux = np.zeros((m,n,p), dtype = np.float64)
    T_aux = cnv.CPD2tens(T_aux, X, Y, Z, m, n, p, r)
    
    return T_aux


def residual_derivative_structure(r, m, n, p):
    data, row, col, datat_id, colt = cnst.residual_derivative_structure(r, m, n, p)
    
    return data, row, col, datat_id, colt


def showtens(T):
    disp.showtens(T)
    
    return


def infotens(T):
    disp.infotens(T)
    
    return


def infospace(m,n,p):
    disp.infospace(m,n,p)
    
    return


def rank1_plot(Lambda, X, Y, Z, r):
    disp.rank1_plot(Lambda, X, Y, Z, r)
    
    return
