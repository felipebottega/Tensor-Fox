"""
General Description
 
 *Tensor Fox* is a vast library of routines related to tensor problems. Since most tensor problems fall in the category of NP-hard problems [1], a great effort was made to make this library as efficient as possible. Some relevant routines and features of Tensor Fox are the following: 
 
 - Canonical polyadic decomposition (CPD)
 
 - Multilinear singular value decomposition (MLSVD)
 
 - Multilinear rank
 
 - Rank estimate
 
 - Rank related statistics, including histograms
 
 - Rank related information about tensors and tensorial spaces
 
 - Convert tensors to matlab format
 
 - High performance with parallelism and GPU computation (soon)
 
 The CPD algorithm is based on the Damped Gauss-Newton method (dGN) with help of line search at each iteration in order to accelerate the convergence. We expect to make more improvements soon.
 
 All functions of Tensor Fox are separated in five categories: *Main*, *Construction*, *Conversion*, *Auxiliar*, *Display* and *Critical*. Each category has its own module, this keeps the whole project more organized. The main module is simply called *Tensor Fox*, since this is the module the user will interact to most of the time. The other modules have the same names as their categories. The construction module deals with constructing more complicated objects necessary to make computations with. The conversion module deals with conversions of one format to another. The auxiliar module includes minor function dealing with certain computations which are not so interesting to see at the main module. The display module has functions to display useful information. Finally, the critical module is not really a module, but a set of compiled Cython functions dealing with the more computationally demanding parts. We list all the functions and their respective categories below.
 
 **Main:** 
 
 - cpd
 
 - dGN 
 
 - mlsvd
 
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

 - smart
 
 **Conversion:**
 
 - x2cpd
 
 - cpd2tens
 
 - tens_entries
 
 - unfold
 
 - foldback

 - transform
 
 **Auxiliar:**
 
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

 - output_info

 - clean_zeros
 
 **Display:**
 
 - showtens
 
 - infotens
 
 - infospace
 
 - rank1_plot
 
 - rank1

 - rank_progress
 
 **Critical:**
 
 - kronecker

 - khatri_rao

 - khatri_rao_inner_computations

 - gramians

 - hadamard

 - vec

 - vect

 - prepare_data

 - prepare_data_rmatvec

 - update_data_rmatvec

 - matvec

 - rmatvec

 - regularization

 - precond

 **Dense**

 - jacobian
 
 - Hessian

 - precond

 - eig_dist

 - plot_structures

 - dense_cpd

 - dense_dGN

 - dense_output_info
 
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
import Dense as dns


def cpd(T, r, options=False):
    """
    Given a tensor T and a rank r, this function computes an approximated CPD of T with 
    rank r. The factors matrices are given in the form of a list [W^(1),...,W^(L)]. 
    These matrices are such that sum_(l=1)^r W[:,l]^(1) ⊗ ... ⊗ W[:,l]^(L) is an approximation 
    for T, where W[:,l]^(1) denotes the l-th column of W^(1). The same goes for the other matrices.

    Inputs
    ------
    T: float L-D ndarray
        Objective tensor in coordinates.
    r: int 
        The desired rank of the approximating tensor.
    options: class with the following parameters to be defined.
    	maxiter (refine): int
        	Number of maximum iterations allowed for the dGN function. Default is 200.
    	tol (refine): float
        	Tolerance criterium to stop the iteration proccess of the dGN function.
    		Default is 1e-12 for tol and 1e-10 for tol_refine.
    	init: string or list
        	This options is used to choose the initial point to start the iterations. For more
    		information, check the function start_point.
    	trunc_dims: int or list of ints
        	If trunc_dims is not 0, then it should be a list with three integers [R1,R2,R3] 
                such that 1 <= R1 <= m, 1 <= R2 <= n, 1 <= R3 <= p. The compressed tensor will have 
                dimensions (R1,R2,R3). Default is 0, which means 'automatic' truncation. This options 
		only serves to the tricpd function.
    	level:  0, 1, 2, 3, 4 or 5
         	The conditions to accept a truncation are defined by the parameter level. Higher means 
    		harder constraints, which means bigger dimensions. Default is 1. For more information 
		check the function set_constraints.
    	symm: bool
        	The user should set symm to True if the objetive tensor is symmetric, otherwise symm 
		is False. Default is False.
        low, upp, factor: floats
        	These values sets constraints to the entries of the tensor. Default for all of them is 0,
        which means no restriction. The parameter factor is auxiliar and influences how tight are the 
        projections into the interval [low, upp]. These parameters are experimental.
        trials: int
                This parameter is only used for tensor with order higher than 3. The computation of the 
                tensor train CPD requires the computation of several CPD of third order tensors. If only
                one of these CPD's is of low quality (divergence or local minima) then all effort is in
                vain. One work around is to compute several CPD'd and keep the best, for third order tensor.
                The parameter trials defines the maximum number of times we repeat the computation of each
                third order CPD. These trials stops when the relative error is less than 1e-4 or when the
                maximum number of trials is reached. Default is trials=1. 
    	display: -1, 0, 1, 2 or 3
        	This options is used to control how information about the computations are displayed
    		on the screen. The possible values are -1, 0, 1 (default), 2, 3. Notice that display=3 makes
    		the overall running time large since it will force the program to show intermediate errors
    		which are computationally costly. -1 is a special option for displaying minimal relevant 
                information for tensors with order higher then 3. We summarize the display options below.
                -1: display only the errors of each CPD computation and the final relevant information 
        	0: no information is printed
        	1: partial information is printed
        	2: full information is printed
        	3: full information + errors of truncation and starting point are printed 
    
    Outputs
    -------
    factors: list of float 2-D ndarrays with shape (dims[i], r) each
    T_approx: float L-D ndarray
        The approximating tensor in coordinates.
    outputs: list of classes
        Each tricpd and bicpd call gives a output class with all sort of information about the
    computations. The list 'outputs' contains all these classes.
    """ 

    # INITIAL PREPARATIONS

    # Compute dimensions and norm of T.
    dims = T.shape
    Tsize = np.linalg.norm(T)

    # Set options
    options = aux.make_class_options(options)
                   
    # Test consistency of dimensions and rank.
    aux.consistency(r, dims, options.symm)  
        
    # Verify if T is a third order tensor.
    L = len(dims)
    if L == 3:
        X, Y, Z, T_approx, output = tricpd(T, r, options)
        return [X, Y, Z], T_approx, output   

    
    # TENSOR TRAIN COMPUTATIONS
    
    if L > 3:
        factors, T_approx, outputs = highcpd(T, r, options)    

    
    # FINAL WORKS
    
    if options.display != 0:
        print()
        print('==========================================================================')
        print('==========================================================================')
        print('Final results')
        if options.refine:
            num_steps = 0
            for output in outputs:
                num_steps += output.num_steps
            print('    Number of steps =', num_steps)
        else:
            num_steps = 0
            for output in outputs:
                num_steps += output.num_steps-1
            print('    Number of steps =', num_steps)
        rel_error = np.linalg.norm(T - T_approx)/Tsize
        print('    Relative error =', rel_error)
        a = float( '%.6e' % Decimal(max(0, 100*(1 - rel_error))) )
        print('    Accuracy = ', a, '%')
    
    return factors, T_approx, outputs


def highcpd(T, r, options):
    """
    This function makes the calls in order to compute the tensor train of T and obtain the final 
    CPD from it. It is important to realize that this function is limited to tensor where each 
    one of its factors is a full rank matrix. In particular, the rank r must be smaller than all 
    dimensions of T.
    """     

    # Create relevant values
    dims = T.shape
    display = options.display
    max_trials = options.trials

    # Outputs is a list containing the output class of each cpd
    outputs = []

    # Compute cores of the tensor train of T
    G = TT_cores(T, r)
    L = len(G)    
    
    # List of CPD's
    cpd_list = []
    
    # Compute cpd of second core
    if display != 0:
        print('Total of', L-2, 'third order CPDs to be computed:')
        print('==========================================================================')
    best_error = np.inf
    for trial in range(max_trials):
        if display > 0:
            print()
            print('CPD 1')
        X, Y, Z, T_approx, output = tricpd(G[1], r, options)
        if output.rel_error < best_error:
            best_output = output
            best_error = output.rel_error
            best_X, best_Y, best_Z = X, Y, Z
            if best_error < 1e-4:
                break
    outputs.append(best_output)
    cpd_list.append([X,Y,Z])
    if display == -1:
        print('1 ) CPD error =', best_error)
    
    # Compute third order CPD's of cores G[2] to G[L-2]
    for l in range(2, L-1):
        best_error = np.inf
        prev_X = np.linalg.pinv((cpd_list[l-2][2]).T)
        for trial in range(max_trials):
            if display > 0:
                print()
                print('CPD', l)
            X, Y, Z, T_approx, output = bicpd(G[l], r, prev_X, options)
            if output.rel_error < best_error:
                best_output = output
                best_error = output.rel_error
                best_X, best_Y, best_Z = X, Y, Z
                if best_error < 1e-4:
                    break
        outputs.append(best_output)
        cpd_list.append([best_X, best_Y, best_Z])
        if display == -1:
            print(l,') CPD error =', best_error)
                
    # Compute of factors of T
    factors = []
    # First factor
    factors.append(np.dot(G[0], cpd_list[0][0]))
    # Factors 2 to L-2
    for l in range(0, L-2):
        factors.append(cpd_list[l][1])
    # Last factor
    factors.append( np.dot(G[-1].T, np.linalg.pinv(cpd_list[-1][-1].T)) )
    
    # Generate approximate tensor in coordinates 
    T_approx = np.zeros(T.shape)
    T_approx = cnv.cpd2tens(factors, dims)
    
    return factors, T_approx, outputs


def tricpd(T, r, options):
    """
    Given a tensor T and a rank r, this function computes an approximated CPD of T 
    with rank r. The result is given in the form [X, Y, Z]. These matrices are such 
    that sum_(l=1)^r X(l) ⊗ Y(l) ⊗ Z(l) is an approximation for T.
    X(l) denotes the l-th column of X. The same goes for Y(l) and Z(l).

    Inputs
    ------
    T: float 3-D ndarray
    r: int
    options: class
    
    Outputs
    -------
    X: float 2-D ndarray with shape (m,r)
    Y: float 2-D ndarray with shape (n,r)
    Z: float 2-D ndarray with shape (p,r)
        X, Y, Z are such that (X,Y,Z)*I ~ T.
    T_approx: float 3-D ndarray
        The approximating tensor in coordinates.
    output: class
        This class contains all information needed about the computations made. We summarize
    these informations below.
        rel_error: relative error |T - T_approx|/|T| of the approximation computed. 
        step_sizes: array with the distances between consecutive computed points at each iteration.
        errors: array with the absolute errors of the approximating tensor at each iteration.
        errors_diff: array with the differences between consecutive absolute errors.
        gradients: array with the gradient of the error function at each iteration. We expect that
    		   these gradients converges to zero as we keep iterating since the objetive point 
                   is a local minimum.
        stop: it is a list of three integers. The first one indicates how the compression was obtained.
    	      The second integer indicates why the dGN stopped at the first run, and the third integer 
              indicates why the dGN stopped at the second run (refinement stage). See the functions mlsvd 
              and dGN for more information.
        num_steps: the total number of steps (iterations) the dGN function used at the two runs.
        accuracy: the accuracy of the solution, which is defined by the formula 100*(1 - rel_error). 
                  0 means 0% of accuracy (worst case) and 100 means 100% of accuracy (best case). 
    """ 

    # Set options
    maxiter, tol, maxiter_refine, tol_refine, init, trunc_dims, level, refine, symm, low, upp, factor, recursive, display = aux.make_options(options)
        
    # Compute relevant variables and arrays.
    m_orig, n_orig, p_orig = T.shape
    m, n, p = m_orig, n_orig, p_orig
    T_orig = np.copy(T)
    Tsize = np.linalg.norm(T)
    tol = np.min([tol*m*n*p, 1e-1])
    tol_refine = np.min([tol_refine*m*n*p, 1e-1])
                   
    # Test consistency of dimensions and rank.
    aux.consistency(r, (m, n, p), symm) 

    # Change ordering of indexes to improve performance if possible.
    T, ordering = aux.sort_dims(T, m, n, p)
    m, n, p = T.shape
    
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('--------------------------------------------------------------------------') 
        print('Computing MLSVD of T')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U1,U2,U3)*S.
    if display == 3:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error = mlsvd(T, Tsize, r, trunc_dims, level, display)
    else:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop = mlsvd(T, Tsize, r, trunc_dims, level, display)

    # When the tensor is symmetric we want S to have equal dimensions. 
    if symm:
        R = min(R1, R2, R3)
        R1, R2, R3 = R, R, R
        S = S[:R, :R, :R]
        U1, U2, U3 = U1[:, :R], U2[:, :R], U3[:, :R]
          
    if display > 0:
        if level == 4:
            print('    Compression without truncation requested by user')
            print('    Compressing from', T.shape, 'to', S.shape)
        elif (R1, R2, R3) == (m, n, p):
            if level == 5:
                print('    No compression and no truncation requested by user')
                print('    Working with dimensions', T.shape) 
            else:
                print('    No compression detected')
                print('    Working with dimensions', T.shape)                         
        else:
            print('    Compression detected')
            print('    Compressing from', T.shape, 'to', S.shape)
            a = float('%.5e' % Decimal(best_energy))
            print('   ', a, '% of the energy was retained')
        if display == 3:
            a = float('%.5e' % Decimal(best_error))
            print('    Compression relative error =', a)

            
    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start dGN.
    if display == 3:
        X, Y, Z, rel_error = cnst.start_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, init, ordering, symm, low, upp, factor, display)  
    else:  
        X, Y, Z = cnst.start_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, init, ordering, symm, low, upp, factor, display)  
    
    if display > 0:
        print('--------------------------------------------------------------------------')        
        if type(init) == list:
            print('Type of initialization: user')
        else:
            print('Type of initialization:', init)
        if display == 3:
            a = float('%.4e' % Decimal(rel_error))
            print('    Initial guess relative error =', a)  
 
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('--------------------------------------------------------------------------')
        print('Computing CPD of T')
    
    # Compute the approximated tensor in coordinates with the dGN method.
    X, Y, Z, step_sizes_main, errors_main, gradients_main, stop_main = dGN(S, X, Y, Z, r, maxiter, tol, symm, low, upp, factor, display) 

    
    # REFINEMENT STAGE
    
    if refine:
        if display > 0:
            print('--------------------------------------------------------------------------') 
            print('Computing refinement of solution') 
        X, Y, Z, step_sizes_refine, errors_refine, gradients_refine, stop_refine = dGN(S, X, Y, Z, r, maxiter_refine, tol_refine, symm, low, upp, factor, display)
    else:
        step_sizes_refine = np.array([0])
        errors_refine = np.array([0]) 
        gradients_refine = np.array([0]) 
        stop_refine = 5 

    
    # FINAL WORKS
    
    # Go back to the original dimensions of T.
    X, Y, Z, U1_sort, U2_sort, U3_sort = aux.unsort_dims(X, Y, Z, U1, U2, U3, ordering)
           
    # Use the orthogonal transformations to obtain the CPD of T.
    X = np.dot(U1_sort,X)
    Y = np.dot(U2_sort,Y)
    Z = np.dot(U3_sort,Z)
    
    # Compute coordinate representation of the CPD of T.
    T_approx = cnv.cpd2tens([X, Y, Z], (m_orig, n_orig, p_orig))
        
    # Save and display final informations.
    output = aux.output_info(T_orig, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, gradients_main, gradients_refine, mlsvd_stop, stop_main, stop_refine)

    if display > 0:
        print('==========================================================================')
        print('Final results')
        if refine:
            print('    Number of steps =', output.num_steps)
        else:
            print('    Number of steps =', output.num_steps-1)
        print('    Relative error =', output.rel_error)
        a = float( '%.6e' % Decimal(output.accuracy) )
        print('    Accuracy = ', a, '%')
    
    return X, Y, Z, T_approx, output


def bicpd(T, r, prev_X, options):
    """
    Practically the same as tricpd, but this function keeps the first factor fixed during all the computations. 
    """

    # Set options
    maxiter, tol, maxiter_refine, tol_refine, init, trunc_dims, level, refine, symm, low, upp, factor, recursive, display = aux.make_options(options)
        
    # Compute relevant variables and arrays.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
    ordering = [0,1,2]
    tol = np.min([tol*m*n*p, 1e-1])
    tol_refine = np.min([tol_refine*m*n*p, 1e-1])
                       
    # Test consistency of dimensions and rank.
    aux.consistency(r, (m, n, p), symm) 
    
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('--------------------------------------------------------------------------') 
        print('Computing MLSVD of T')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U1,U2,U3)*S.
    if display == 3:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error = mlsvd(T, Tsize, r, trunc_dims, level, display)
    else:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop = mlsvd(T, Tsize, r, trunc_dims, level, display)

    # When the tensor is symmetric we want S to have equal dimensions. 
    if symm:
        R = min(R1, R2, R3)
        R1, R2, R3 = R, R, R
        S = S[:R, :R, :R]
        U1, U2, U3 = U1[:, :R], U2[:, :R], U3[:, :R]
          
    if display > 0:
        if level == 4:
            print('    Compression without truncation requested by user')
            print('    Compressing from', T.shape, 'to', S.shape)  
        elif (R1, R2, R3) == (m, n, p):
            if level == 5:
                print('    No compression and no truncation requested by user')
                print('    Working with dimensions', T.shape) 
            else:
                print('    No compression detected')
                print('    Working with dimensions', T.shape)                           
        else:
            print('    Compression detected')
            print('    Compressing from', T.shape, 'to', S.shape)
            a = float('%.5e' % Decimal(best_energy))
            print('   ', a, '% of the energy was retained')
        if display == 3:
            a = float('%.5e' % Decimal(best_error))
            print('    Compression relative error =', a)


    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start dGN.
    if display == 3:
        X, Y, Z, rel_error = cnst.start_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, init, ordering, symm, low, upp, factor, display)  
    else:  
        X, Y, Z = cnst.start_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, init, ordering, symm, low, upp, factor, display)

    # Discard the X computed in start_point and use the previous one. After that project this factor on the compressed space.
    X = np.dot(U1.T, prev_X)
    X = [X]
    
    if display > 0:
        print('--------------------------------------------------------------------------')        
        if type(init) == list:
            print('Type of initialization: fixed + user')
        else:
            print('Type of initialization: fixed +', init)
        if display == 3:
            S_init = cnv.cpd2tens([X[0], Y, Z], (R1, R2, R3))
            Ssize = np.linalg.norm(S)
            rel_error = aux.compute_error(T, Tsize, S_init, R1, R2, R3, U1, U2, U3)
            a = float('%.4e' % Decimal(rel_error))
            print('    Initial guess relative error =', a)
            
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('--------------------------------------------------------------------------') 
        print('Computing CPD of T')
    
    # Compute the approximated tensor in coordinates with the dGN method. 
    X, Y, Z, step_sizes_main, errors_main, gradients_main, stop_main = dGN(S, X, Y, Z, r, maxiter, tol, symm, low, upp, factor, display)
 
    
    # REFINEMENT STAGE
    
    if refine:
        if display > 0:
            print('--------------------------------------------------------------------------') 
            print('Computing refinement of solution') 
        X = np.dot(U1.T, prev_X)
        
        X = [X]
        X, Y, Z, step_sizes_refine, errors_refine, gradients_refine, stop_refine = dGN(S, X, Y, Z, r, maxiter_refine, tol_refine, symm, low, upp, factor, display)
    else:
        step_sizes_refine = np.array([0])
        errors_refine = np.array([0]) 
        gradients_refine = np.array([0]) 
        stop_refine = 5 

    
    # FINAL WORKS
    
    # Use the orthogonal transformations to obtain the CPD of T.
    Y = np.dot(U2, Y)
    Z = np.dot(U3, Z)
    
    # Compute coordinate representation of the CPD of T.
    T_approx = cnv.cpd2tens([prev_X, Y, Z], (m, n, p))
    
    # Save and display final informations.
    output = aux.output_info(T, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, gradients_main, gradients_refine, mlsvd_stop, stop_main, stop_refine)

    if display > 0:
        print('==========================================================================')
        print('Final results of bicpd')
        if refine:
            print('    Number of steps =', output.num_steps)
        else:
            print('    Number of steps =', output.num_steps-1)
        print('    Relative error =', output.rel_error)
        a = float( '%.6e' % Decimal(output.accuracy) )
        print('    Accuracy = ', a, '%')
    
    return X, Y, Z, T_approx, output


def dGN(T, X, Y, Z, r, maxiter, tol, symm, low, upp, factor, display):
    """
    This function uses the Damped Gauss-Newton method to compute an approximation of T 
    with rank r. An initial point to start the iterations must be given. This point is
    described by the arrays X, Y, Z.
    
    The Damped Gauss-Newton method is a iterative method, updating a point x at each 
    iteration. The last computed x is gives an approximate CPD in flat form, and from this 
    we have the components to form the actual CPD. This program also gives some additional 
    information such as the size of the steps (distance between each x computed), the 
    absolute errors between the approximate and target tensor, and the path of solutions 
    (the points x computed at each iteration are saved). 

    Inputs
    ------
    T: float 3-D ndarray
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)
    r: int. 
        The desired rank of the approximating tensor.
    maxiter: int
        Number of maximum iterations permitted. 
    tol: float
        Tolerance criterium to stop the iteration proccess. This value is used in more than
    one stopping criteria.
    symm: bool
    display: int
    
    Outputs
    -------
    x: float 1-D ndarray with r+3*r*n entries 
        Each entry represents the components of the approximating tensor in the CPD form.
    More precisely, x is a flattened version of the CPD, which is given by
    x = [X[1,1],...,X[m,1],...,X[1,r],...,X[m,r],Y[1,1],...,Z[p,r]].
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration. 
    gradients: float 1-D ndarray
        Gradient of the error function at each iteration.
    stop: 0, 1, 2, 3 or 4
        This value indicates why the dGN function stopped. Below we summarize the cases.
        0: step_sizes[it] < tol. This means the steps are too small.
        1: errors_diff < tol. This means the improvement in the error is too small.
        2: gradients[it] < np.sqrt(tol). This means the gradient is close enough to 0.
        3: np.mean(np.abs(errors[it-k : it] - errors[it-k-1 : it-1]))/Tsize < 10*tol. This
    means the average of the last k relative errors is too small. Keeping track of the averages
    is useful when the errors improvements are just a little above the threshold for a long time.
    We want them above the threshold indeed, but not too close for a long time. 
        4: limit of iterations reached.
    """  
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
        
    # INITIALIZE RELEVANT VARIABLES
    
    # error is the current absolute error of the approximation.
    error = np.inf
    best_error = np.inf
    # damp is the damping factos in the damping Gauss-Newton method.
    damp = np.mean(np.abs(T))
    old_damp = damp
    # Constant used in the third stopping condition.
    const = 1 + int(maxiter/10)
    stop = 4
    # Verify if X should be fixed or not.
    fix_X = False
    if type(X) == list:
        fix_X = True
        X_orig = np.copy(X[0])
        X = X[0]
                            
    # INITIALIZE RELEVANT ARRAYS
    
    x = np.concatenate((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
    y = x
    step_sizes = np.zeros(maxiter)
    errors = np.zeros(maxiter)
    gradients = np.zeros(maxiter)
    # Create auxiliary arrays.
    T_approx = np.zeros((m,n,p), dtype=np.float64)
    T_approx = cnv.cpd2tens([X, Y, Z], (m, n, p))
    # res is the array with the residuals (see the residual function for more information).
    res = np.zeros(m*n*p, dtype = np.float64)
    g = np.zeros(r*(m+n+p), dtype = np.float64)
    y = np.zeros(r*(m+n+p), dtype = np.float64)

    # Prepare data to use in each Gauss-Newton iteration.
    data = crt.prepare_data(m, n, p, r)    
    data_rmatvec = crt.prepare_data_rmatvec(m, n, p, r)
        
    if display > 1:
        print('    Iteration | Rel Error  | Rel Error Diff |     ||g||    | Damp| #CG iterations')
    
    # START GAUSS-NEWTON ITERATIONS
    
    for it in range(0,maxiter):      
        # Update of all residuals at x. 
        res = cnst.residual(res, T, T_approx, m, n, p)
                                        
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_error = error
               
        # cg_maxiter is the maximum number of iterations of the Conjugate Gradient. We obtain it randomly.
        cg_maxiter = np.random.randint(1 + it**0.4, 2 + it**0.9)
                             
        # Computation of the Gauss-Newton iteration formula to obtain the new point x + y, where x is the 
        # previous point and y is the new step obtained as the solution of min_y |Ay - b|, with 
        # A = Dres(x) and b = -res(x).         
        y, g, itn, residualnorm = cg(X, Y, Z, data, data_rmatvec, y, g, -res, m, n, p, r, damp, cg_maxiter, tol)       
              
        # Update point obtained by the iteration.         
        x = x + y
        
        # Compute X, Y, Z.
        X, Y, Z = cnv.x2cpd(x, X, Y, Z, m, n, p, r)
        X, Y, Z = cnv.transform(X, Y, Z, m, n, p, r, low, upp, factor, symm)
        if fix_X == True:
            X = np.copy(X_orig)
                                          
        # Compute error.
        T_approx = cnv.cpd2tens([X, Y, Z], (m, n, p))
        error = np.linalg.norm(T - T_approx)

        # Update best solution.
        if error < best_error:
            best_error = error
            best_X = np.copy(X)
            best_Y = np.copy(Y)
            best_Z = np.copy(Z)
                    
        # Update damp. 
        old_damp = damp
        damp = aux.update_damp(damp, old_error, error, residualnorm)
        
        # Save relevant information about the current iteration.
        step_sizes[it] = np.linalg.norm(x - old_x)   
        errors[it] = error
        gradients[it] = np.linalg.norm(g, np.inf)
        if it == 0:
            errors_diff = errors[it]/Tsize
        else:
            errors_diff = np.abs(errors[it] - errors[it-1])/Tsize
        
        # Show information about current iteration.
        if display > 1:
            a = float('%.2e' % Decimal(old_damp))
            print('       ',it+1,'    | ', '{0:.6f}'.format(error/Tsize), ' |   ', '{0:.6f}'.format(errors_diff), '   | ', '{0:.6f}'.format(gradients[it]), ' |', a, '|   ', itn)
                   
        # Stopping conditions.
        if it >= 3:
            if step_sizes[it] < tol:
                stop = 0
                break
            if errors_diff < tol:
                stop = 1
                break
            if gradients[it] < np.sqrt(tol):
                stop = 2
                break 
            # Let const=1 + int(maxiter/10). If the average of the last const error improvements is less than 10*tol, then 
            # we stop iterating. We don't want to waste time computing with 'almost negligible' improvements for long time.
            if it > const and it%const == 0: 
                if np.mean(np.abs(errors[it-const : it] - errors[it-const-1 : it-1]))/Tsize < 10*tol:
                    stop = 3
                    break  
            if error > Tsize/tol:
                stop = 6
                break 
    
    # SAVE LAST COMPUTED INFORMATIONS
    
    step_sizes = step_sizes[0:it+1]
    errors = errors[0:it+1]
    gradients = gradients[0:it+1]
    
    return best_X, best_Y, best_Z, step_sizes, errors, gradients, stop


def mlsvd(T, Tsize, r, trunc_dims, level, display): 
    """
    This function computes the High order singular value decomposition (MLSVD) of a tensor T.
    This decomposition is given by T = (U1,U2,U3)*S, where U1, U2, U3 are orthogonal matrices,
    S is the central tensor and * is the multilinear multiplication. The MLSVD is a particular
    case of the Tucker decomposition.

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
       It is a value between 0 and 100 indicating how much of the energy was preserverd after 
    truncating the central tensor. If no truncation occurs (best_energy == 100), then we are
    working with the original central tensor, which has the same size as T.
    R1, R2, R3: int
        S.shape = (R1, R2, R3)
    U1, U2, U3: float 2-D ndarrays
        U1.shape = (m, R1), U2.shape = (n, R2), U3.shape = (p, R3)
    sigma1, sigma2, sigma3: float 1-D arrays
        Each one of these array is an ordered list with the singular values of the respective 
    unfolding of T. We have that sigma1.size = R1, sigma2.size = R2, sigma3.size = R3.
    mlsvd_stop: int
        It is a integer between 0 and 7, indicating how the compression was obtained. Below we 
    summarize the possible situations.
        0: truncation is given manually by the user with trunc_dims.
        1: user choose level = 4 so there is no truncation to be done.
        2: when testing the truncations a big gap between singular values were detected and the
    program lock the size of the truncation. 
        3: the program was unable to compress at the very first iteration. In this case the 
    tensor singular values are equal or almost equal. We stop the truncation process when this
    happens.
        4: when no truncation were detected in 90 % of the attempts we suppose the tensor is
    random or has a lot of noise. In this case we take a small truncation based on the rank r.
    This verification is made only made at the first stage.
        5: overfit was found and the user will have to try again or try a smaller rank.
        6: the energy of the truncation is accepted for it is big enough (this 'big' depends
    on the level choice and the size of the tensor).
        7: none of the previous conditions were satisfied and we keep the last truncation 
    computed. This condition is only possible at the second stage.
    """    

    # Compute dimensions of T.
    m, n, p = T.shape
        
    # Compute all unfoldings of T and its SVD's. 
    T1 = np.zeros((m, n*p), dtype = np.float64)
    T2 = np.zeros((n, m*p), dtype = np.float64)
    T3 = np.zeros((p, m*n), dtype = np.float64) 
    T1 = cnv.unfold(T, T1, m, n, p, 1)
    T2 = cnv.unfold(T, T2, m, n, p, 2)
    T3 = cnv.unfold(T, T3, m, n, p, 3)
    sigma1, sigma2, sigma3, U1, U2, U3 = aux.unfoldings_svd(T1, T2, T3, m, n, p)
                            
    # TRUNCATE SVD'S OF UNFOLDINGS

    # Specific truncation is given by the user.
    if type(trunc_dims) == list:
        mlsvd_stop = 0
        S_energy = np.sum(sigma1**2) + np.sum(sigma2**2) + np.sum(sigma3**2)
        R1, R2, R3 = trunc_dims 
        U1, U2, U3 = U1[:, :R1], U2[:, :R2], U3[:, :R3] 
        sigma1, sigma2, sigma3 = sigma1[:R1], sigma2[:R2], sigma3[:R3]
        S = aux.multilin_mult(T, U1.transpose(), U2.transpose(), U3.transpose(), m, n, p)  
        best_energy = aux.compute_energy(S_energy, sigma1, sigma2, sigma3) 
        if display == 3:
            best_error = aux.compute_error(T, Tsize, S, R1, R2, R3, U1, U2, U3)
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        else:
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # Level = 4 means no truncation but with compression, that is, we use the MLSVD without truncating it.
    if level == 4:
        mlsvd_stop = 8
        R1, R2, R3 = sigma1.size, sigma2.size, sigma3.size
        S = aux.multilin_mult(T, U1.transpose(), U2.transpose(), U3.transpose(), m, n, p)
        if display == 3:
            best_error = aux.compute_error(T, Tsize, S, R1, R2, R3, U1, U2, U3)
            return S, 100, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, 0.0
        else:
            return S, 100, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # Level = 5 means no truncation and no compression, in other words, the original tensor.
    if level == 5:
        mlsvd_stop = 1
        U1, U2, U3 = np.eye(m), np.eye(n), np.eye(p)
        sigma1, sigma2, sigma3 = np.ones(m), np.ones(n), np.ones(p)
        if display == 3:
            return T, 100, m, n, p, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, 0.0
        else:
            return T, 100, m, n, p, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # The original SVD factors may have extra information due to noise or numerical error. We clean this SVD performing
    # a specialized truncation. 
    stage = 1
    S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, situation = aux.clean_compression(T, Tsize, T, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage)
                      
    # Overfit occurred and the user should decrease the value of the rank.
    if situation == 'overfit':
        sys.exit('Rank chosen is to big and caused overfit.')

    # TRUNCATE THE TRUNCATION

    # If one of the conditions below is true we don't truncate again.
    if (R1, R2, R3) == (m, n, p) or situation == 'random':
        if display == 3:
            best_error = aux.compute_error(T, Tsize, S, R1, R2, R3, U1, U2, U3)
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        else:
            return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop

    # Sometimes the first truncation still is too large. To fix this we consider a second truncation over the first truncation.
    else:
        if level < 3:
            level += 1
        stage = 2
        best_energy2 = 100
        S, best_energy2, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, situation = aux.clean_compression(T, Tsize, S, sigma1, sigma2, sigma3, U1, U2, U3, m, n, p, r, level, stage)
        # The second energy is a fraction of the first one. To compare with the original MLSVD we update the energy accordingly.
        best_energy = best_energy*best_energy2/100
                             
        if situation == 'overfit':
            sys.exit('Rank chosen is to big and caused overfit.')

    # Compute error of compressed tensor.
    if display == 3:
        best_error = aux.compute_error(T, Tsize, S, R1, R2, R3, U1, U2, U3)
        return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error
        
    return S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop
           

def rank(T, plot=True):
    """
    This function computes several approximations of T for r = 1...max rank. 
    These computations will be used to determine the (most probable) rank of T. The function 
    also returns an array `errors_per_rank` with the relative errors for each rank computed. 
    It is relevant to say that the rank r computed can also be the `border rank` of T, not the 
    actual rank. 

    The idea is that the minimum of |T - T_approx|, for each rank r, stabilizes when S 
    has the same rank as T. This function also plots the graph of the errors so the user 
    are able to visualize the moment when the error stabilizes.
    
    Inputs
    ------
    T: float L-D ndarray
    display: int
            
    Outputs
    -------
    final_rank: int
        The computed rank of T.
    errors_per_rank: float 1-D ndarray
        The error |T - T_approx| computed for each rank.    
    """
    
    # Compute norm of T.
    Tsize = np.linalg.norm(T)

    # Set random initialization (works more generally)
    class options:
        init = 'random' 

    # START THE PROCCESS OF FINDING THE RANK

    # Decide bounds for rank.
    dims = np.array(T.shape)
    L = dims.size
    
    if L > 3:
        Rmin, Rmax = 2, np.min(dims)
    else:
        m, n, p = T.shape
        Rmin, Rmax = 1, min(m*n, m*p, n*p) 
        
    # error_per_rank saves the relative error of the CPD for each rank r.
    error_per_rank = np.zeros(Rmax)
    
    print('Start searching for rank')
    print('Stops at r =',Rmax,' or less')
    print('-----------------------------')

    for r in range(Rmin, Rmax+1):  
        s = "Testing r = " + str(r)
        sys.stdout.write('\r'+s)
    
        factors, T_approx, outputs = cpd(T, r, options)
    
        # Save relative error of this approximation.
        rel_error = np.linalg.norm(T - T_approx)/Tsize
        error_per_rank[r-Rmin] = rel_error   
                
        # Stopping conditions
        if rel_error < 1e-4:
            break
        elif r > Rmin:
            if np.abs(error_per_rank[r-1] - error_per_rank[r-2]) < 1e-5:
                break
    
    # SAVE LAST INFORMATIONS
    
    if L > 3:
        error_per_rank = error_per_rank[0:r-1] 
        final_rank = np.nanargmin(error_per_rank)+2
    else:
        error_per_rank = error_per_rank[0:r] 
        final_rank = np.nanargmin(error_per_rank)+1
            
    # DISPLAY AND PLOT ALL RESULTS
    
    print('\nEstimated rank(T) =', final_rank)
    print('|T - T_approx|/|T| =', error_per_rank[final_rank - Rmin])
    
    if plot:
        plt.plot(range(Rmin, r+1), np.log10(error_per_rank))
        plt.plot(final_rank, np.log10(error_per_rank[final_rank - Rmin]), marker = 'o', color = 'k')
        plt.title('Rank trials')
        plt.xlabel('r')
        plt.ylabel(r'$\log_{10} \|T - S\|/|T|$')
        plt.grid()
        plt.show()
            
    return int(final_rank), error_per_rank


def stats(T, r, options=False, num_samples = 100):
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

    * The relative error |T - T_approx|/|T| obtained in each trial.

    Inputs
    ------
    T: float L-D ndarray
    r: int 
        The desired rank of the approximating tensor.
    maxiter: int
    tol: float
    num_samples: int
        Total of CPD's we want to compute to make statistics. Default is 100.
    """
    
    # Compute dimensions and norm of T.
    Tsize = np.linalg.norm(T)
    dims = T.shape
    L = len(dims)

    # Set options
    options = aux.make_class_options(options)

    # INITIALIZE RELEVANT ARRAYS
    
    times = np.zeros(num_samples)
    steps = np.zeros(num_samples)
    rel_errors = np.zeros(num_samples)
      
    # BEGINNING OF SAMPLING AND COMPUTING
    
    # At each run, the program computes a CPD for T with random guess for initial point.
    for trial in range(1, num_samples+1):            
        start = time.time()
        factors, T_approx, outputs = cpd(T, r, options)
               
        # Update the vectors with general information.
        if L > 3:
            if options.refine:
                num_steps = 0
                for output in outputs:
                    num_steps += output.num_steps
            else:
                num_steps = 0
                for output in outputs:
                    num_steps += output.num_steps-1

        else:
            if options.refine:
                num_steps = outputs.num_steps
            else:
                num_steps = outputs.num_steps-1

        rel_error = np.linalg.norm(T - T_approx)/Tsize
        
        end = time.time()
        times[trial-1] = end - start
        steps[trial-1] = num_steps
        rel_errors[trial-1] = rel_error
        
        # Display progress bar.
        s = "[" + trial*"=" + (num_samples-trial)*" " + "]" + " " + str( np.round(100*trial/num_samples, 2) ) + "%"
        sys.stdout.write('\r'+s)
     
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


def cg(X, Y, Z, data, data_rmatvec, y, g, b, m, n, p, r, damp, cg_maxiter, tol):
    """
    Conjugate gradient algorithm specialized to the tensor case.
    """

    # Give names to the arrays.
    Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, M, L, residual_cg, P, Q, z = data
    M_X, M_Y, M_Z, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z = data_rmatvec
    
    # Compute the values of all arrays.
    Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ = crt.gramians(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ)
    N_X, N_Y, N_Z = crt.update_data_rmatvec(X, Y, Z, M_X, M_Y, M_Z)
    L = crt.regularization(X, Y, Z, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, m, n, p, r)
    M = crt.precond(X, Y, Z, L, M, damp, m, n, p, r)
    const = 2 + int(cg_maxiter/5)
    
    y = 0*y
    
    # g = Dres^T*res is the gradient of the error function E.    
    g = crt.rmatvec(b, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z, m, n, p, r)
    residual = M*g
    P = residual
    residualnorm = np.dot(residual, residual)
    if residualnorm == 0.0:
        residualnorm = 1e-6
    residualnorm_new = 0.0
    alpha = 0.0
    beta = 0.0
    residual_list = []
        
    for itn in range(0, cg_maxiter):
        Q = M*P
        z = crt.matvec(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, Q, m, n, p, r) + damp*L*Q
        z = M*z
        denominator = np.dot(P.T, z)
        if denominator == 0.0:
            denominator = 1e-6
        alpha = residualnorm/denominator
        y += alpha*P
        residual = residual - alpha*z
        residualnorm_new = np.dot(residual, residual)
        beta = residualnorm_new/residualnorm
        residualnorm = residualnorm_new
        residual_list.append(residualnorm)
        P = residual + beta*P
        
        # Stopping criteria.
        if residualnorm < tol:
            break   

        # Stop if the average residual norms from itn-2*const to itn-const is less than the average of residual norms from itn-const to itn.
        if itn >= 2*const and itn%const == 0:  
             if np.mean(residual_list[itn-2*const : itn-const]) < np.mean(residual_list[itn-const : itn]):
                 break
    
    return M*y, g, itn+1, residualnorm


def TT_cores(T, r):
    """
    Function to compute the tensor train cores of T with specific format to 
    obtain the CPD of T. This tensor train follow the format
    dims[0] x r -> r x dims[1] x r -> ... -> r x dims[L-2] x r -> r x dims[L-1].
    """

    # Compute dimensions and norm of T
    dims = np.array(T.shape)
    L = dims.size
    Tsize = np.linalg.norm(T)
    
    # Compute first unfolding of T 
    T1 = cnv.high_unfold1(T, dims)
    
    # List of cores
    G = []
    
    # Compute first core
    U, S, V = np.linalg.svd(T1, full_matrices=False, compute_uv=True)
    U = U[:,:r]
    S = S[:r]
    S = np.diag(S)
    V = V[:r,:]
    V = np.dot(S, V)
    G.append(U)
    
    # Compute remaining cores, except for the last one
    for l in range(1,L-1):
        V, g = aux.compute_core(V, dims, r, l)
        G.append(g)
        
    # Last core
    G.append(V)
    
    return G
