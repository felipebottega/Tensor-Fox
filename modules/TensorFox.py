"""
 General Description
 ===================
 *Tensor Fox* is a vast library of routines related to tensor problems. Since most tensor problems fall in the category 
of NP-hard problems, a great effort was made to make this library as efficient as possible. Some relevant routines and 
features of Tensor Fox are the following: 
 
 - Canonical polyadic decomposition (CPD)
 
 - Multilinear singular value decomposition (MLSVD)
 
 - Multilinear rank
 
 - Rank estimate
 
 - Rank related statistics, including histograms
 
 - Rank related information about tensors and tensorial spaces
 
 - CPD tensor train
 
 - High performance with parallelism 
"""

# Python modules
import numpy as np
from numpy import inf, copy, dot, zeros, empty, array, nanargmin, log10, diag, arange, prod
from numpy.linalg import norm, pinv
import sys
import time
import copy as cp
from decimal import Decimal
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd as rand_svd

# Tensor Fox modules
import Auxiliar as aux
import Compression as cmpr
import Conversion as cnv
import Critical as crt
import Dense as dns
import Display as disp
import GaussNewton as gn
import Initialization as init
import MultilinearAlgebra as mlinalg


def cpd(T, r, options=False):
    """
    Given a tensor T and a rank r, this function computes an approximated CPD of T with rank r. The factors matrices are 
    given in the form of a list [W^(1),...,W^(L)]. These matrices are such that sum_(l=1)^r W[:,l]^(1) ⊗ ... ⊗ W[:,l]^(L) 
    is an approximation for T, where W[:,l]^(1) denotes the l-th column of W^(1). The same goes for the other matrices.

    Inputs
    ------
    T: float L-D ndarray
        Objective tensor in coordinates.
    r: int 
        The desired rank of the approximating tensor.
    options: class with the following parameters to be defined.
        maxiter: int
            Number of maximum iterations allowed for the dGN function. Default is 200.
        tol: float
            Tolerance criterium to stop the iteration proccess of the dGN function. Default is 1e-12.
        init_method: string or list
            This options is used to choose the initial point to start the iterations. For more information, check the 
                function starting_point.
        trunc_dims: int or list of ints
            If trunc_dims is not 0, then it should be a list with three integers [R1,R2,R3] such that 1 <= R1 <= m, 
                1 <= R2 <= n, 1 <= R3 <= p. The compressed tensor will have dimensions (R1,R2,R3). Default is 0, which 
                means 'automatic' truncation. This options only serves to the tricpd function.
        level:  0, 1, 2, 3, 4 or 5
             The conditions to accept a truncation are defined by the parameter level. Higher means harder constraints, 
                which means bigger dimensions. Default is 1. For more information check the function set_constraints.
        symm: bool
            The user should set symm to True if the objetive tensor is symmetric, otherwise symm is False. Default is 
                False.
        low, upp, factor: floats
            These values sets constraints to the entries of the tensor. Default for all of them is 0, which means no 
                restriction. The parameter factor is auxiliar and influences how tight are the projections into the 
                interval [low, upp]. These parameters are experimental.
        trials: int
                This parameter is only used for tensor with order higher than 3. The computation of the tensor train CPD 
                requires the computation of several CPD of third order tensors. If only one of these CPD's is of low 
                quality (divergence or local minima) then all effort is in vain. One work around is to compute several 
                CPD'd and keep the best, for third order tensor. The parameter trials defines the maximum number of times 
                we repeat the computation of each third order CPD. These trials stops when the relative error is less than 
                1e-4 or when the maximum number of trials is reached. Default is trials=1. 
        display: -1, 0, 1, 2 or 3
            This options is used to control how information about the computations are displayed on the screen. The 
                possible values are -1, 0, 1 (default), 2, 3. Notice that display=3 makes the overall running time large 
                since it will force the program to show intermediate errors which are computationally costly. -1 is a 
                special option for displaying minimal relevant information for tensors with order higher then 3. We 
                summarize the display options below.
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
        Each tricpd and bicpd call gives a output class with all sort of information about the computations. The list 
        'outputs' contains all these classes.
    """ 

    # INITIAL PREPARATIONS

    # Compute dimensions and norm of T.
    dims = T.shape
    L = len(dims)
    Tsize = norm(T)
    T_orig = copy(T)
    
    # Set options
    options = aux.make_options(options, dims)
    display = options.display
    level = options.level
    level = options.level
    if type(level) == list:
        if L > 3:
            level = level[0]
        else:
            level = level[1]
                   
    # Test consistency of dimensions and rank.
    aux.consistency(r, dims, options.symm)  
        
    # Verify if T is a third order tensor.
    L = len(dims)
    if L == 3:
        X, Y, Z, T_approx, output = tricpd(T, r, options)
        return [X, Y, Z], T_approx, output   
    
    # START COMPUTATIONS
    
    if L > 3:

        # Increase dimensions if r > min(dims).
        if r > min(dims):
            inflate_status = True
            T, orig_dims, dims = cnv.inflate(T, r, dims)
        else:
            inflate_status = False
            orig_dims = dims

        # COMPRESSION STAGE

        if display != 0:
            print('-----------------------------------------------------------------------------------------------')
            print('Computing MLSVD')

        # Compute compressed version of T with the MLSVD. We have that T = (U_1,...,U_L)*S.
        if display > 2 or display < -1:
            S, best_energy, best_dims, U, UT, sigmas, mlsvd_stop, best_error = cmpr.mlsvd(T, Tsize, r, options)
        else: 
            S, best_energy, best_dims, U, UT, sigmas, mlsvd_stop = cmpr.mlsvd(T, Tsize, r, options)
        
        if display != 0:
            if level == 4:
                print('    Compression without truncation requested by user')
                print('    Compressing from', T.shape, 'to', S.shape)
            elif prod(array(best_dims) == array(dims)):
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
            if display > 2 or display < -1:
                print('    Compression relative error = {:5e}'.format(best_error))
            print()

        # For higher order tensors the trunc_dims options is only valid for the original tensor and its MLSVD.
        options.trunc_dims = 0

        # TENSOR TRAIN AND DAMPED GAUSS-NEWTON STAGE

        factors, S_approx, outputs = highcpd(S, r, options)  

        # Use the orthogonal transformations to work in the original space.
        for l in range(L):
            factors[l] = dot(U[l], factors[l])
    
    # FINAL WORKS

    num_steps = 0
    for output in outputs:
            num_steps += output.num_steps
    T_approx = zeros(dims)
    T_approx = cnv.cpd2tens(T_approx, factors, dims)
    T_approx = cnv.deflate(T_approx, orig_dims, dims, inflate_status)
    rel_error = norm(T_orig - T_approx)/Tsize
    accuracy = max(0, 100*(1 - rel_error))
    
    if options.display != 0:
        print()
        print('===============================================================================================')
        print('===============================================================================================')
        print('Final results')
        print('    Number of steps =', num_steps)
        print('    Relative error =', rel_error)
        acc = float( '%.6e' % Decimal(accuracy) )
        print('    Accuracy = ', acc, '%')

    final_outputs = aux.make_final_outputs(num_steps, rel_error, accuracy, outputs, options)
    
    return factors, T_approx, final_outputs


def highcpd(T, r, options):
    """
    This function makes the calls in order to compute the tensor train of T and obtain the final CPD from it. It is 
    important to realize that this function is limited to tensor where each one of its factors is a full rank matrix. 
    In particular, the rank r must be smaller than all dimensions of T.
    """     

    # Create relevant values
    dims = T.shape
    display = options.display
    max_trials = options.trials
    options.refine = False

    # Outputs is a list containing the output class of each cpd
    outputs = []

    # Compute cores of the tensor train of T
    G = cpdtt(T, r)
    L = len(G)   
    if display > 2 or display < -1:
        print('===============================================================================================')
        print('SVD Tensor train error = ', aux.tt_error(T, G, dims, L))
        print('===============================================================================================')
        print() 
    
    # List of CPD's
    cpd_list = []
    
    # Compute cpd of second core
    if display != 0:
        print('Total of', L-2, 'third order CPDs to be computed:')
        print('===============================================================================================')
    best_error = inf
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
    cpd_list.append([best_X, best_Y, best_Z])
    if display < 0:
        print('CPD 1 error =', best_error)
    
    # Compute third order CPD's of cores G[2] to G[L-2]
    for l in range(2, L-1):
        best_error = inf
        fixed_X = pinv(best_Z.T)
        for trial in range(max_trials):
            if display > 0:
                print()
                print('CPD', l)
            X, Y, Z, T_approx, output = bicpd(G[l], r, [fixed_X,0], options)
            if output.rel_error < best_error:
                best_output = output
                best_error = output.rel_error
                best_X, best_Y, best_Z = X, Y, Z
                if best_error < 1e-4:
                    break
        outputs.append(best_output)
        cpd_list.append([fixed_X, best_Y, best_Z])
        if display < 0:
            print('CPD', l, 'error =', best_error)
                
    # Compute of factors of T
    factors = []
    # First factor
    factors.append(dot(G[0], cpd_list[0][0]))
    # Factors 2 to L-2
    for l in range(0, L-2):
        factors.append(cpd_list[l][1])
    B = dot(G[-1].T, best_Z)
    factors.append( B )

    if display > 2 or display < -1:
        G_approx = [G[0]]
        for l in range(1,L-1):
            temp_factors = cpd_list[l-1]
            temp_dims = temp_factors[0].shape[0], temp_factors[1].shape[0], temp_factors[2].shape[0], 
            T_approx = empty(temp_dims)
            T_approx = cnv.cpd2tens(T_approx, temp_factors, temp_dims)
            G_approx.append(T_approx)            
        G_approx.append(G[-1])
        print()
        print('===============================================================================================')
        print('CPD Tensor train error = ', aux.tt_error(T, G_approx, dims, L))
        print('===============================================================================================')

    # Generate approximate tensor in coordinates. 
    T_approx = empty(T.shape)
    T_approx = cnv.cpd2tens(T_approx, factors, dims)
    
    return factors, T_approx, outputs


def tricpd(T, r, options):
    """
    Given a tensor T and a rank r, this function computes an approximated CPD of T with rank r. The result is given in the 
    form [X, Y, Z]. These matrices are such that sum_(l=1)^r X(l) ⊗ Y(l) ⊗ Z(l) is an approximation for T. 
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
        This class contains all information needed about the computations made. We summarize these informations below.
            rel_error: relative error |T - T_approx|/|T| of the approximation computed. 
            step_sizes: array with the distances between consecutive computed points at each iteration.
            errors: array with the absolute errors of the approximating tensor at each iteration.
            errors_diff: array with the differences between consecutive absolute errors.
            gradients: array with the gradient of the error function at each iteration. We expect that these gradients 
                       converges to zero as we keep iterating since the objetive point is a local minimum.
            stop: it is a list of three integers. The first one indicates how the compression was obtained. The second 
                  integer indicates why the dGN stopped at the first run, and the third integer indicates why the dGN 
                  stopped at the second run (refinement stage). See the functions mlsvd and dGN for more information.
            num_steps: the total number of steps (iterations) the dGN function used at the two runs.
            accuracy: the accuracy of the solution, which is defined by the formula 100*(1 - rel_error). 0 means 0% of 
                      accuracy (worst case) and 100 means 100% of accuracy (best case). 
    """ 

    # INITIALIZE RELEVANT VARIABLES 

    # Extract all variable from the class of options.
    tol = options.tol
    init_method = options.init_method
    display = options.display
    refine = options.refine
    symm = options.symm
    display = options.display
    level = options.level
    if type(level) == list:
        level = level[1]

    # Set the other variables.
    m_orig, n_orig, p_orig = T.shape
    m, n, p = m_orig, n_orig, p_orig
    T_orig = copy(T)
    Tsize = norm(T)
                   
    # Test consistency of dimensions and rank.
    aux.consistency(r, (m, n, p), symm) 

    # Change ordering of indexes to improve performance if possible.
    T, ordering = aux.sort_dims(T, m, n, p)
    m, n, p = T.shape    
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')
        print('Computing MLSVD')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U1,U2,U3)*S.
    if display > 2:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error = cmpr.mlsvd(T, Tsize, r, options)
    else:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop = cmpr.mlsvd(T, Tsize, r, options)

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
        if display > 2:
            print('    Compression relative error = {:5e}'.format(best_error))
            
    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start dGN.
    if display > 2:
        X, Y, Z, rel_error = init.starting_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, ordering, options)  
    else:  
        X, Y, Z = init.starting_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, ordering, options)  
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')        
        if type(init_method) == list:
            print('Type of initialization: user')
        else:
            print('Type of initialization:', init_method)
        if display > 2:
            print('    Initial guess relative error = {:5e}'.format(rel_error))   
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')
        print('Computing CPD')
    
    # Compute the approximated tensor in coordinates with the dGN method.
    X, Y, Z, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = gn.dGN(S, X, Y, Z, r, options) 

    # Use the orthogonal transformations to work in the original space.
    X = dot(U1, X)
    Y = dot(U2, Y)
    Z = dot(U3, Z)
    
    # REFINEMENT STAGE
    
    if refine:   
        if display > 0:
            print()
            print('===============================================================================================') 
            print('Computing refinement of solution') 
     
        if display > 2:
            T_approx = empty((m, n, p))
            T_approx = cnv.cpd2tens(T_approx, [X, Y, Z], (m, n, p))
            rel_error = norm(T - T_approx)/Tsize
            print('    Initial guess relative error = {:5e}'.format(rel_error))
        if display > 0:
            print('-----------------------------------------------------------------------------------------------')
            print('Computing CPD')

        X, Y, Z, step_sizes_refine, errors_refine, improv_refine, gradients_refine, stop_refine = gn.dGN(T, X, Y, Z, r, options)

    else:
        step_sizes_refine = array([0])
        errors_refine = array([0])
        improv_refine = array([0]) 
        gradients_refine = array([0]) 
        stop_refine = 5 
    
    # FINAL WORKS

    # Go back to the original dimension ordering.
    X, Y, Z = aux.unsort_dims(X, Y, Z, ordering)
    
    # Compute coordinate representation of the CPD of T.
    T_approx = empty((m_orig, n_orig, p_orig))
    T_approx = cnv.cpd2tens(T_approx, [X, Y, Z], (m_orig, n_orig, p_orig))
        
    # Save and display final informations.
    output = aux.output_info(T_orig, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, improv_main, improv_refine, gradients_main, gradients_refine, mlsvd_stop, stop_main, stop_refine, options)

    if display > 0:
        print('===============================================================================================')
        print('Final results')
        if refine:
            print('    Number of steps =', output.num_steps)
        else:
            print('    Number of steps =', output.num_steps)
        print('    Relative error =', output.rel_error)
        acc = float( '%.6e' % Decimal(output.accuracy) )
        print('    Accuracy = ', acc, '%')
    
    return X, Y, Z, T_approx, output


def bicpd(T, r, fixed_factor, options):
    """
    Practically the same as tricpd, but this function keeps the first factor fixed during all the computations. 
    """

    # INITIALIZE RELEVANT VARIABLES 

    # Extract all variable from the class of options.
    tol = options.tol
    init_method = options.init_method
    display = options.display
    refine = options.refine
    symm = options.symm
    display = options.display
    level = options.level
    if type(level) == list:
        level = level[1]

    # Set the other variables.
    m, n, p = T.shape
    Tsize = norm(T)
    ordering = [0,1,2]
                           
    # Test consistency of dimensions and rank.
    aux.consistency(r, (m, n, p), symm)     
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------') 
        print('Computing MLSVD of T')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U1,U2,U3)*S.
    if display > 2:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error = cmpr.mlsvd(T, Tsize, r, options)
    else:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop = cmpr.mlsvd(T, Tsize, r, options)

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
        if display > 2:
            print('    Compression relative error = {:5e}'.format(best_error))

    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start dGN.
    if display > 2:
        X, Y, Z, rel_error = init.starting_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, ordering, options)  
    else:  
        X, Y, Z = init.starting_point(T, Tsize, S, U1, U2, U3, r, R1, R2, R3, ordering, options)

    # Discard the factor computed in start_point and use the previous one. Then project it on the compressed space.
    if fixed_factor[1] == 0:
        X = dot(U1.T, fixed_factor[0])
        X = [X, 0]
    elif fixed_factor[1] == 1:
        Y = dot(U2.T, fixed_factor[0])
        Y = [Y, 1]
    elif fixed_factor[1] == 2:
        Z = dot(U3.T, fixed_factor[0])
        Z = [Z, 2]
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')        
        if type(init_method) == list:
            print('Type of initialization: fixed + user')
        else:
            print('Type of initialization: fixed +', init_method)
        if display > 2:
            S_init = empty((R1, R2, R3))
            if fixed_factor[1] == 0:
                S_init = cnv.cpd2tens(S_init, [X[0], Y, Z], (R1, R2, R3))
            elif fixed_factor[1] == 1:
                S_init = cnv.cpd2tens(S_init, [X, Y[0], Z], (R1, R2, R3))
            elif fixed_factor[1] == 2:
                S_init = cnv.cpd2tens(S_init, [X, Y, Z[0]], (R1, R2, R3))
            S1_init = cnv.unfold(S_init, 1, (R1, R2, R3))
            rel_error = aux.compute_error(T, Tsize, S_init, S1_init, [U1, U2, U3], (R1, R2, R3))
            print('    Initial guess relative error = {:5e}'.format(rel_error))           
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------') 
        print('Computing CPD of T')
    
    # Compute the approximated tensor in coordinates with the dGN method. 
    X, Y, Z, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = gn.dGN(S, X, Y, Z, r, options)
 
    # FINAL WORKS
    
    # Use the orthogonal transformations to obtain the CPD of T.
    if fixed_factor[1] == 0:               
        Y = dot(U2, Y)
        Z = dot(U3, Z)
    elif fixed_factor[1] == 1:               
        X = dot(U1, X)
        Z = dot(U3, Z)
    elif fixed_factor[1] == 2:               
        X = dot(U1, X)
        Y = dot(U2, Y)
    
    # Compute coordinate representation of the CPD of T.
    T_approx = empty((m, n, p))
    if fixed_factor[1] == 0:
        T_approx = cnv.cpd2tens(T_approx, [fixed_factor[0], Y, Z], (m, n, p))
    elif fixed_factor[1] == 1:
        T_approx = cnv.cpd2tens(T_approx, [X, fixed_factor[0], Z], (m, n, p))
    elif fixed_factor[1] == 2:
        T_approx = cnv.cpd2tens(T_approx, [X, Y, fixed_factor[0]], (m, n, p))
    
    # Save and display final informations.
    step_sizes_refine = array([0])
    errors_refine = array([0])
    improv_refine = array([0]) 
    gradients_refine = array([0]) 
    stop_refine = 5 
    output = aux.output_info(T, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, improv_main, improv_refine, gradients_main, gradients_refine, mlsvd_stop, stop_main, stop_refine, options)

    if display > 0:
        print('===============================================================================================')
        print('Final results of bicpd')
        if refine:
            print('    Number of steps =', output.num_steps)
        else:
            print('    Number of steps =', output.num_steps)
        print('    Relative error =', output.rel_error)
        acc = float( '%.6e' % Decimal(output.accuracy) )
        print('    Accuracy = ', acc, '%')
    
    return X, Y, Z, T_approx, output
           

def rank(T, options=False, plot=True):
    """
    This function computes several approximations of T for r = 1...max rank. These computations will be used to determine 
    the (most probable) rank of T. The function also returns an array `errors_per_rank` with the relative errors for each 
    rank computed. It is relevant to say that the rank r computed can also be the `border rank` of T, not the actual rank.
    The idea is that the minimum of |T - T_approx|, for each rank r, stabilizes when T_approx has the same rank as T. This 
    function also plots the graph of the errors so the user are able to visualize the moment when the error stabilizes.
    
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
    Tsize = norm(T)
    dims = T.shape

    # Set options
    options = aux.complete_options(options, dims) 

    # START THE PROCCESS OF FINDING THE RANK

    # Decide bounds for rank.
    dims = array(T.shape)
    L = dims.size
    
    if L > 3:
        Rmin, Rmax = 2, np.min(dims)
    else:
        m, n, p = T.shape
        Rmin, Rmax = 1, min(m*n, m*p, n*p) 
        
    # error_per_rank saves the relative error of the CPD for each rank r.
    error_per_rank = empty(Rmax)
    
    print('Start searching for rank')
    print('Stops at r =',Rmax,' or less')
    print('-----------------------------')

    for r in range(Rmin, Rmax+1):  
        s = "Testing r = " + str(r)
        sys.stdout.write('\r'+s)
    
        factors, T_approx, outputs = cpd(T, r, options)
    
        # Save relative error of this approximation.
        rel_error = norm(T - T_approx)/Tsize
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
        final_rank = nanargmin(error_per_rank)+2
    else:
        error_per_rank = error_per_rank[0:r] 
        final_rank = nanargmin(error_per_rank)+1
            
    # DISPLAY AND PLOT ALL RESULTS
    
    print('\nEstimated rank(T) =', final_rank)
    print('|T - T_approx|/|T| =', error_per_rank[final_rank - Rmin])
    
    if plot:
        plt.plot(range(Rmin, r+1), log10(error_per_rank))
        plt.plot(final_rank, log10(error_per_rank[final_rank - Rmin]), marker = 'o', color = 'k')
        plt.title('Rank trials')
        plt.xlabel('rank')
        plt.ylabel(r'$\log_{10} \|T - S\|/|T|$')
        plt.grid()
        plt.show()
            
    return int(final_rank), error_per_rank


def stats(T, r, options=False, num_samples = 100):
    """
    This function makes several calls of the Gauss-Newton function with random initial points. Each call turns into a 
    sample to recorded so we can make statistics lates. By defalt this functions takes 100 samples to analyze. The user 
    may choose the number of samples the program makes, but the computational time may be very costly. Also, the user may 
    choose the maximum number of iterations and the tolerance to be used in each Gauss-Newton function.
    The outputs plots with general information about all the trials. These informations are the following:
        - The total time spent in each trial.
        - The number of steps used in each trial.
        - The relative error |T - T_approx|/|T| obtained in each trial.

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
    Tsize = norm(T)
    dims = T.shape
    L = len(dims)

    # Set options
    options = aux.complete_options(options, dims)

    # INITIALIZE RELEVANT ARRAYS
    
    times = empty(num_samples)
    steps = empty(num_samples)
    errors = empty(num_samples)
      
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

        rel_error = norm(T - T_approx)/Tsize
        
        end = time.time()
        times[trial-1] = end - start
        steps[trial-1] = num_steps
        errors[trial-1] = rel_error
        
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

    [array,bins,patches] = plt.hist(log10(errors), 50)
    plt.xlabel(r'$\log_{10} \|T - \tilde{T}\|/\|T\|$')
    plt.ylabel('Quantity')
    plt.title('Histogram of the log10 of the relative error of each trial')
    plt.show()

    return times, steps, errors


def cpdtt(T, r):
    """
    Function to compute the tensor train cores of T with specific format to obtain the CPD of T. This tensor train follow 
    the format dims[0] x r -> r x dims[1] x r -> ... -> r x dims[L-2] x r -> r x dims[L-1].
    """

    # Compute dimensions and norm of T
    dims = array(T.shape)
    L = dims.size
    Tsize = norm(T)
    
    # Compute first unfolding of T 
    T1 = cnv.unfold(T, 1, dims)
    
    # List of cores
    G = []
    
    # Compute first core
    low_rank = min(T1.shape[0], T1.shape[1])
    U, S, V = rand_svd(T1, low_rank, n_iter=0)
    U = U[:,:r]
    S = diag(S)
    V = dot(S, V)
    V = V[:r,:]
    G.append(U)
    
    # Compute remaining cores, except for the last one
    for l in range(1,L-1):
        V, g = aux.tt_core(V, dims, r, l)
        G.append(g)
        
    # Last core
    G.append(V)
    
    return G


def foxit(T, r, options=False, bestof=1):
    """
    This is a special function made for the convenience of the user, i.e., this function makes the following:
        1) computes the desired CPD with the requested options
        2) prints the relevants results on the screen
        3) prints the parameters used
        4) plots the evolution of the step sizes, errors, improvements and gradients

    Additionally, the extra option 'bestof' tells the program to compute a certain number of CPD's and keep
    only the best one.    
    """

    dims = T.shape
    best_error = inf
    options = aux.complete_options(options, dims)

    for i in range(bestof):
        factors, T_approx, outputs = cpd(T, r, options)
        if outputs.rel_error < best_error:
            best_factors = copy(factors)
            best_T_approx = copy(T_approx)
            best_outputs = cp.deepcopy(outputs)

    print('Final results')
    print('    Number of steps =', best_outputs.num_steps)
    print('    Relative error =', best_outputs.rel_error)
    acc = float( '%.6e' % Decimal(best_outputs.accuracy) )
    print('    Accuracy = ', acc, '%')
    print()
    print('==========================================================================')
    print()
    print('Parameters used')
    print('    maximum of iterations:', options.maxiter)
    print('    tolerance:', options.tol)
    print('    initialization:', options.init_method)
    if options.method == 'lsmr_static':
        print('    algorithm: least squares with minimal residual (static)', )
    elif options.method == 'lsmr':
        print('    algorithm: least squares with minimal residual (dynamic)', )
    elif options.method == 'cg_static':
        print('    algorithm: conjugate gradient (static)')
    elif options.method == 'cg':
        print('    algorithm: conjugate gradient (dynamic)')
    print()

    plt.figure(figsize=[9,6])
    if options.refine:

        # sz1 is the size of the arrays of the main stage.
        sz1 = best_outputs.step_sizes[0].size
        x1 = arange(sz1)
        # sz2 is the size of the arrays of the refinement stage.
        sz2 = best_outputs.step_sizes[1].size
        x2 = arange(sz1-1, sz1 + sz2 - 1)

        # Step sizes
        plt.plot(x1, best_outputs.step_sizes[0],'k-' , markersize=2, label='Step sizes - Main')
        plt.plot(x2, best_outputs.step_size[1],'k--' , markersize=2, label='Step sizes - Refinement')

        # Errors
        plt.plot(x1, best_outputs.errors[0],'b-' , markersize=2, label='Relative errors - Main')
        plt.plot(x2, best_outputs.errors[1],'b--' , markersize=2, label='Relative errors - Refinement')

        # Improvements
        plt.plot(x1, best_outputs.improv[0], 'g-', markersize=2, label='Improvements - Main')
        plt.plot(x2, best_outputs.improv[1],'g--' , markersize=2, label='Improvements - Refinement')

        # Gradients
        plt.plot(x1, best_outputs.gradients[0], 'r-', markersize=2, label='Gradients - Main')
        plt.plot(x2, best_outputs.gradients[1],'r--' , markersize=2, label='Gradients - Refinement')

        plt.xlabel('iteration')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()

    else:
        # sz1 is the size of the arrays of the main stage.
        sz1 = best_outputs.step_sizes[0].size
        x1 = arange(sz1)

        # Step sizes
        plt.plot(x1, best_outputs.step_sizes[0],'k-' , markersize=2, label='Step sizes - Main')

        # Errors
        plt.plot(x1, best_outputs.errors[0],'b-' , markersize=2, label='Relative errors - Main')

        # Improvements
        plt.plot(x1, best_outputs.improv[0], 'g-', markersize=2, label='Improvements - Main')

        # Gradients
        plt.plot(x1, best_outputs.gradients[0], 'r-', markersize=2, label='Gradients - Main')

        plt.xlabel('iteration')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()

    return best_factors, best_T_approx, best_outputs
