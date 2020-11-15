"""
 General Description
 ===================
 Tensor Fox is a vast library of routines related to tensor problems. Since most tensor problems fall in the category
 of NP-hard problems, a great effort was made to make this library as efficient as possible. Some relevant routines and 
 features of Tensor Fox are the following: 
 
 - Canonical polyadic decomposition (CPD)
 
 - Multilinear singular value decomposition (MLSVD)
 
 - Multilinear rank
 
 - Rank estimate
 
 - Rank related statistics, including histograms
 
 - Rank related information about tensors and tensor spaces
 
 - CPD tensor train
 
 - High performance with parallelism

 References
 ==========

 - C. J. Hillar, and L.-H. Lim. Most tensor problems are NP-hard, Journal of the ACM, 60(6):45:1-45:39, November 2013.
   ISSN 0004-5411. doi: 10.1145/2512329.

 - T. G. Kolda and B. W. Bader, Tensor Decompositions and Applications, SIAM Review, 51:3, in press (2009).

 - P. Comon, X. Luciani, and A. L. F. de Almeida, Tensor Decompositions, Alternating Least Squares and other Tales,
   Journal of Chemometrics, Wiley, 2009.
"""

# Python modules
import numpy as np
from numpy import inf, dot, empty, array, nanargmin, log10, arange, prod, ndarray
from numpy.linalg import norm
import sys
import time
from copy import deepcopy
from decimal import Decimal
import matplotlib.pyplot as plt
import numba
import warnings

# Tensor Fox modules
from TensorFox.Alternating_Least_Squares import *
from TensorFox.Auxiliar import *
from TensorFox.Compression import *
from TensorFox.Conversion import *
from TensorFox.Critical import *
from TensorFox.Display import *
from TensorFox.GaussNewton import *
from TensorFox.Initialization import *
from TensorFox.MultilinearAlgebra import *


NumbaDeprecationWarning = numba.errors.NumbaDeprecationWarning
NumbaPendingDeprecationWarning = numba.errors.NumbaPendingDeprecationWarning
NumbaPerformanceWarning = numba.errors.NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def cpd(T, R, options=False):
    """
    Given a tensor T and a rank R, this function computes an approximated CPD of T with rank r. The factors matrices are
    given in the form of a list [W^(1),...,W^(L)]. They are such that sum_(r=1)^R W[:,r]^(1) ⊗ ... ⊗ W[:,r]^(L) is an
    approximation for T, where W[:,r]^(l) denotes the r-th column of W^(l). The same goes for the other factor matrices.

    Inputs
    ------
<<<<<<< HEAD:modules/TensorFox/TensorFox.py
    T: float array or list
=======
    T: float array
>>>>>>> 58034b32f2e81273c97359dca2f76c76f1eab7da:modules/TensorFox.py
        Objective tensor in coordinates. If T is a sparse tensor, give it as a list T = [data, idxs, dims], where dims
        are the dimensions of T (it can be a list, tuple or array). We remark that each idxs[i] is a tuple (or list or
        array) of the coordinates of T such that T[idxs[i]] = data[i]. If idxs is given as an array, note that each row
        of idxs is taken as a set of coordinates.
    R: int
        The desired rank of the approximating tensor.
    options: class with the following parameters
        maxiter: int
            Number of maximum iterations allowed for the dGN function. Default is 200.
        tol, tol_step, tol_improv, tol_grad: float
            Tolerance criterion to stop the iteration process of the dGN function. Default is 1e-6 for all. Let T^(k) be
            the approximation at the k-th iteration, with corresponding CPD w^(k) in vectorized form. The program stops 
            if
                1) |T - T^(k)| / |T| < tol
                2) | w^(k-1) - w^(k) | < tol_step
                3) | |T - T^(k-1)| / |T| - |T - T^(k)| / |T| | < tol_improv
                4) | grad F(w^(k)) | < tol_grad, where F(w^(k)) = 1/2 |T - T^(k)|^2
        tol_mlsvd: float
            Tolerance criterion for the truncation. The idea is to obtain a truncation (U_1,...,U_L)*S such that
            |T - (U_1,...,U_L)*S| / |T| < tol_mlsvd. Default is 1e-16. If tol_mlsvd = -1 the program uses the original 
        tensor, so the computation of the MLSVD is not performed.
        trunc_dims: int or list of ints
            Consider a third order tensor T. If trunc_dims is not 0, then it should be a list with three integers
            [R1,R2,R3] such that 1 <= R1 <= m, 1 <= R2 <= n, 1 <= R3 <= p. The compressed tensor will have dimensions
            (R1,R2,R3). Default is 0, which means 'automatic' truncation.
        initialization: string or list
            This options is used to choose the initial point to start the iterations. For more information, check the 
            function starting_point.
        refine: bool
            If True, after the dGN iterations the program uses the solution to repeat the dGN over the original space
            using the solution as starting point. Default is False.
        symm: bool
            The user should set symm to True if the objective tensor is symmetric, otherwise symm is False. Default is
            False.
        trials: int
            This parameter is only used for tensor with order higher than 3. The computation of the tensor train CPD 
            requires the computation of several CPD of third order tensors. If only one of these CPD's is of low 
            quality (divergence or local minimum) then all effort is in vain. One work around is to compute several
            CPD'd and keep the best, for third order tensor. The parameter trials defines the maximum number of
            times we repeat the computation of each third order CPD. These trials stops when the relative error is
            less than 1e-4 or when the maximum number of trials is reached. Default is trials=1.
        display: -2, -1, 0, 1, 2, 3 or 4
            This options is used to control how information about the computations are displayed on the screen. The 
            possible values are -1, 0, 1 (default), 2, 3, 4. Notice that display=3 makes the overall running time large
            since it will force the program to show intermediate errors which are computationally costly. -1 is a
            special option for displaying minimal relevant information for tensors with order higher then 3. We
            summarize the display options below.
                -2: display same as options -1 plus the Tensor Train error
                -1: display only the errors of each CPD computation and the final relevant information
                0: no information is printed
                1: partial information is printed
                2: full information is printed
                3: full information + errors of truncation and starting point are printed
                4: almost equal to display = 3 but now there are more digits displayed on the screen (display = 3 is a
                "cleaner" version of display = 4, with less information).
        epochs: int
            Number of Tensor Train CPD cycles. Use only for tensor with order higher than 3. Default is epochs=1.

    It is not necessary to create 'options' with all parameters described above. Any missing parameter is assigned to
    its default value automatically. For more information about the options, check the Tensor Fox tutorial at

        https://github.com/felipebottega/Tensor-Fox/tree/master/tutorial
    
    Outputs
    -------
    factors: list of float 2D arrays with shape (dims[i], R) each
        The factors matrices which corresponds to an approximate CPD for T.
    final_outputs: list of classes
        Each tricpd and bicpd call gives a output class with all sort of information about the computations. The list 
        'final_outputs' contains all these classes.
    """ 

    # INITIAL PREPARATIONS

    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        T_orig = deepcopy(T)
        T = deepcopy(T_orig)
        data_orig, idxs_orig, dims_orig = T_orig
        idxs_orig = np.array(idxs_orig)
    else:
        dims_orig = T.shape
    L = len(dims_orig)
    
    # Set options.
    options = make_options(options, L)
    method = options.method
    display = options.display
    tol_mlsvd = options.tol_mlsvd
    if type(tol_mlsvd) == list:
        if L > 3:
            tol_mlsvd = tol_mlsvd[0]
        else:
            tol_mlsvd = tol_mlsvd[1]
                   
    # Test consistency of dimensions and rank.
    consistency(R, dims_orig, options)
        
    # Verify method.
    if method == 'dGN' or method == 'als':
        factors, output = tricpd(T, R, options)
        return factors, output 
    
    # Change ordering of indexes to improve performance if possible.
    T, ordering = sort_dims(T)
    if type(T) == list:
        Tsize = norm(T[0])
        dims = T[2]
        # If T is sparse, we must use the classic method, and tol_mlsvd is set to the default 1e-16 in the case the
        # user requested -1 or 0.
        if tol_mlsvd < 0:
            options.tol_mlsvd = 1e-16
            tol_mlsvd = 1e-16
    else:
        Tsize = norm(T)
        dims = T.shape   

    # COMPRESSION STAGE

    if display != 0:
        print('-----------------------------------------------------------------------------------------------')
        print('Computing MLSVD')

    # Compute compressed version of T with the MLSVD. We have that T = (U_1,...,U_L)*S.
    if display > 2 or display < -1:
        S, U, T1, sigmas, best_error = mlsvd(T, Tsize, R, options)
    else: 
        S, U, T1, sigmas = mlsvd(T, Tsize, R, options)

    if display != 0:
        if prod(array(S.shape) == array(dims)):
            if tol_mlsvd == -1:
                print('    No compression and no truncation requested by user')
                print('    Working with dimensions', dims) 
            else:
                print('    No compression detected')
                print('    Working with dimensions', dims)                         
        else:
            print('    Compression detected')
            print('    Compressing from', dims, 'to', S.shape)
        if display > 2 or display < -1:
            print('    Compression relative error = {:7e}'.format(best_error))
        print()

    # Increase dimensions if r > min(S.shape).
    S_orig_dims = S.shape
    if R > min(S_orig_dims):
        inflate_status = True
        S = inflate(S, R, S_orig_dims)
    else:
        inflate_status = False

    # For higher order tensors the trunc_dims options is only valid for the original tensor and its MLSVD.
    options.trunc_dims = 0

    # TENSOR TRAIN AND DAMPED GAUSS-NEWTON STAGE

    factors, outputs = highcpd(S, R, options)
    factors = deflate(factors, S_orig_dims, inflate_status)

    # Use the orthogonal transformations to work in the original space.
    for l in range(L):
        factors[l] = dot(U[l], factors[l])
    
    # FINAL WORKS

    # Compute error.
    if type(T1) == ndarray:
        T1_approx = empty(T1.shape)
        T1_approx = cpd2unfold1(T1_approx, factors)
        rel_error = fastnorm(T1, T1_approx)/Tsize

        # Go back to the original dimension ordering.
        factors = unsort_dims(factors, ordering)

    else:
        # Go back to the original dimension ordering.
        factors = unsort_dims(factors, ordering)

        rel_error = sparse_fastnorm(data_orig, idxs_orig, dims_orig, factors)/Tsize

    num_steps = 0
    for output in outputs:
        num_steps += output.num_steps
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

    final_outputs = make_final_outputs(num_steps, rel_error, accuracy, outputs, options)
    
    return factors, final_outputs


def highcpd(T, R, options):
    """
    This function makes the calls in order to compute the tensor train of T and obtain the final CPD from it. It is 
    important to realize that this function is limited to tensor where each one of its factors is a full rank matrix. 
    In particular, the rank R must be smaller than all dimensions of T.
    """     

    # Create relevant values.
    dims = T.shape
    L = len(dims)
    display = options.display
    max_trials = options.trials
    options.refine = False
    epochs = options.epochs

    # Compute cores of the tensor train of T.
    G = cpdtt(T, R)
    if display > 2 or display < -1:
        print('===============================================================================================')
        print('SVD Tensor train error = ', tt_error(T, G, dims, L))
        print('===============================================================================================')
        print()     
    
    if display != 0:
        print('Total of', L-2, 'third order CPDs to be computed:')
        print('===============================================================================================')
   
    cpd_list, outputs, best_Z = cpd_cores(G, max_trials, epochs, R, display, options)
                
    # Compute of factors of T.

    # First factor
    factors = [dot(G[0], cpd_list[0][0])]
    # Factors 2 to L-2.
    for l in range(0, L-2):
        factors.append(cpd_list[l][1])
    B = dot(G[-1].T, best_Z)
    factors.append(B)
    factors = equalize(factors, R)

    if display > 2 or display < -1:
        G_approx = [G[0]]
        for l in range(1, L-1):
            temp_factors = cpd_list[l-1]
            temp_dims = temp_factors[0].shape[0], temp_factors[1].shape[0], temp_factors[2].shape[0], 
            T_approx = cpd2tens(temp_factors)
            G_approx.append(T_approx)            
        G_approx.append(G[-1])
        print()
        print('===============================================================================================')
        print('CPD Tensor train error = ', tt_error(T, G_approx, dims, L))
        print('===============================================================================================')
    
    return factors, outputs


def tricpd(T, R, options):
    """
    Given a tensor T and a rank R, this function computes an approximated CPD of T with rank R. This function is called
    when the user sets method = 'dGN'.

    Inputs
    ------
    T: float array
    R: int
    options: class
    
    Outputs
    -------
    factors: list of float 2D arrays
    output: class
        This class contains all information needed about the computations made. We summarize these information below.
            num_steps: the total number of steps (iterations) the dGN function used at the two runs.
            accuracy: the accuracy of the solution, which is defined by the formula 100*(1 - rel_error). 0 means 0% of 
                      accuracy (worst case) and 100 means 100% of accuracy (best case).
            rel_error: relative error |T - T_approx|/|T| of the approximation computed. 
            step_sizes: array with the distances between consecutive computed points at each iteration.
            errors: array with the absolute errors of the approximating tensor at each iteration.
            improv: array with the differences between consecutive absolute errors.
            gradients: array with the gradient of the error function at each iteration. We expect that these gradients 
                       converges to zero as we keep iterating since the objective point is a local minimum.
            stop: it is a list of two integers. The first integer indicates why the dGN stopped at the first run, and
                  the second integer indicates why the dGN stopped at the second run (refinement stage). Check the 
                  functions mlsvd and dGN for more information. 
    """ 

    # INITIALIZE RELEVANT VARIABLES 

    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        T_orig = deepcopy(T)
        T = deepcopy(T_orig)
        dims_orig = T_orig[2]
    else:
        dims_orig = T.shape
    L = len(dims_orig) 
    
    # Set options.
    initialization = options.initialization
    refine = options.refine
    symm = options.symm
    display = options.display
    tol_mlsvd = options.tol_mlsvd
    method = options.method
    if type(tol_mlsvd) == list:
        tol_mlsvd = tol_mlsvd[0]
        
    # Change ordering of indexes to improve performance if possible.
    T, ordering = sort_dims(T)
    if type(T) == list:
        Tsize = norm(T[0])
        dims = T[2]
        # If T is sparse, we must use the classic method, and tol_mlsvd is set to the default 1e-16 in the case the
        # user requested -1 or 0.
        if tol_mlsvd < 0:
            tol_mlsvd = 1e-16
            if type(tol_mlsvd) == list:
                options.tol_mlsvd[0] = 1e-16
            else:
                options.tol_mlsvd = 1e-16
    else:
        Tsize = norm(T)
        dims = T.shape  
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')
        print('Computing MLSVD')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U_1, ..., U_L)*S.
    if display > 2 or display < -1:
        S, U, T1, sigmas, best_error = mlsvd(T, Tsize, R, options)
    else:
        S, U, T1, sigmas = mlsvd(T, Tsize, R, options)
    dims_cmpr = S.shape

    # When the tensor is symmetric we want S to have equal dimensions. 
    if symm:
        R_min = min(dims_cmpr)
        dims_cmpr = [R_min for l in range(L)]
        dims_cmpr_slices = tuple(slice(R_min) for l in range(L))
        S = S[dims_cmpr_slices]
        U = [U[l][:, :R_min] for l in range(L)]
          
    if display > 0:
        if dims_cmpr == dims:
            if tol_mlsvd == -1:
                print('    No compression and no truncation requested by user')
                print('    Working with dimensions', dims) 
            else:
                print('    No compression detected')
                print('    Working with dimensions', dims)                         
        else:
            print('    Compression detected')
            print('    Compressing from', dims, 'to', S.shape)
        if display > 2:
            print('    Compression relative error = {:7e}'.format(best_error))
            
    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start d
    if display > 2 or display < -1:
        init_factors, init_error = starting_point(T, Tsize, S, U, R, ordering, options)
    else:  
        init_factors = starting_point(T, Tsize, S, U, R, ordering, options)
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')        
        if type(initialization) == list:
            print('Type of initialization: user')
        else:
            print('Type of initialization:', initialization)
        if display > 2:
            print('    Initial guess relative error = {:5e}'.format(init_error))
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------')
        print('Computing CPD')
    
    # Compute the approximated tensor in coordinates with dGN or ALS.
    if method == 'als':
        factors, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = \
            als(S, init_factors, R, options)
    else:
        factors, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = \
            dGN(S, init_factors, R, options)

    # Use the orthogonal transformations to work in the original space.
    for l in range(L):
        factors[l] = dot(U[l], factors[l])
    
    # REFINEMENT STAGE

    # If T is sparse, no refinement is made.
    if type(T) == list:
        refine = False
    
    if refine:   
        if display > 0:
            print()
            print('===============================================================================================') 
            print('Computing refinement of solution') 
     
        if display > 2:
            T1_approx = empty(T1.shape)
            T1_approx = cpd2unfold1(T1_approx, factors)
            init_error = fastnorm(T1, T1_approx)/Tsize
            print('    Initial guess relative error = {:5e}'.format(init_error))

        if display > 0:
            print('-----------------------------------------------------------------------------------------------')
            print('Computing CPD')

        if method == 'als':
            factors, step_sizes_refine, errors_refine, improv_refine, gradients_refine, stop_refine = \
                als(T, factors, R, options)
        else:
            factors, step_sizes_refine, errors_refine, improv_refine, gradients_refine, stop_refine = \
                dGN(T, factors, R, options)

    else:
        step_sizes_refine = array([0])
        errors_refine = array([0])
        improv_refine = array([0]) 
        gradients_refine = array([0]) 
        stop_refine = 8
    
    # FINAL WORKS

    # Compute error.
    if type(T1) == ndarray:
        T1_approx = empty(T1.shape)
        T1_approx = cpd2unfold1(T1_approx, factors)

        # Go back to the original dimension ordering.
        factors = unsort_dims(factors, ordering)

        # Save and display final informations.
        output = output_info(T1, Tsize, T1_approx,
                                 step_sizes_main, step_sizes_refine,
                                 errors_main, errors_refine,
                                 improv_main, improv_refine,
                                 gradients_main, gradients_refine,
                                 stop_main, stop_refine,
                                 options)
    else:
        # Go back to the original dimension ordering.
        factors = unsort_dims(factors, ordering)

        # Save and display final informations.
        output = output_info(T_orig, Tsize, factors,
                                 step_sizes_main, step_sizes_refine,
                                 errors_main, errors_refine,
                                 improv_main, improv_refine,
                                 gradients_main, gradients_refine,
                                 stop_main, stop_refine,
                                 options)

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
    
    return factors, output


def bicpd(T, R, fixed_factor, options):
    """
    Practically the same as tricpd, but this function keeps the some factor fixed during all the computations. This
    function is to be used as part of the tensor train cpd.
    """

    # INITIALIZE RELEVANT VARIABLES 

    # Extract all variable from the class of options.
    initialization = options.initialization
    refine = options.refine
    symm = options.symm
    display = options.display
    tol_mlsvd = options.tol_mlsvd
    bi_method = options.bi_method_parameters[0]
    if type(tol_mlsvd) == list:
        tol_mlsvd = tol_mlsvd[1]

    # Set the other variables.
    m, n, p = T.shape
    Tsize = norm(T)
    ordering = [0, 1, 2]
                           
    # Test consistency of dimensions and rank.
    consistency(R, (m, n, p), options)
    
    # COMPRESSION STAGE
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------') 
        print('Computing MLSVD of T')

    # Compute compressed version of T with the MLSVD. We have that T = (U1, U2, U3)*S.
    if display > 2 or display < -1:
        S, U, T1, sigmas, best_error = mlsvd(T, Tsize, R, options)
    else:
        S, U, T1, sigmas = mlsvd(T, Tsize, R, options)
    R1, R2, R3 = S.shape
    U1, U2, U3 = U

    # When the tensor is symmetric we want S to have equal dimensions. 
    if symm:
        R_min = min(R1, R2, R3)
        R1, R2, R3 = R_min, R_min, R_min
        S = S[:R_min, :R_min, :R_min]
        U1, U2, U3 = U1[:, :R_min], U2[:, :R_min], U3[:, :R_min]
          
    if display > 0:
        if (R1, R2, R3) == (m, n, p):
            if tol_mlsvd == -1:
                print('    No compression and no truncation requested by user')
                print('    Working with dimensions', T.shape) 
            else:
                print('    No compression detected')
                print('    Working with dimensions', T.shape)                           
        else:
            print('    Compression detected')
            print('    Compressing from', T.shape, 'to', S.shape)
        if display > 2:
            print('    Compression relative error = {:7e}'.format(best_error))

    # GENERATION OF STARTING POINT STAGE
        
    # Generate initial to start d
    if display > 2 or display < -1:
        [X, Y, Z], init_error = starting_point(T, Tsize, S, U, R, ordering, options)
    else:  
        [X, Y, Z] = starting_point(T, Tsize, S, U, R, ordering, options)

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
        if type(initialization) == list:
            print('Type of initialization: fixed + user')
        else:
            print('Type of initialization: fixed +', initialization)
        if display > 2:
            if fixed_factor[1] == 0:
                S_init = cpd2tens([X[0], Y, Z])
            elif fixed_factor[1] == 1:
                S_init = cpd2tens([X, Y[0], Z])
            elif fixed_factor[1] == 2:
                S_init = cpd2tens([X, Y, Z[0]])
            S1_init = unfold(S_init, 1)
            init_error = compute_error(T, Tsize, S1_init, [U1, U2, U3], (R1, R2, R3))
            print('    Initial guess relative error = {:5e}'.format(init_error))
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display > 0:
        print('-----------------------------------------------------------------------------------------------') 
        print('Computing CPD of T')
    
    # Compute the approximated tensor in coordinates with dGN or ALS. 
    if bi_method == 'als':
        factors, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = \
            als(S, [X, Y, Z], R, options)
    else:
        factors, step_sizes_main, errors_main, improv_main, gradients_main, stop_main = \
            dGN(S, [X, Y, Z], R, options)
    X, Y, Z = factors
 
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

    # Compute error.
    T1_approx = empty(T1.shape)
    if fixed_factor[1] == 0:
        T1_approx = cpd2unfold1(T1_approx, [fixed_factor[0], Y, Z])
    elif fixed_factor[1] == 1:
        T1_approx = cpd2unfold1(T1_approx, [X, fixed_factor[0], Z])
    elif fixed_factor[1] == 2:
        T1_approx = cpd2unfold1(T1_approx, [X, Y, fixed_factor[0]])
    
    # Save and display final information.
    step_sizes_refine = array([0])
    errors_refine = array([0])
    improv_refine = array([0]) 
    gradients_refine = array([0]) 
    stop_refine = 5 
    output = output_info(T1, Tsize, T1_approx,
                             step_sizes_main, step_sizes_refine,
                             errors_main, errors_refine,
                             improv_main, improv_refine,
                             gradients_main, gradients_refine,
                             stop_main, stop_refine,
                             options)

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
    
    return X, Y, Z, output
           

def rank(T, options=False, plot=True, trials=3):
    """
    This function computes several approximations of T for r = 1...max rank. These computations will be used to
    determine the (most probable) rank of T. The function also returns an array `errors_per_rank` with the relative
    errors for each rank computed. It is relevant to say that the rank r computed can also be the `border rank` of T,
    not the actual rank. The idea is that the minimum of |T - T_approx|, for each rank r, stabilizes when T_approx has
    the same rank as T. This function also plots the graph of the errors so the user are able to visualize the moment
    when the error stabilizes.
    
    Inputs
    ------
    T: float array or list
    options: class or bool
        The user can give the options to the cpd function if necessary.
    plot: bool
        If True (default), the program creates a plot with the relation between the error and rank.
    trial: int
        Number of times the program will compute a CPD for the same rank. The idea is to retain only the best one.
            
    Outputs
    -------
    final_rank: int
        The computed rank of T.
    errors_per_rank: float 1-D array
        The error |T - T_approx| computed for each rank.    
    """
    
    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        data, idxs, dims = T
        dims = array(dims)
    else:
        dims = T.shape
    L = len(dims)
<<<<<<< HEAD:modules/TensorFox/TensorFox.py
    
    # Set options
=======
>>>>>>> 58034b32f2e81273c97359dca2f76c76f1eab7da:modules/TensorFox.py
    options = make_options(options, L)

    # START THE PROCESS OF FINDING THE RANK
    
    if L > 3:
        Rmin, Rmax = 2, np.min(dims)
    else:
        m, n, p = dims
        Rmin, Rmax = 1, min(m*n, m*p, n*p) 
        
    # error_per_rank saves the relative error of the CPD for each rank r.
    error_per_rank = empty(Rmax)
    
    print('Start searching for rank')
    print('Stops at R =', Rmax, ' or less')
    print('-----------------------------')

    for r in range(1, Rmax+1):  
        s = "Testing R = " + str(r)
        sys.stdout.write('\r'+s)
    
        best_error = inf
        for t in range(trials):
            factors, outputs = cpd(T, r, options)
            rel_error = outputs.rel_error
            if rel_error < best_error:
                best_error = rel_error
    
        # Save relative error of this approximation.
        error_per_rank[r-1] = best_error
        
        # STOPPING CONDITIONS
        
        # Error small enough.
        if best_error < 1e-14:
            final_rank = r
            final_error = error_per_rank[r-1]
            break
        # Difference between errors small enough.
        if r > Rmin:
            if np.abs(error_per_rank[r-1] - error_per_rank[r-2]) < 1e-5:
                final_rank = nanargmin(error_per_rank[0:r])+1
                final_error = error_per_rank[final_rank-1]
                break
        # Error decreased orders of magnitude abruptly.
        if r > 2:
            previous_diff = np.abs(error_per_rank[r-2] - error_per_rank[r-3])
            current_diff = np.abs(error_per_rank[r-1] - error_per_rank[r-2])
            if previous_diff / current_diff > 100:
                final_rank = r-1
                final_error = error_per_rank[r-2]
                break
    
    # SAVE LAST INFORMATION
    
    error_per_rank = error_per_rank[0:r]
            
    # DISPLAY AND PLOT ALL RESULTS
    try:
        print('\nrank(T) =', final_rank)
        print('|T - T_approx|/|T| =', final_error)
    except:
        final_rank = Rmax
        final_error = error_per_rank[-1]
        print('\nrank(T) =', final_rank)
        print('|T - T_approx|/|T| =', final_error)
    
    if plot:
        plt.figure(figsize=[14, 4])
        plt.plot(range(1, r+1), error_per_rank, color='blue')
        plt.plot(range(1, r+1), error_per_rank, 's', color='blue')
        plt.plot(final_rank, final_error, marker='s', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Relative error')
        plt.yscale('log')
        if r > 20:
            plt.xticks(range(1, r+1, int((r+1)/20)))
        else:
            plt.xticks(range(1, r+1))
        plt.grid()
        plt.show()
            
    return int(final_rank), error_per_rank


def stats(T, R, options=False, num_samples=100):
    """
    This function makes several calls of the Gauss-Newton function with random initial points. Each call turns into a 
    sample to recorded so we can make statistics estimate. By default this functions takes 100 samples to analyze. The
    user may choose the number of samples the program makes, but the computational time may be very costly. Also, the
    user may choose the maximum number of iterations and the tolerance to be used in each Gauss-Newton function. The
    outputs plots with general information about all the trials. These information are the following:
        - The total time spent in each trial.
        - The number of steps used in each trial.
        - The relative error |T - T_approx|/|T| obtained in each trial.

    Inputs
    ------
    T: float array or list
    R: int
        The desired rank of the approximating tensor.
    options: class or bool
        The user can give the options to the cpd function if necessary.
    num_samples: int
        Total of CPD's we want to compute to make statistics. Default is 100.
        
    Outputs
    -------
    times, steps, errors: arrays
        Each times[i], steps[i], errors[i] correspond to the time spent, number of steps and error of the i-th trial.
    """
    
    # Verify if T is sparse, in which case it will be given as a list with the data.
    if type(T) == list:
        data, idxs, dims = T
    else:
        dims = T.shape
    L = len(dims)

    # Set options
    options = make_options(options, L)

    # INITIALIZE RELEVANT ARRAYS
    
    times = empty(num_samples)
    steps = empty(num_samples)
    errors = empty(num_samples)
      
    # BEGINNING OF SAMPLING AND COMPUTING
    
    # At each run, the program computes a CPD for T with random guess for initial point.
    for trial in range(1, num_samples+1):            
        start = time.time()
        factors, outputs = cpd(T, R, options)                     
        end = time.time()

        # Update info.
        rel_error = outputs.rel_error
        num_steps = outputs.num_steps  
        times[trial-1] = end - start
        steps[trial-1] = num_steps
        errors[trial-1] = rel_error
        
        # Display progress bar.
        x = 100*trial//num_samples
        s = "[" + x*"=" + (100-x)*" " + "]" + " " + str( np.round(100*trial/num_samples, 2) ) + "%"
        sys.stdout.write('\r'+s)
     
    # PLOT HISTOGRAMS

    [array, bins, patches] = plt.hist(times, 50, edgecolor='black')
    plt.xlabel('Seconds')
    plt.ylabel('Quantity')
    plt.title('Histogram of the total time of each trial')
    plt.show()

    [array, bins, patches] = plt.hist(steps, 50, edgecolor='black')
    plt.xlabel('Number of steps')
    plt.ylabel('Quantity')
    plt.title('Histogram of the number of steps of each trial')
    plt.show()

    [array, bins, patches] = plt.hist(log10(errors), 50, edgecolor='black')
    plt.xlabel(r'$\log_{10} \|T - \tilde{T}\|/\|T\|$')
    plt.ylabel('Quantity')
    plt.title('Histogram of the log10 of the relative error of each trial')
    plt.show()

    return times, steps, errors


def cpdtt(T, R):
    """
    Function to compute the tensor train cores of T with specific format to obtain the CPD of T. This tensor train
    follows the format dims[0] x R -> R x dims[1] x R -> ... -> R x dims[L-2] x R -> R x dims[L-1].
    """

    # Compute dimensions and norm of T.
    dims = array(T.shape)
    L = len(dims)
    
    # List of cores.
    G = []
    
    # Compute remaining cores, except for the last one.
    r1, r2 = 1, R
    V = T
    for l in range(0, L-1):
        V, g = tt_core(V, dims, r1, r2, l)
        r1, r2 = R, R
        G.append(g)
        
    # Last core.
    G.append(V)
    
    return G


def foxit(T, R, options=False, bestof=1):
    """
    This is a special function made for the convenience of the user, i.e., this function makes the following:
        1) computes the desired CPD with the requested options
        2) prints the relevant results on the screen
        3) prints the parameters used
        4) plots the evolution of the step sizes, errors, improvements and gradients

    Additionally, the extra option 'bestof' tells the program to compute a certain number of CPD's and retain only the
    best one.
    """

    best_error = inf
    if type(T) == list:
        data, idxs, dims = T
        idxs = np.array(idxs)
    else:
        dims = T.shape
    L = len(dims)
    options = make_options(options, L)

    for i in range(bestof):
        factors, outputs = cpd(T, R, options)
        if outputs.rel_error < best_error:
            best_error = outputs.rel_error
            best_factors = deepcopy(factors)
            best_outputs = deepcopy(outputs)

    print('Final results')
    print('    Number of steps =', best_outputs.num_steps)
    print('    Relative error =', best_outputs.rel_error)
    acc = float( '%.6e' % Decimal(best_outputs.accuracy) )
    print('    Accuracy = ', acc, '%')
    print()
    print('==========================================================================')
    print()
    print('Parameters used')
    print('    initialization:', options.initialization)
    print('    maximum of iterations:', options.maxiter)
    print('    error tolerance:', options.tol)
    print('    steps size tolerance:', options.tol_step)
    print('    improvement tolerance:', options.tol_improv)
    print('    gradient norm tolerance:', options.tol_grad)
    print('    inner algorithm parameters:') 
    if options.method == 'als':
        print('        method: alternating least squares')
    elif options.method == 'ttcpd':
        print('        method: tensor train cpd')
    elif options.inner_method == 'cg_static':
        print('        method: conjugate gradient static')
        print('        cg maximum of iterations:', options.cg_maxiter)
        print('        cg tolerance:', options.cg_tol)
    elif options.inner_method == 'cg':
        print('        method: conjugate gradient dynamic/random')
        print('        cg factor:', options.cg_factor)
        print('        cg tolerance:', options.cg_tol)
    elif options.inner_method == 'direct':
        print('        method: direct solver')
    elif type(options.inner_method) == list:
        print('        method: hybrid strategy')
    print()

    plt.figure(figsize=[9, 6])
    if options.refine:

        # sz1 is the size of the arrays of the main stage.
        sz1 = best_outputs.step_sizes[0].size
        x1 = arange(sz1)
        # sz2 is the size of the arrays of the refinement stage.
        sz2 = best_outputs.step_sizes[1].size
        x2 = arange(sz1-1, sz1 + sz2 - 1)

        # Step sizes
        plt.plot(x1, best_outputs.step_sizes[0], 'k-', markersize=2, label='Step sizes - Main')
        plt.plot(x2, best_outputs.step_size[1], 'k--', markersize=2, label='Step sizes - Refinement')

        # Errors
        plt.plot(x1, best_outputs.errors[0], 'b-', markersize=2, label='Relative errors - Main')
        plt.plot(x2, best_outputs.errors[1], 'b--', markersize=2, label='Relative errors - Refinement')

        # Improvements
        plt.plot(x1, best_outputs.improv[0], 'g-', markersize=2, label='Improvements - Main')
        plt.plot(x2, best_outputs.improv[1], 'g--', markersize=2, label='Improvements - Refinement')

        # Gradients
        plt.plot(x1, best_outputs.gradients[0], 'r-', markersize=2, label='Gradients - Main')
        plt.plot(x2, best_outputs.gradients[1], 'r--', markersize=2, label='Gradients - Refinement')

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
        plt.plot(x1, best_outputs.step_sizes[0], 'k-', markersize=2, label='Step sizes - Main')

        # Errors
        plt.plot(x1, best_outputs.errors[0], 'b-', markersize=2, label='Relative errors - Main')

        # Improvements
        plt.plot(x1, best_outputs.improv[0], 'g-', markersize=2, label='Improvements - Main')

        # Gradients
        plt.plot(x1, best_outputs.gradients[0], 'r-', markersize=2, label='Gradients - Main')

        plt.xlabel('iteration')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()

    return best_factors, best_outputs
