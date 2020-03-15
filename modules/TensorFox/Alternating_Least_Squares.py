"""
 Alternating Least Squares Module
 ===============================
 This module implement the Alternating Least Squares algorithm.
"""

# Python modules
import numpy as np
from numpy import inf, mean, copy, concatenate, empty, float64, dot
from numpy.linalg import norm, pinv

# Tensor Fox modules
import TensorFox.Conversion as cnv
import TensorFox.Critical as crt
import TensorFox.MultilinearAlgebra as mlinalg


def als(T, factors, R, options):
    """
    This function uses the ALS method to compute an approximation of T with rank R. 

    Inputs
    ------
    T: float array
    factors: list of float 2-D array
        The factor matrices to be used as starting point.
    R: int. 
        The desired rank of the approximating tensor.
    options: class
        See the function cpd for more information about the options available.
    
    Outputs
    -------
    factors: list of float 2-D array
        The factor matrices of the CPD of T.
    step_sizes: float 1-D array
        Distance between the computed points at each iteration.
    errors: float 1-D array
        Error of the computed approximating tensor at each iteration. 
    improv: float 1-D array
        Improvement of the error at each iteration. More precisely, the difference between the relative error of the
        current iteration and the previous one.
    gradients: float 1-D array
        Gradient of the error function at each iteration.
    stop: 0, 1, 2, 3, 4, 5, 6 or 7
        This value indicates why the function stopped. See the function dGN for more details.
    """  

    # INITIALIZE RELEVANT VARIABLES 
    
    # Extract all relevant variables from the class of options.
    maxiter = options.maxiter
    tol = options.tol
    tol_step = options.tol_step
    tol_improv = options.tol_improv
    tol_grad = options.tol_grad
    symm = options.symm
    display = options.display
    factors_norm = options.factors_norm

    # Verify if some factor should be fixed or not. This only happens when the bicpd function was called.
    L = len(factors)
    fix_mode = -1
    orig_factors = [[] for l in range(L)]
    for l in range(L):            
        if type(factors[l]) == list:
            fix_mode = l
            orig_factors[l] = factors[l][0].copy()
            factors[l] = factors[l][0]
                
    # Set the other variables.
    Tsize = norm(T)
    error = 1
    best_error = inf
    stop = 5
    const = 1 + int(maxiter/10)
                               
    # INITIALIZE RELEVANT ARRAYS
    
    x = concatenate([factors[l].flatten('F') for l in range(L)])
    step_sizes = empty(maxiter)
    errors = empty(maxiter)
    improv = empty(maxiter)
    gradients = empty(maxiter)
    best_factors = [copy(factors[l]) for l in range(L)]

    # Compute unfoldings.
    Tl = [cnv.unfold(T, l+1) for l in range(L)]
    T1_approx = empty(Tl[0].shape, dtype=float64)

    if display > 1:
        if display == 4:
            print('   ',
                  '{:^9}'.format('Iteration'),
                  '| {:^11}'.format('Rel error'),
                  '| {:^11}'.format('Step size'),
                  '| {:^11}'.format('Improvement'),
                  '| {:^11}'.format('norm(grad)'))
        else:
            print('   ',
                  '{:^9}'.format('Iteration'),
                  '| {:^9}'.format('Rel error'),
                  '| {:^11}'.format('Step size'),
                  '| {:^10}'.format('Improvement'),
                  '| {:^10}'.format('norm(grad)'))               
    
    # START ALS ITERATIONS
    
    for it in range(maxiter):      
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_error = error
                       
        # ALS iteration call.
        factors = als_iteration(Tl, factors, fix_mode)
        x = concatenate([factors[l].flatten('F') for l in range(L)])
                                     
        # Transform factors.
        factors = cnv.transform(factors, symm, factors_norm)
        # Some mode may be fixed when the bicpd is called.
        if L == 3:
            for l in range(L):
                if fix_mode == l:
                    factors[l] = copy(orig_factors[l])
                                          
        # Compute error.
        T1_approx = cnv.cpd2unfold1(T1_approx, factors)
        error = crt.fastnorm(Tl[0], T1_approx) / Tsize

        # Update best solution.
        if error < best_error:
            best_error = error
            for l in range(L):
                best_factors[l] = copy(factors[l])
                           
        # Save relevant information about the current iteration.
        step_sizes[it] = norm(x - old_x)
        errors[it] = error
        gradients[it] = np.abs(old_error - error)/step_sizes[it]
        if it == 0:
            improv[it] = errors[it]
        else:
            improv[it] = np.abs(errors[it-1] - errors[it])

        # Show information about current iteration.
        if display > 1:
            if display == 4:
                print('    ',
                      '{:^8}'.format(it + 1),
                      '| {:^10.5e}'.format(errors[it]),
                      '| {:^10.5e}'.format(step_sizes[it]),
                      '| {:^10.5e}'.format(improv[it]),
                      '| {:^11.5e}'.format(gradients[it]))
            else:
                print('   ',
                      '{:^9}'.format(it + 1),
                      '| {:^9.2e}'.format(errors[it]),
                      '| {:^11.2e}'.format(step_sizes[it]),
                      '| {:^11.2e}'.format(improv[it]),
                      '| {:^10.2e}'.format(gradients[it]))

        # Stopping conditions.
        if it > 1:
            if errors[it] < tol:
                stop = 0
                break
            if step_sizes[it] < tol_step:
                stop = 1
                break
            if improv[it] < tol_improv:
                stop = 2
                break
            if gradients[it] < tol_grad:
                stop = 3
                break
            if it > 2*const and it % const == 0:
                # Let const=1+int(maxiter/10). Comparing the average errors of const consecutive iterations prevents
                # the program to continue iterating when the error starts to oscillate without decreasing.
                mean1 = mean(errors[it - 2 * const: it - const])
                mean2 = mean(errors[it - const: it])
                if mean1 - mean2 <= tol_improv:
                    stop = 4
                    break
                # If the average improvements is too small compared to the average errors, the program stops.
                mean3 = mean(improv[it - const: it])
                if mean3 < 1e-3 * mean2:
                    stop = 7
                    break
            # Prevent blow ups.
            if error > max(1, Tsize ** 2) / (1e-16 + tol):
                stop = 6
                break
    
    # SAVE LAST COMPUTED INFORMATION
    
    errors = errors[0: it+1]
    step_sizes = step_sizes[0: it+1]
    improv = improv[0: it+1]
    gradients = gradients[0: it+1]
    
    return factors, step_sizes, errors, improv, gradients, stop


def als_iteration(Tl, factors, fix_mode):
    """
    This function the ALS iterations, that is, it computes the pseudoinverse with respect to the modes. This 
    implementation is simple and not intended to be optimal. 
    """
    
    # Initialize first variables.
    L = len(factors)
    R = factors[0].shape[1]
    dims = [factors[l].shape[0] for l in range(L)]
    
    # Main computations for the general case.
    if fix_mode == -1:
        for l in range(L):
            itr = [l for l in reversed(range(L))]
            itr.remove(l)
            M = factors[itr[0]]

            # Compute Khatri-Rao products W^(L) ⊙ ... ⊙ W^(l+1) ⊙ W^(l-1) ⊙ ... ⊙ W^(1).
            for ll in range(L-2):
                tmp = M
                dim1, dim2 = tmp.shape[0], dims[itr[ll+1]]
                M = empty((dim1*dim2, R), dtype=float64)
                M = mlinalg.khatri_rao(tmp, factors[itr[ll+1]], M)

            factors[l] = dot(Tl[l], pinv(M.T))

        return factors
    
    # If fix_mode != -1, it is assumed that the program is using the bicpd function.
    # This part is only used for third order tensors.
    X, Y, Z = factors
    T1, T2, T3 = Tl
        
    if fix_mode == 0:
        M = empty((Z.shape[0] * X.shape[0], R))
        M = mlinalg.khatri_rao(Z, X, M)
        Y = dot( T2, pinv(M.T) )
        M = empty((Y.shape[0] * X.shape[0], R))
        M = mlinalg.khatri_rao(Y, X, M)
        Z = dot( T3, pinv(M.T) )
                
    elif fix_mode == 1:
        X, Y, Z = factors
        M = empty((Z.shape[0] * Y.shape[0], R))
        M = mlinalg.khatri_rao(Z, Y, M)
        X = dot( T1, pinv(M.T) )
        M = empty((Y.shape[0] * X.shape[0], R))
        M = mlinalg.khatri_rao(Y, X, M)
        Z = dot( T3, pinv(M.T) )
                
    elif fix_mode == 2:
        X, Y, Z = factors
        M = empty((Z.shape[0] * Y.shape[0], R))
        M = mlinalg.khatri_rao(Z, Y, M)
        X = dot( T1, pinv(M.T) )
        M = empty((Z.shape[0] * X.shape[0], R))
        M = mlinalg.khatri_rao(Z, X, M)
        Y = dot( T2, pinv(M.T) )
        
    return [X, Y, Z]       
