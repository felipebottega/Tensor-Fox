"""
 Auxiliar Module
 ===============
 This module is composed by minor functions, designed to work on very specific tasks. Some of them may be useful for the 
user to use directly, but most of them are  just some piece of another (and more important) function. 
""" 

# Python modules
from numpy import zeros, prod, diag, dot, empty, float64, argsort, array, size, inf
from numpy.linalg import norm, pinv
import sys
import warnings
import scipy.io
from sklearn.utils.extmath import randomized_svd as rand_svd
from numba import njit, prange

# Tensor Fox modules
import Critical as crt
import MultilinearAlgebra as mlinalg
import TensorFox as tfx


def consistency(R, dims, symm):
    """ 
    This function checks the validity of rank and dimensions before anything is done in the program. 
    """

    L = len(dims)

    # Check if order is not higher than 12.
    if L > 12:
        sys.exit('Tensor Fox does not work with tensors of order higher than 12.')

    # If some dimension is equal to 1, the user may just use classical SVD with numpy.
    # We won't address this situation here.
    for l in range(L):
        if dims[l] == 1:
            sys.exit('At least one dimension is equal to 1. This situation not supported by Tensor Fox.')
        
    # Consistency of rank value.
    if (type(R) != int) or (R < 1):
        sys.exit('Rank must be a positive integer.')
        
    # Check if rank is well defined in the third order case.
    if L == 3:
        m, n, p = dims[0], dims[1], dims[2]
        if R > min(m*n, m*p, n*p):
            msg = 'Rank must satisfy 1 <= rank <= min(m*n, m*p, n*p) = ' + str(min(m*n, m*p, n*p)) + '.'
            sys.exit(msg)

    if L > 3 and R == 1:
        msg = 'Rank must be greater than 1 for tensor with order greater than 3.'
        sys.exit(msg)

    if L > 3 and R > min(dims):
        warnings.warn('For tensors of order higher than 3 it is advisable that the rank is smaller or equal than at' 
                      ' least one of the dimensions of the tensor. The ideal would to be smaller or equal than all' 
                      ' dimensions. In the case this condition is not met the computations can be slower and the'
                      ' program may not converge to a good solution.', category=Warning, stacklevel=3)

    if symm:
        for i in range(L):
            for j in range(L):
                if dims[i] != dims[j]:
                    msg = 'Symmetric tensors must have equal dimensions.'
                    sys.exit(msg)
        
    return


def tens2matlab(T, filename):
    """ 
    This function creates a matlab file containing the tensor T. The parameter filename should be a string.
    """
    
    # Save the tensor in matlab format.
    scipy.io.savemat(filename + '.mat', {filename:T})
    
    return


def sort_dims(T, m, n, p):
    """
    Consider the following identifications.
        "m = 0", "n = 1", "p = 2"
    We will use them to reorder the dimensions of the tensor, in such a way that we have m_new >= n_new >= p_new.
    """

    if m >= n >= p:
        ordering = [0, 1, 2]
        return T, ordering
  
    elif p >= n >= m:
        ordering = [2, 1, 0]
       
    elif n >= p >= m:
        ordering = [1, 2, 0]

    elif m >= p >= n:
        ordering = [0, 2, 1]

    elif n >= m >= p:
        ordering = [1, 0, 2]

    elif p >= m >= n:
        ordering = [2, 0, 1]
 
    # Define m_s, n_s, p_s such that T_sorted.shape == m_s, n_s, p_s.
    m_s, n_s, p_s = T.shape[ordering[0]], T.shape[ordering[1]], T.shape[ordering[2]]
    T_sorted = empty((m_s, n_s, p_s), dtype=float64)

    # In the function sort_T, inv_sort is such that T_sorted[inv_sort[i,j,k]] == T[i,j,k].
    inv_sort = argsort(ordering)
    T_sorted = sort_T(T, T_sorted, inv_sort, m_s, n_s, p_s)

    return T_sorted, ordering
   

@njit(nogil=True)
def sort_T(T, T_sorted, inv_sort, m, n, p):
    """
    Subroutine of the function sort_dims. Here the program deals with the computationally costly part, which is the 
    assignment of values to the new tensor.
    """

    # id receives the current triple (i,j,k) at each iteration.
    idx = array([0, 0, 0])
    
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                idx[0], idx[1], idx[2] = i, j, k
                T_sorted[i, j, k] = T[idx[inv_sort[0]], idx[inv_sort[1]], idx[inv_sort[2]]]
                              
    return T_sorted
        

def unsort_dims(X, Y, Z, ordering):
    """
    Put the CPD factors and orthogonal transformations to the original ordering of dimensions.
    """

    if ordering == [0, 1, 2]:
        return X, Y, Z

    elif ordering == [0, 2, 1]:
        return X, Z, Y

    elif ordering == [1, 0, 2]:
        return Y, X, Z

    elif ordering == [1, 2, 0]:
        return Z, X, Y

    elif ordering == [2, 0, 1]:
        return Y, Z, X

    elif ordering == [2, 1, 0]:
        return Z, Y, X


def compute_error(T, Tsize, S1, U, dims):
    """
    Compute relative error between T and (U_1,...,U_L)*S using multilinear multiplication, where S.shape == dims.
    """

    T_compress = mlinalg.multilin_mult(U, S1, dims)
    error = norm(T - T_compress)/Tsize 
    return error


def output_info(T_orig, Tsize, T_approx, 
                step_sizes_main, step_sizes_refine, 
                errors_main, errors_refine, 
                improv_main, improv_refine, 
                gradients_main, gradients_refine, 
                stop_main, stop_refine,
                options):
    """
    Constructs the class containing the information of all relevant outputs relative to the computation of a 
    third order CPD.
    """

    if options.refine:
        num_steps = size(step_sizes_main) + size(step_sizes_refine)
    else:
        num_steps = size(step_sizes_main)

    rel_error = norm(T_orig - T_approx)/Tsize

    class output:
        def __init__(self):
            self.num_steps = num_steps
            self.rel_error = rel_error
            self.accuracy = max(0, 100*(1 - rel_error))
            self.step_sizes = [step_sizes_main, step_sizes_refine]
            self.errors = [errors_main, errors_refine]
            self.improv = [improv_main, improv_refine]
            self.gradients = [gradients_main, gradients_refine]
            self.stop = [stop_main, stop_refine]
            self.options = options

        def stop_msg(self):
            # stop_main message
            print()
            print('Main stop:')
            if self.stop[0] == 0:
                print('0 - Relative error is small enough.')
            if self.stop[0] == 1:
                print('1 - Steps are small enough.')
            if self.stop[0] == 2:
                print('2 - Improvement in the relative error is small enough.')
            if self.stop[0] == 3:
                print('3 - Gradient is small enough.')
            if self.stop[0] == 4:
                print('4 - Average of the last k = 1 + int(maxiter/10) relative errors is small enough.')
            if self.stop[0] == 5:
                print('5 - Limit of iterations was reached.')
            if self.stop[0] == 6:
                print('6 - dGN diverged.')

            # stop_refine message
            print()
            print('Refinement stop:')
            if self.stop[1] == 0:
                print('0 - Relative error is small enough.')
            if self.stop[1] == 1:
                print('1 - Steps are small enough.')
            if self.stop[1] == 2:
                print('2 - Improvement in the relative error is small enough.')
            if self.stop[1] == 3:
                print('3 - Gradient is small enough.')
            if self.stop[1] == 4:
                print('4 - Average of the last k = 1 + int(maxiter/10) relative errors is small enough.')
            if self.stop[1] == 5:
                print('5 - Limit of iterations was reached.')
            if self.stop[1] == 6:
                print('6 - dGN diverged.')
            if self.stop[1] == 7:
                print('7 - No refinement was performed.')
           
            return ''

    output = output()

    return output


def make_final_outputs(num_steps, rel_error, accuracy, outputs, options):
    """
    Constructs the class containing the information of all relevant outputs relative to the computation of a 
    high order CPD.
    """

    class temp_outputs:
        def __init__(self):
            self.num_steps = num_steps
            self.rel_error = rel_error
            self.accuracy = accuracy
            self.cpd_output = outputs
            self.options = options

    final_outputs = temp_outputs()
   
    return final_outputs


def make_options(options):
    """
    This function constructs the whole class of options based on the options the user requested. 
    This is the format read by the program.

    Some observations about the CG parameters:
        - method is the method used to compute each iteration, the choices are 'cg', 'cg_static' and 'als'.
        - cg_maxiter is the maximum number of iterations for 'cg_static'.
        - cg_factor is the multiplying factor cg_factor for 'cg'. 
        - cg_tol is the tolerance error to stop the iterations of the method.
    """

    # Initialize default options.
    class temp_options:
        def __init__(self):
            self.maxiter = 200  
            self.tol = 1e-6
            self.tol_step = 1e-6
            self.tol_improv = 1e-6
            self.tol_grad = 1e-6
            self.method = 'cg'
            self.cg_maxiter = 300
            self.cg_factor = 1
            self.cg_tol = 1e-6
            self.bi_method_parameters = ['als', 500, 1e-6] 
            self.initialization = 'random'
            self.trunc_dims = 0
            self.tol_mlsvd = 1e-16
            self.init_damp = 1
            self.refine = False
            self.symm = False
            self.constraints = [0, 0, 0]
            self.factors_norm = 0
            self.trials = 10
            self.display = 0
            self.epochs = 1

    temp_options = temp_options()

    # User defined options.
    if 'maxiter' in dir(options):
        temp_options.maxiter = options.maxiter
    if 'tol' in dir(options):
        temp_options.tol = options.tol
    if 'tol_step' in dir(options):
        temp_options.tol_step = options.tol_step
    if 'tol_improv' in dir(options):
        temp_options.tol_improv = options.tol_improv
    if 'tol_grad' in dir(options):
        temp_options.tol_grad = options.tol_grad
        
    if 'method' in dir(options):
        temp_options.method = options.method
    if 'cg_maxiter' in dir(options):
        temp_options.cg_maxiter = options.cg_maxiter
    if 'cg_factor' in dir(options):
        temp_options.cg_factor = options.cg_factor   
    if 'cg_tol' in dir(options):
        temp_options.cg_tol = options.cg_tol 
        
    if 'bi_method' in dir(options):
        temp_options.bi_method_parameters[0] = options.bi_method
        # Set default maxiter for each possible algorithm (bicpd).
        if options.bi_method == 'cg':
            temp_options.bi_method_parameters[1] = 1
        elif options.bi_method == 'cg_static':
            temp_options.bi_method_parameters[1] = 300
        elif options.bi_method == 'als':
            temp_options.bi_method_parameters[1] = 500
    if 'bi_method_maxiter' in dir(options):
        temp_options.bi_method_parameters[1] = options.bi_method_maxiter   
    if 'bi_method_tol' in dir(options):
        temp_options.bi_method_parameters[2] = options.bi_method_tol    
        
    if 'initialization' in dir(options):
        temp_options.initialization = options.initialization
    if 'trunc_dims' in dir(options):
        temp_options.trunc_dims = options.trunc_dims
    if 'tol_mlsvd' in dir(options):
        temp_options.tol_mlsvd = options.tol_mlsvd
    if 'init_damp' in dir(options):
        temp_options.init_damp = options.init_damp
    if 'refine' in dir(options):
        temp_options.refine = options.refine
    if 'symm' in dir(options):
        temp_options.symm = options.symm
    if 'low' in dir(options):
        temp_options.constraints[0] = options.low
    if 'upp' in dir(options):
        temp_options.constraints[1] = options.upp
    if 'factor' in dir(options):
        temp_options.constraints[2] = options.factor
    if 'factors_norm' in dir(options):
        temp_options.factors_norm = options.factors_norm
    if 'trials' in dir(options):
        temp_options.trials = options.trials
    if 'display' in dir(options):
        temp_options.display = options.display
    if 'epochs' in dir(options):
        temp_options.epochs = options.epochs
    
    return temp_options


def tt_core(V, dims, r1, r2, l):
    """
    Computation of one core of the CPD Tensor Train function (cpdtt).
    """

    V = V.reshape(r1*dims[l], prod(dims[l+1:]), order='F')
    low_rank = min(V.shape[0], V.shape[1])
    U, S, V = rand_svd(V, low_rank, n_iter=0)
    U = U[:, :r2]
    S = diag(S)
    V = dot(S, V)
    V = V[:r2, :]
    if r1 == 1:
        g = U.reshape(dims[l], r2, order='F') 
    else:
        g = U.reshape(r1, dims[l], r2, order='F')   
    return V, g


def tt_error(T, G, dims, L):
    """
    Given a tensor T and a computed CPD Tensor Train G = (G1,...,GL), this function computes the error between T and the 
    tensor associated to G.
    """

    if L == 4:
        G0, G1, G2, G3 = G
        T_approx = crt.tt_error_order4(T, G0, G1, G2, G3, dims, L)
    if L == 5:
        G0, G1, G2, G3, G4 = G
        T_approx = crt.tt_error_order5(T, G0, G1, G2, G3, G4, dims, L)
    if L == 6:
        G0, G1, G2, G3, G4, G5 = G
        T_approx = crt.tt_error_order6(T, G0, G1, G2, G3, G4, G5, dims, L)
    if L == 7:
        G0, G1, G2, G3, G4, G5, G6 = G
        T_approx = crt.tt_error_order7(T, G0, G1, G2, G3, G4, G5, G6, dims, L)
    if L == 8:
        G0, G1, G2, G3, G4, G5, G6, G7 = G
        T_approx = crt.tt_error_order8(T, G0, G1, G2, G3, G4, G5, G6, G7, dims, L)
    if L == 9:
        G0, G1, G2, G3, G4, G5, G6, G7, G8 = G
        T_approx = crt.tt_error_order9(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, dims, L)
    if L == 10:
        G0, G1, G2, G3, G4, G5, G6, G7, G8, G9 = G
        T_approx = crt.tt_error_order10(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, dims, L)
    if L == 11:
        G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10 = G
        T_approx = crt.tt_error_order11(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, dims, L)
    if L == 12:
        G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11 = G
        T_approx = crt.tt_error_order12(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, dims, L)

    error = norm(T - T_approx)/norm(T)
    return error


@njit(nogil=True, parallel=True)
def rank1(X, Y, Z, m, n, R, k):
    """
    Compute each rank 1 term of the CPD given by X, Y, Z. Them this function converts these factors into a matrix, which 
    is the first frontal slice of the tensor in coordinates obtained by this rank 1 term. By doing this for all R terms,
    we have a tensor with R slices, each one representing a rank-1 term of the original CPD.

    Inputs
    ------
    X, Y, Z: 2-D float ndarray
        The CPD factors of some tensor.
    m, n, p, R: int
    k: int
        Slice we want to compute.

    Outputs
    -------
    rank1_sections: 3-D float ndarray
        Each matrix rank1_slices[:, :, l] is the k-th slices associated with the l-th factor in the CPD of some tensor.
    """
    
    # Each frontal slice of rank1_slices is the coordinate representation of a
    # rank one term of the CPD given by (X,Y,Z)*Lambda.
    rank1_slices = zeros((m, n, R), dtype=float64)

    for r in prange(R):
        for i in range(m):
            for j in range(n):
                rank1_slices[i, j, r] = X[i, r]*Y[j, r]*Z[k, r]
                        
    return rank1_slices


def cpd_cores(G, max_trials, epochs, R, display, options):
    
    L = len(G)
    
    # The number of epochs is increased in 1 if necessary to be odd.
    if epochs % 2 == 0:
        epochs += 1
    
    # List of CPD's.
    cpd_list = [l for l in range(L-2)]
    
    # Outputs is a list containing the output class of each CPD.
    outputs = [l for l in range(L-2)]
    
    if display < 0 and epochs > 1:
        print('Epoch ', 1)
        
    # Compute cpd of second core.
    best_error = inf
    for trial in range(max_trials):
        if display > 0:
            print()
            print('CPD 1')
        X, Y, Z, T_approx, output = tfx.tricpd(G[1], R, options)
        if output.rel_error < best_error:
            best_output = output
            best_error = output.rel_error
            best_X, best_Y, best_Z = X, Y, Z
            if best_error < 1e-4:
                break
                
    outputs[0] = best_output
    cpd_list[0] = [best_X, best_Y, best_Z]
        
    if display < 0:
        print('CPD 1 error =', best_error)
                
    for epoch in range(epochs):
        
        if display < 0 < epoch:
            print()
            print('Epoch ', epoch+1)
    
        # Following the tensor train from G[1] to G[L-2].
        if epoch % 2 == 0:
            for l in range(2, L-1):
                best_error = inf
                fixed_X = pinv(best_Z.T)
                for trial in range(max_trials):
                    if display > 0:
                        print()
                        print('CPD', l)
                    X, Y, Z, T_approx, output = tfx.bicpd(G[l], R, [fixed_X, 0], options)
                    if output.rel_error < best_error:
                        best_output = output
                        best_error = output.rel_error
                        best_X, best_Y, best_Z = fixed_X, Y, Z
                        if best_error < 1e-4:
                            break
                
                if epoch == epochs-1:
                    outputs[l-1] = best_output
                    cpd_list[l-1] = [fixed_X, best_Y, best_Z]
                    
                if display < 0:
                    print('CPD', l, 'error =', best_error)

        # Following the tensor train backwards, from G[L-2] to G[L].
        else:
            for l in reversed(range(1, L-2)):
                best_error = inf
                fixed_Z = pinv(best_X.T)
                for trial in range(max_trials):
                    if display > 0:
                        print()
                        print('CPD', l)
                    X, Y, Z, T_approx, output = tfx.bicpd(G[l], R, [fixed_Z, 2], options)
                    if output.rel_error < best_error:
                        best_output = output
                        best_error = output.rel_error
                        best_X, best_Y, best_Z = X, Y, fixed_Z
                        if best_error < 1e-4:
                            break
                            
                if epoch == epochs-2:
                    outputs[l-1] = best_output
                    cpd_list[l-1] = [best_X, best_Y, fixed_Z]
                            
                if display < 0:
                    print('CPD', l, 'error =', best_error)
                   
    return cpd_list, outputs, best_Z
