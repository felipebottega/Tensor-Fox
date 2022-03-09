"""
 Auxiliar Module
 ===============
 This module is composed by minor functions, designed to work on very specific tasks. Some of them may be useful for the 
 user to use directly, but most of them are  just some piece of another (and more important) function. 
""" 

# Python modules
from numpy import diag, dot, argsort, array, size, inf, moveaxis, arange, ndarray, unique, prod, uint64
from numpy.linalg import norm, pinv
from numpy.random import randn, randint
import sys
import scipy.io
from sklearn.utils.extmath import randomized_svd as rand_svd
from itertools import product

# Tensor Fox modules
import TensorFox.Critical as crt
import TensorFox.TensorFox as tfx


def consistency(R, dims, options):
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
        m, n, p = int(dims[0]), int(dims[1]), int(dims[2])
        if R > min(m*n, m*p, n*p):
            msg = 'Rank must satisfy 1 <= rank <= min(m*n, m*p, n*p) = ' + str(min(m*n, m*p, n*p)) + '.'
            sys.exit(msg)

    if L > 3 and R == 1:
        msg = 'Rank must be greater than 1 for tensor with order greater than 3.'
        sys.exit(msg)

    if L > 3 and R > min(dims) and options.method == 'ttcpd':
        print("        The CPD should not be computed with the Tensor Train method when the order is higher than 3 and the rank is greater than any dimension of the tensor.\n"
              "        The computations can be slower and the program may not converge to a good solution.\n"
              "        We recommend to use the parameter method='dGN' (within the class of options, see the cpd docstring).", 
              file=sys.stderr)

    if options.symm:
        for i in range(L):
            for j in range(L):
                if dims[i] != dims[j]:
                    msg = 'Symmetric tensors must have equal dimensions.'
                    sys.exit(msg)

    if options.method != 'dGN' and options.method != 'als' and options.method != 'ttcpd':
        msg = "Wrong method name. Must be 'dGN', 'als' or 'ttcpd'."
        sys.exit(msg)
        
    return


def tens2matlab(T, filename):
    """ 
    This function creates a matlab file containing the tensor T. The parameter filename should be a string.
    """
    
    # Save the tensor in matlab format.
    scipy.io.savemat(filename + '.mat', {filename: T})
    
    return


def sort_dims(T):
    """
    Change the axis of T in decreasing order. This can speed up the mlsvd function.
    """
    
    # Sparse tensor.
    if type(T) == list:
        data, idxs, dims = T
        idxs = array(idxs, dtype=uint64)
        dims = array(dims)
        ordering = argsort(-dims)
        dims = tuple(dims[ordering])
        idxs = idxs[:, ordering]
        T_sorted = [data, idxs, dims]
    
    # Dense tensor.
    else:    
        dims = array(T.shape)
        L = len(dims)
        ordering = argsort(-dims)
        T_sorted = moveaxis(T, ordering, arange(-L, 0))

    return T_sorted, ordering
        

def unsort_dims(factors, ordering):
    """
    Put the CPD factors to their original dimension ordering.
    """
    
    L = len(factors)
    new_factors = [[] for l in range(L)]
    for l in range(L):
        new_factors[ordering[l]] = factors[l]

    return new_factors


def output_info(T1, Tsize, T1_approx, 
                step_sizes_main, step_sizes_refine, 
                errors_main, errors_refine, 
                improv_main, improv_refine, 
                gradients_main, gradients_refine, 
                stop_main, stop_refine,
                options):
    """
    Constructs the class containing the information of all relevant outputs relative to the computation of a third order
    CPD.
    """

    if options.refine:
        num_steps = size(step_sizes_main) + size(step_sizes_refine)
    else:
        num_steps = size(step_sizes_main)

    if options.refine:
        rel_error = errors_refine[-1]
    elif type(T1) == ndarray:
        rel_error = crt.fastnorm(T1, T1_approx)/Tsize

    # In the sparse case, the variable T1 is the triple T = [data, idxs, dims] and T1_approx is the variable factors.
    # We keep the original variable names used for the dense case but this distinction is important to know.
    else:
        data, idxs, dims = T1
        idxs = array(idxs)
        factors = T1_approx
        rel_error = crt.sparse_fastnorm(data, idxs, dims, factors)/Tsize

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
                print('4 - Average error increased.')
            if self.stop[0] == 5:
                print('5 - Limit of iterations was reached.')
            if self.stop[0] == 6:
                print('6 - dGN diverged.')
            if self.stop[0] == 7:
                print('7 - Average improvement is too small compared to the average error.')

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
                print('4 - Average error increased.')
            if self.stop[1] == 5:
                print('5 - Limit of iterations was reached.')
            if self.stop[1] == 6:
                print('6 - dGN diverged.')
            if self.stop[1] == 7:
                print('7 - Average improvement is too small compared to the average error.')
            if self.stop[1] == 8:
                print('8 - No refinement was performed.')
           
            return ''

    output = output()

    return output


def make_final_outputs(num_steps, rel_error, accuracy, outputs, options):
    """
    Constructs the class containing the information of all relevant outputs relative to the computation of a high order
    CPD.
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
        - inner_method is the name of the method used to compute each iteration, the choices are 'cg', 'cg_static',
          'als' and 'direct'.
        - cg_maxiter is the maximum number of iterations for 'cg_static'.
        - cg_factor is the multiplying factor for the 'cg' method.
        - cg_tol is the tolerance error to stop the iterations of the inner method.
    """

    # Initialize default options.
    class temp_options:
        def __init__(self):
            self.maxiter = 200  
            self.tol = 1e-16
            self.tol_step = 1e-6
            self.tol_improv = 1e-6
            self.tol_grad = 1e-6
            self.tol_jump = 10
            self.method = 'dGN'
            self.inner_method = 'cg'
            self.cg_maxiter = 100
            self.cg_factor = 1
            self.cg_tol = 1e-16
            self.bi_method_parameters = ['als', 500, 1e-6] 
            self.initialization = 'random'
            self.trunc_dims = 0
            self.mlsvd_method = 'seq'
            self.tol_mlsvd = 1e-16
            self.init_damp = 1
            self.refine = False
            self.symm = False
            self.factors_norm = 0
            self.trials = 3
            self.display = 0
            self.epochs = 1
            self.gpu = False
            self.mkl_dot = True

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
    if 'tol_jump' in dir(options):
        temp_options.tol_jump = options.tol_jump
    if 'method' in dir(options):
        temp_options.method = options.method
        
    if 'inner_method' in dir(options):
        temp_options.inner_method = options.inner_method
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
            temp_options.bi_method_parameters[1] = 100
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
    if 'mlsvd_method' in dir(options):
        temp_options.mlsvd_method = options.mlsvd_method
    if 'tol_mlsvd' in dir(options):
        temp_options.tol_mlsvd = options.tol_mlsvd
    if 'init_damp' in dir(options):
        temp_options.init_damp = options.init_damp
    if 'refine' in dir(options):
        temp_options.refine = options.refine
    if 'symm' in dir(options):
        temp_options.symm = options.symm
    if 'factors_norm' in dir(options):
        temp_options.factors_norm = options.factors_norm
    if 'trials' in dir(options):
        temp_options.trials = options.trials
    if 'display' in dir(options):
        temp_options.display = options.display
    if 'epochs' in dir(options):
        temp_options.epochs = options.epochs
    if 'gpu' in dir(options):
        temp_options.gpu = options.gpu
    if 'mkl_dot' in dir(options):
        temp_options.mkl_dot = options.mkl_dot

    # If gpu is True, the variable mlsvd_method is set to 'gpu', which is a special strategy aiming to minimize the
    # memory size of the data passed to the GPU. This strategy is based on the classic MLSVD method. In the case the
    # user wants to use a specific MLSVD method, then the variable mlsvd_method should be passed to the function cpd.
    # This is only valid for dense tensors. Sparse tensors has its own strategy which isn't affect by mlsvd_method.
    if temp_options.gpu and 'mlsvd_method' not in dir(options):
        temp_options.mlsvd_method = 'gpu'
    
    return temp_options


def tt_core(V, dims, r1, r2, l):
    """
    Computation of one core of the CPD Tensor Train function (cpdtt).
    """

    V = V.reshape(r1*dims[l], int(prod(dims[l+1:], dtype=uint64)), order='F')
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

    if L == 3:
        G0, G1, G2 = G
        T_approx = crt.tt_error_order3(T, G0, G1, G2, dims, L)
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


def cpd_cores(G, max_trials, epochs, R, display, options):
    """
    Routines to compute the cores of the CPD tensor train.
    """
    
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
        factors, output = tfx.tricpd(G[1], R, options)
        X, Y, Z = factors
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

    low = 2
    upp = L - 2

    for epoch in range(epochs):
        
        if display < 0 < epoch:
            print()
            print('Epoch ', epoch+1)

        # Following the tensor train from G[1] to G[L-2].
        if epoch % 2 == 0:
            for l in range(low, L-1):
                best_error = inf
                fixed_X = pinv(best_Z.T)
                for trial in range(max_trials):
                    if display > 0:
                        print()
                        print('CPD', l)
                    X, Y, Z, output = tfx.bicpd(G[l], R, [fixed_X, 0], options)
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
            # low and upp must be different for the third order case.
            if L == 3:
                low = 1
                upp = L - 1
            for l in reversed(range(1, upp)):
                best_error = inf
                fixed_Z = pinv(best_X.T)
                for trial in range(max_trials):
                    if display > 0:
                        print()
                        print('CPD', l)
                    X, Y, Z, output = tfx.bicpd(G[l], R, [fixed_Z, 2], options)
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


def gen_rand_tensor(dims, R, noise=0):
    """
    This function generates a random rank-R tensor T of shape (dims[0], dims[1], ..., dims[L-1]), where L is the order
    of T. Each factor matrix of T is a matrix of shape (dims[l], R) with its entries drawn from the standard Gaussian
    distribution (mean zero and variance one).
    Let W[l] be the l-th factor matrix of T, then T = (W[0], W[1], ..., W[L-1])*I, where I is a diagonal tensor of shape
    R x R x... x R (L times).

    Input
    -----
    dims: tuple or list of ints
        The dimensions of the tensor
    R: int
        The rank of the tensor (must satisfy R < min(dims))
    noise: float
        Size of the noise added to the original tensor. Default is without noise.

    Output
    ------
    T: float L-D array 
        The tensor in coordinate format.
    T_noise: float L-D array 
        Noisy tensor in coordinate format.
    orig_factors: list
        List of the factor matrices of T. We have that orig_factors[l] = W[l], as described above.
    """

    L = len(dims)
    orig_factors = []

    for l in range(L):
        M = randn(dims[l], R)
        orig_factors.append(M)

    T = tfx.cnv.cpd2tens(orig_factors)
    
    if noise != 0:
        E = noise * randn(*dims)
        T_noise = T + E
        return T, T_noise, orig_factors
    
    else:
        return T, orig_factors


def gen_rand_sparse_tensor(dims, R, nnz):
    """
    This function generates a sparse random rank-R tensor T of shape (dims[0], dims[1], ..., dims[L-1]), where L is the
    order of T. Each factor matrix of T is a sparse matrix of shape (dims[l], R) with its entries drawn from the
    standard Gaussian distribution (mean zero and variance one). The sparse representation is given by a dictionary
    where the keys are the coordinates associated to nonzero values of the matrix. T is given in the sparse format
    [data, idxs, dims] as described in the cpd function.

    Input
    -----
    dims: tuple or list of ints
        The dimensions of the tensor
    R: int
        The rank of the tensor (must satisfy R < min(dims))
    nnz: int
        Number of nonzero entries of each factor matrix. Usually the tensor will have more nonzero entries.

    Output
    ------
    data: float 1-D arrays
        data[i] is the nonzero value of the tensor at index idxs[i, :].
    idxs: int 2-D array
        Let nnz be the number of nonzero entries of the tensor. Then idxs is an array
        of shape (nnz, L) such that idxs[i, :] is the index of the i-th nonzero entry.
    dims: list or tuple
        The dimensions (shape) of the tensor.
    orig_factors: list
        List of the factor matrices of the tensor. Each matrix is sparse and defined by a dictionary such that each key
        is a tuple with a nonzero coordinate and the corresponding value is the nonzero value of the matrix. Along with
        the dictionary there is a tuple with the dimensions of the matrix.
    """

    L = len(dims)

    # Generate factors.
    orig_factors = []
    for l in range(L):
        # Create the nonzero values of each factor.
        data = randn(nnz)

        # Create indexes.
        rows = randint(0, dims[l], size=nnz)
        cols = randint(0, R, size=nnz)
        idxs = array([[rows[i], cols[i]] for i in range(nnz)])

        # Remove duplicates and update data accordingly.
        idxs = unique(idxs, axis=0)
        data = data[:idxs.shape[0]]

        # Define sparse factors.
        orig_factors.append({tuple(idxs[i, :]): d for i, d in enumerate(data)})

    # Define sparse representation of the tensor in the format data, idxs, dims.
    idxs = []
    for l in range(L):
        idxs.append([])
        for idx in orig_factors[l].keys():
            idxs[l].append(idx[0])

    idxs = list(set(product(*(idx for idx in idxs))))

    data = []
    for j in idxs:
        tmp = 0
        for r in range(R):
            p = 1
            for l in range(L):
                if (j[l], r) in orig_factors[l].keys():
                    p *= orig_factors[l][j[l], r]
            tmp += p
        data.append(tmp)

    return array(data), array(idxs), array(dims), orig_factors
