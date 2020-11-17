"""
 Gauss-Newton Module
 ===================
 This module implement the damped Gauss-Newton algorithm, with iterations performed with aid of the conjugate gradient 
 method.

 References
 ==========
 - K. Madsen, H. B. Nielsen, and O. Tingleff, Methods for Non-Linear Least Squares Problems, 2nd edition, Informatics
   and Mathematical Modelling, Technical University of Denmark, 2004.
"""

# Python modules
import numpy as np
from numpy import inf, mean, concatenate, empty, array, zeros, ones, identity, float64, sqrt, dot, nan, diag, exp, sign
from numpy.linalg import norm, solve, qr, LinAlgError
from numpy.random import randint
import sys
from numba import njit
from copy import deepcopy

# Tensor Fox modules
import TensorFox.Alternating_Least_Squares as alsq
import TensorFox.Conversion as cnv
import TensorFox.Critical as crt
import TensorFox.MultilinearAlgebra as mlinalg


def dGN(T, factors, R, options):
    """
    This function uses the Damped Gauss-Newton method to compute an approximation of T with rank R. A starting point to
    initiate the iterations must be given. This point is given by the parameter factors.
    The Damped Gauss-Newton method is an iterative method, updating a point x at each iteration. The last computed x is
    gives an approximate CPD in flat form, and from this we have the components to form the actual CPD.

    Inputs
    ------
    T: float array
    factors: list of 2-D arrays
        The factor matrices used as starting point.
    R: int
        The desired rank of the approximating tensor.
    options: class
        Class with the options. See the Auxiliar module documentation for more information.

    Outputs
    -------
    best_factors: list of 2-D arrays
        The factor matrices of the approximated CPD of T.
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
        This value indicates why the dGN function stopped. Below we summarize the cases.
        0: errors[it] < tol. Relative error is small enough.
        1: step_sizes[it] < tol_steps. Steps are small enough.
        2: improv[it] < tol_improv. Improvement in the relative error is small enough.
        3: gradients[it] < tol_grad. Gradient is small enough (infinity norm).
        4: mean(abs(errors[it-k : it] - errors[it-k-1 : it-1]))/Tsize < 10*tol_improv. Average of the last
            k = 1 + int(maxiter/10) relative errors is small enough. Keeping track of the averages is useful when the
            errors improvements are just a little above the threshold for a long time. We want them above the threshold
            indeed, but not too close for a long time.
        5: limit of iterations reached.
        6: dGN diverged.
        7: Average improvement is too small compared to the average error.
        8: no refinement was performed (this is not really a stopping condition, but it is necessary to indicate when
        the program can't give a stopping condition in the refinement stage).
    """

    # INITIALIZE RELEVANT VARIABLES 

    # Extract all relevant variables from the class of options.
    init_damp = options.init_damp
    maxiter = options.maxiter
    tol = options.tol
    tol_step = options.tol_step
    tol_improv = options.tol_improv
    tol_grad = options.tol_grad
    tol_jump = options.tol_jump
    symm = options.symm
    display = options.display
    factors_norm = options.factors_norm
    inner_method = options.inner_method 
    cg_maxiter = options.cg_maxiter 
    cg_factor = options.cg_factor 
    cg_tol = options.cg_tol

    # Verify if some factor should be fixed or not. This only happens when the bicpd function was called.
    L = len(factors)
    fix_mode = -1
    orig_factors = [[] for l in range(L)]
    for l in range(L):
        if type(factors[l]) == list:
            fix_mode = l
            orig_factors[l] = deepcopy(factors[l][0])
            factors[l] = factors[l][0]

    # Set the other variables.
    dims = T.shape
    Tsize = norm(T)
    error = 1
    best_error = inf
    stop = 5
    if type(init_damp) == list:
        damp = init_damp[0]
    else:
        damp = init_damp * mean(np.abs(T))
    const = 1 + int(maxiter / 10)
    
    # The program is encouraged to make more iterations for small problems. 
    if R * sum(dims) <= 100: 
        tol = 0
        tol_step = 0
        tol_improv = 0
        tol_grad = 0

    # INITIALIZE RELEVANT ARRAYS

    x = concatenate([factors[l].flatten('F') for l in range(L)])
    y = zeros(R * sum(dims), dtype=float64)
    step_sizes = zeros(maxiter)
    errors = zeros(maxiter)
    improv = zeros(maxiter)
    gradients = zeros(maxiter)
    best_factors = deepcopy(factors)

    # Prepare data to use in each Gauss-Newton iteration.
    data = prepare_data(dims, R)

    # Compute unfoldings.
    Tl = [cnv.unfold_C(T, l+1) for l in range(L)]
    T1_approx = zeros(Tl[0].shape, dtype=float64)

    if display > 1:
        if display == 4:
            print('   ',
                  '{:^9}'.format('Iteration'),
                  '| {:^11}'.format('Rel error'),
                  '| {:^11}'.format('Step size'),
                  '| {:^11}'.format('Improvement'),
                  '| {:^11}'.format('norm(grad)'),
                  '| {:^11}'.format('Predicted error'),
                  '| {:^10}'.format('# Inner iterations'))
        else:
            print('   ',
                  '{:^9}'.format('Iteration'),
                  '| {:^9}'.format('Rel error'),
                  '| {:^11}'.format('Step size'),
                  '| {:^10}'.format('Improvement'),
                  '| {:^10}'.format('norm(grad)'),
                  '| {:^10}'.format('Predicted error'),
                  '| {:^10}'.format('# Inner iterations'))

            # START GAUSS-NEWTON ITERATIONS

    for it in range(maxiter):
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_error = error

        # Computation of the Gauss-Newton iteration formula to obtain the new point x + y, where x is the 
        # previous point and y is the new step obtained as the solution of min_y |Ay - b|, with 
        inner_parameters = damp, inner_method, cg_maxiter, cg_factor, cg_tol, tol_jump, symm, factors_norm, fix_mode
        T1_approx, factors, x, y, grad, itn, residualnorm, error = \
            compute_step(Tsize, Tl, T1_approx, factors, orig_factors, data, x, y, inner_parameters, it, old_error)

        # Update gain ratio and damping parameter. 
        gain_ratio = update_gain_ratio(damp, old_error, error, Tsize, old_x, x, grad)
        damp = update_damp(damp, init_damp, gain_ratio, it)

        # Update best solution.
        if error < best_error:
            best_error = error
            best_factors = deepcopy(factors)

        # Save relevant information about the current iteration.
        errors[it] = error
        step_sizes[it] = norm(x - old_x) / norm(old_x)
        gradients[it] = norm(grad, inf)
        if it == 0:
            improv[it] = errors[it]
        else:
            improv[it] = np.abs(errors[it] - errors[it-1])

        # Show information about current iteration.
        if display > 1:
            if display == 4:
                print('    ',
                      '{:^8}'.format(it+1),
                      '| {:^10.5e}'.format(errors[it]),
                      '| {:^10.5e}'.format(step_sizes[it]),
                      '| {:^10.5e}'.format(improv[it]),
                      '| {:^11.5e}'.format(gradients[it]),
                      '| {:^15.5e}'.format(residualnorm),
                      '| {:^16}'.format(itn))
            else:
                print('   ',
                      '{:^9}'.format(it+1),
                      '| {:^9.2e}'.format(errors[it]),
                      '| {:^11.2e}'.format(step_sizes[it]),
                      '| {:^11.2e}'.format(improv[it]),
                      '| {:^10.2e}'.format(gradients[it]),
                      '| {:^15.2e}'.format(residualnorm),
                      '| {:^16}'.format(itn))

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
                mean1 = mean(errors[it - 2*const: it - const])
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

    return best_factors, step_sizes, errors, improv, gradients, stop


def compute_step(Tsize, Tl, T1_approx, factors, orig_factors, data, x, y, inner_parameters, it, old_error):
    """    
    This function uses the chosen inner method to compute the next step.
    """

    # Initialize first variables.
    L = len(factors)
    damp, inner_method, cg_maxiter, cg_factor, cg_tol, tol_jump, symm, factors_norm, fix_mode = inner_parameters
    if type(inner_method) == list:
        inner_method = inner_method[it]

    # Call the inner method.
    if inner_method == 'cg':
        cg_maxiter = 1 + (L-2) * int(cg_factor * randint(1 + it**0.4, 2 + it**0.9))
        y, grad, JT_J_grad, itn, residualnorm = cg(Tl, factors, data, y, damp, cg_maxiter, cg_tol)
        
    elif inner_method == 'cg_static':
        y, grad, JT_J_grad, itn, residualnorm = cg(Tl, factors, data, y, damp, cg_maxiter, cg_tol)

    elif inner_method == 'als':
        factors = alsq.als_iteration(Tl, factors, fix_mode)
        x = concatenate([factors[l].flatten('F') for l in range(L)])
        y *= 0
        
    elif inner_method == 'direct':
        y, grad, itn, residualnorm = direct(Tl, factors, data, y, damp)

    else:
        sys.exit("Wrong inner method name. Must be 'cg', 'cg_static', 'als' or 'direct'.")

    # Update results.
    x = x + y

    # Balance and transform factors.
    factors = cnv.x2cpd(x, factors)
    factors = cnv.transform(factors, symm, factors_norm)

    # Some mode may be fixed when the bicpd is called.
    if L == 3:
        for l in range(L):
            if fix_mode == l:
                factors[l] = deepcopy(orig_factors[l])

    # Compute error.
    T1_approx = cnv.cpd2unfold1(T1_approx, factors)
    error = crt.fastnorm(Tl[0], T1_approx) / Tsize
    
    # Sometimes the step is too bad and increase the error by much. In this case we discard the computed step and
    # use the DogLeg method to compute the next step.
    if it > 3:
        if inner_method == 'cg' or inner_method == 'cg_static':
            if error > tol_jump * old_error:
                x = x - y
                factors, x, y, error = compute_dogleg_steps(Tsize, Tl, T1_approx, factors, grad, JT_J_grad, x, y, error, inner_parameters)

    if inner_method == 'als':
        return T1_approx, factors, x, y, [nan], '-', Tsize*error, error

    return T1_approx, factors, x, y, grad, itn, residualnorm, error


def cg(Tl, factors, data, y, damp, maxiter, tol):
    """
    Conjugate gradient algorithm specialized to the tensor case.
    """

    L = len(factors)
    R = factors[0].shape[1]
    dims = array([factors[l].shape[0] for l in range(L)])
    # The program is encouraged to make more CG iterations for small problems. 
    if R * sum(dims) > 100: 
        maxiter = min(maxiter, R * sum(dims))
    else:
        tol = 0

    # Give names to the arrays.
    Gr, P1, P2, A, B, P_VT_W, result, result_tmp, Gamma, gamma, sum_dims, M, residual_cg, P, Q, z, g, JT_J_grad, N, gg = data

    # Compute the values of all arrays.
    Gr, P1, P2 = gramians(factors, Gr, P1, P2)
    Gamma, gamma = regularization(Gamma, gamma, P1, dims, sum_dims)
    M = precond(Gamma, gamma, M, damp, dims, sum_dims)    
    y *= 0

    # Compute grad.
    grad = -compute_grad(Tl, factors, P1, g, N, gg, dims, sum_dims)

    # Compute J^T*J*grad.
    V = [ grad[sum_dims[l]: sum_dims[l+1]].reshape(R, dims[l]) for l in range(L) ]
    for l in range(L):
        dot(V[l], factors[l], out=A[l])
        dot(V[l].T, P1[l], out=B[l])        
    JT_J_grad = matvec(factors, P2, P_VT_W, result, result_tmp, JT_J_grad, A, B, dims, sum_dims)

    # Compute initial variables for CG.        
    residual_cg = M * grad
    P = residual_cg
    residualnorm = dot(residual_cg.T, residual_cg)
    if residualnorm == 0.0:
        residualnorm = 1e-6

    # CG iterations.
    y, itn, residualnorm = cg_iterations(factors, P1, P2, A, B, P_VT_W, result, result_tmp, M, P,
                                         Gamma, damp, z, residual_cg, residualnorm, y, tol, maxiter, dims, sum_dims)

    return M * y, grad, JT_J_grad, itn + 1, residualnorm


def cg_iterations(factors, P1, P2, A, B, P_VT_W, result, result_tmp, M, P,
                  Gamma, damp, z, residual_cg, residualnorm, y, tol, maxiter, dims, sum_dims):
    """
    Conjugate gradient iterations.
    """

    L = len(dims)
    R = factors[0].shape[1]

    for itn in range(maxiter):
        Q = M * P
        V = [ Q[sum_dims[l]: sum_dims[l+1]].reshape(R, dims[l]) for l in range(L) ]
        
        for l in range(L):
            dot(V[l], factors[l], out=A[l])
            dot(V[l].T, P1[l], out=B[l])
            
        z = matvec(factors, P2, P_VT_W, result, result_tmp, z, A, B, dims, sum_dims) + damp * Gamma * Q
        z = M * z
        denominator = dot(P.T, z)
        if denominator == 0.0:
            denominator = 1e-6
            
        # Updates.    
        alpha = residualnorm / denominator
        y += alpha * P
        residual_cg -= alpha * z
        residualnorm_new = dot(residual_cg.T, residual_cg)
        beta = residualnorm_new / residualnorm
        residualnorm = residualnorm_new
        P = residual_cg + beta * P

        # Stopping condition.
        if residualnorm <= tol:
            break

    return y, itn, residualnorm


def compute_grad(Tl, factors, P1, g, N, gg, dims, sum_dims):
    """
    This function computes the gradient of the error function.
    """

    # Initialize first variables.
    L = len(factors)
    R = factors[0].shape[1]

    # Main computations.
    for l in range(L):
        itr = [l for l in reversed(range(L))]
        itr.remove(l)
        M = factors[itr[0]]

        # Compute Khatri-Rao products W^(L) ⊙ ... ⊙ W^(l+1) ⊙ W^(l-1) ⊙ ... ⊙ W^(1).
        for ll in range(L-2):
            tmp = M
            dim1, dim2 = tmp.shape[0], dims[itr[ll+1]]
            M = empty((dim1 * dim2, R), dtype=float64)
            M = mlinalg.khatri_rao(tmp, factors[itr[ll+1]], M)

        dot(Tl[l], M, out=N[l])
        dot(factors[l], P1[l], out=gg[l])
        g[sum_dims[l]: sum_dims[l+1]] = (gg[l] - N[l]).T.ravel()

    return g


def prepare_data(dims, R):
    """
    Initialize all necessary matrices to keep the values of several computations during the program.
    """

    L = len(dims)

    # Gramians
    Gr = zeros((L, R, R), dtype=float64)
    P1 = ones((L, R, R), dtype=float64)
    P2 = ones((L, L, R, R), dtype=float64)

    # Initializations of matrices to receive the results of the computations.
    A = zeros((L, R, R), dtype=float64)
    B = [zeros((dims[l], R)) for l in range(L)]
    P_VT_W = zeros((R, R), dtype=float64)
    result = [zeros((dims[l], R), dtype=float64) for l in range(L)]
    result_tmp = zeros((L, R, R), dtype=float64)

    # Matrices to use when constructing the Tikhonov matrix for regularization.
    Gamma = zeros(R * sum(dims), dtype=float64)
    gamma = zeros((L, R), dtype=float64)

    # Arrays to be used in the Conjugated Gradient.
    sum_dims = array([R * sum(dims[0:l]) for l in range(L+1)])
    M = ones(R * sum(dims), dtype=float64)
    residual_cg = zeros(R * sum(dims), dtype=float64)
    P = zeros(R * sum(dims), dtype=float64)
    Q = zeros(R * sum(dims), dtype=float64)
    z = zeros(R * sum(dims), dtype=float64)
    JT_J_grad = zeros(R * sum(dims), dtype=float64)
    N = [zeros((dims[l], R), dtype=float64) for l in range(L)]
    gg = [zeros((dims[l], R), dtype=float64) for l in range(L)]

    # Arrays to be used in the compute_grad function.
    g = zeros(R * sum(dims), dtype=float64)

    data = [Gr, P1, P2, A, B, P_VT_W, result, result_tmp, Gamma, gamma, sum_dims, M, residual_cg, P, Q, z, g, JT_J_grad, N, gg]

    return data


def update_gain_ratio(damp, old_error, error, Tsize, old_x, x, grad):
    """
    Update gain ratio.
    """
    
    numerator = (Tsize**2) * (old_error**2 - error**2)
    denominator = dot(x - old_x, grad + damp*(x - old_x))
    
    if denominator != 0.0:
        gain_ratio = gain_ratio = numerator / denominator
    elif numerator != 0.0 and denominator == 0.0:
        gain_ratio = sign(numerator) * inf
    else:
        gain_ratio = - inf
    
    return gain_ratio


def update_damp(damp, init_damp, gain_ratio, it):
    """
    Update damping parameter.
    """

    if type(init_damp) == list:
        damp = init_damp[it]
    else:
        if gain_ratio < 0.25:
            damp = 3*damp
        elif gain_ratio > 0.75:
            damp = damp/3

    return damp


def update_delta(delta, gain_ratio, step_size):
    """
    Update trust region radius.
    """    

    if gain_ratio > 0.75:
        delta = max(delta, 2*step_size)
    else:
        sigma = 0.25/(1 + exp(-14*(gain_ratio - 0.75))) + 0.75
        delta = min(sigma*delta, delta)
        
    return delta


def gramians(factors, Gr, P1, P2):
    """ 
    Computes all Gramian matrices of the factor matrices. Also it computes all Hadamard products between the 
    different Gramians.
    """

    L = len(factors)
    R = factors[0].shape[1]

    for l in range(L):
        Gr[l] = dot(factors[l].T, factors[l], out=Gr[l])

    for l in range(1, L):
        for ll in range(l):
            P2[l][ll] = ones((R, R), dtype=float64)
            itr = [i for i in range(L)]
            itr.remove(l)
            itr.remove(ll)
            for lll in itr:
                P2[l][ll] = mlinalg.hadamard(P2[l][ll], Gr[lll], P2[l][ll])
            P2[ll][l] = P2[l][ll]
        if l < L-1:
            P1[l] = mlinalg.hadamard(P2[l][ll], Gr[ll], P1[l])
        else:
            P1[l] = mlinalg.hadamard(P2[l][0], Gr[0], P1[l])

    P1[0] = mlinalg.hadamard(P2[0][1], Gr[1], P1[0])

    return Gr, P1, P2


def matvec(factors, P2, P_VT_W, result, result_tmp, z, A, B, dims, sum_dims):
    """
    Makes the matrix-vector computation (Jf^T * Jf)*v.
    """

    L = len(factors)
    R = factors[0].shape[1]
    
    result_tmp *= 0    
    result_tmp = matvec_inner(A, P2, P_VT_W, result_tmp, L)
                
    for l in range(L):    
        dot(factors[l], result_tmp[l], out=result[l])
        result[l] += B[l]
        z[sum_dims[l]: sum_dims[l+1]] = cnv.vec(result[l], z[sum_dims[l]: sum_dims[l+1]], dims[l], R)

    return z


@njit(nogil=True)
def matvec_inner(A, P2, P_VT_W, result_tmp, L):
    for ll in range(L):
        X = A[ll]
        for l in range(ll):
            P_VT_W = mlinalg.hadamard(P2[l][ll], X, P_VT_W)
            result_tmp[l] += P_VT_W
        for l in range(ll+1, L):
            P_VT_W = mlinalg.hadamard(P2[l][ll], X, P_VT_W)
            result_tmp[l] += P_VT_W
                
    return result_tmp


@njit(nogil=True)
def regularization(Gamma, gamma, P1, dims, sum_dims):
    """
    Computes the Tikhonov matrix Gamma, where Gamma is a diagonal matrix designed specifically to make Jf^T * Jf + Gamma
    diagonally dominant.
    """

    L, R = gamma.shape

    s = 0
    for l in range(L):
        tmp = np.max(np.abs(P1[l]))
        if s < tmp:
            s = tmp
        for r in range(R):
            gamma[l, r] = abs(P1[l][r, r])
    gamma = tmp * np.sqrt(gamma)

    for l in range(L):
        for r in range(R):
            Gamma[sum_dims[l] + r*dims[l]: sum_dims[l] + (r+1)*dims[l]] = gamma[l, r]

    return Gamma, gamma


@njit(nogil=True)
def precond(Gamma, gamma, M, damp, dims, sum_dims):
    """
    This function constructs a preconditioner in order to accelerate the Conjugate Gradient function. It is a diagonal
    preconditioner designed to make Jf^T*J + damp*I a unit diagonal matrix. Since the matrix is diagonally dominant,
    the result will be close to the identity matrix (the equalize function does that). Therefore, it will have its 
    eigenvalues clustered together.
    """

    L, R = gamma.shape

    for l in range(L):
        for r in range(R):
            M[sum_dims[l] + r*dims[l]: sum_dims[l] + (r+1)*dims[l]] = \
                gamma[l, r]**2 + damp**2 * Gamma[sum_dims[l] + r*dims[l]: sum_dims[l] + (r+1)*dims[l]]**2

    M = 1/sqrt(M)

    return M


def direct(Tl, factors, data, y, damp):
    """
    This function computes the next dGN step using a direct method. It is very heavy computationally since it
    constructs the full Hessian matrix. Do not use this function for large problems.
    """

    L = len(factors)
    R = factors[0].shape[1]
    dims = [factors[l].shape[0] for l in range(L)]

    # Give names to the arrays.
    Gr, P1, P2, A, B, P_VT_W, result, result_tmp, Gamma, gamma, sum_dims, M, residual_cg, P, Q, z, g, JT_J_grad, N, gg = data

    # Compute the values of all arrays.
    Gr, P1, P2 = gramians(factors, Gr, P1, P2)
    Gamma, gamma = regularization(Gamma, gamma, P1, dims, sum_dims)
    M = precond(Gamma, gamma, M, damp, dims, sum_dims)
    grad = -compute_grad(Tl, factors, P1, g, N, gg, dims, sum_dims)
    H = hessian(factors, P1, P2, sum_dims)
    Hd = H + damp * diag(Gamma)
    MHd = ((Hd.T) * (M**2)).T
    Mgrad = (M**2) * grad            
    
    # Solve equation MH*y = Mgrad using QR decomposition, followed by a triangular system.
    try:
        q, r = qr(MHd)
        y = dot(q.T, Mgrad)
        y = solve(r, y)
    except LinAlgError:
        y *= 0
        
    residualnorm = norm(dot(H, y) - grad)
    
    return y, grad, '-', residualnorm 


def hessian(factors, P1, P2, sum_dims):
    """
    Approximate Hessian matrix of the error function.
    """

    L = len(factors)
    R = factors[0].shape[1]
    dims = [factors[l].shape[0] for l in range(L)]
    H = zeros((R * sum(dims), R * sum(dims)))
    vec_factors = [zeros(R*dims[l]) for l in range(L)]
    fortran_factors = [array(factors[l], order='F') for l in range(L)]
    for l in range(L):
        vec_factors[l] = cnv.vec(factors[l], vec_factors[l], dims[l], R) 
        vec_factors[l] = vec_factors[l].reshape(R*dims[l], 1).T
                
    for l in range(L):
        I = identity(dims[l])
        # Block H_{ll}.
        H[sum_dims[l]:sum_dims[l+1], sum_dims[l]:sum_dims[l+1]] = mlinalg.kronecker(P1[l, :, :], I)
        for ll in range(l):              
            I = ones((dims[l], dims[ll]))
            tmp1 = mlinalg.kronecker(P2[l, ll, :, :], I)
            tmp2 = zeros((R*dims[l], R*dims[ll]))
            tmp2 = compute_blocks(tmp2, fortran_factors[l], vec_factors[ll], tuple(dims), R, l, ll)  
            # Blocks H_{l, ll} and H_{ll, l}.
            H[sum_dims[l]:sum_dims[l+1], sum_dims[ll]:sum_dims[ll+1]] = \
                mlinalg.hadamard(tmp1, tmp2, H[sum_dims[l]:sum_dims[l+1], sum_dims[ll]:sum_dims[ll+1]])  
            H[sum_dims[ll]:sum_dims[ll+1], sum_dims[l]:sum_dims[l+1]] = H[sum_dims[l]:sum_dims[l+1], sum_dims[ll]:sum_dims[ll+1]].T               
              
    return H


@njit(nogil=True)
def compute_blocks(tmp2, factor, vec, dims, R, l, ll):
    """
    Auxiliary function for the hessian function. The computation of the rank one matrices between the factor matrices
    (factors[l][:, r] * factors[ll][:, rr].T) are done here.
    """
    
    for r in range(R):
        tmp = dot(factor[:, r].reshape(dims[l], 1), vec)
        for rr in range(R):
            tmp2[dims[l]*rr:dims[l]*(rr+1), dims[ll]*r:dims[ll]*(r+1)] = tmp[:, dims[ll]*rr:dims[ll]*(rr+1)]
            
    return tmp2


def compute_dogleg_steps(Tsize, Tl, T1_approx, factors, grad, JT_J_grad, x, y, error, inner_parameters):
    """
    Compute Dogleg step.
    """

    count = 0
    best_x = x.copy()
    best_y = y.copy()
    best_error = error
    best_factors = deepcopy(factors)
    gain_ratio = 1
    delta = 1
    
    while gain_ratio > 0:
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_y = y
        old_error = error
        damp, inner_method, cg_maxiter, cg_factor, cg_tol, tol_jump, symm, factors_norm, fix_mode = inner_parameters
        
        # Apply dog leg method.
        y = dogleg(y, grad, JT_J_grad, delta)
        
        # Update results.
        x = x + y

        # Balance and transform factors.
        factors = cnv.x2cpd(x, factors, eq=False)
        factors = cnv.transform(factors, symm, factors_norm)

        # Compute error.
        T1_approx = cnv.cpd2unfold1(T1_approx, factors)
        error = crt.fastnorm(Tl[0], T1_approx) / Tsize

        # Update gain ratio.
        gain_ratio = update_gain_ratio(damp, old_error, error, Tsize, old_x, x, grad)
       
        if error < old_error:
            best_x = x.copy()
            best_y = y.copy()
            best_error = error
            best_factors = deepcopy(factors)

        # Update delta.
        delta = update_delta(delta, gain_ratio, norm(x - old_x))
                
        count += 1
        if count > 10:
            break
        
    return best_factors, best_x, best_y, best_error


def dogleg(y, grad, JT_J_grad, delta):
    """
    Subroutine for function compute_dogleg_steps.
    """

    # Initialize constants.
    alpha = norm(grad)**2 / dot(grad, JT_J_grad)
    y_sd = alpha * grad
    c1 = norm(y_sd)
    c2 = dot(y - y_sd, y_sd)
    c3 = norm(y - y_sd)**2
    c4 = delta**2 - c1**2
    m = c2**2 + c3 * c4
    
    # Decide next step.
    if norm(y) < delta:
        y_dl = y
    elif c1 >= delta:
        y_dl = (delta/c1) * y_sd
    elif m < 0:
        y_dl = y
    else:   
        c5 = sqrt(m)
        if c2 <= 0:
            beta = (-c2 + c5)/c3
        else:
            beta = c4/(c2 + c5)
        y_dl = y_sd + beta*(y - y_sd)
    
    return y_dl
