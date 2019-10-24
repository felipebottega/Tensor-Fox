"""
 Gauss-Newton Module
 ===================
 This module implement the damped Gauss-Newton algorithm, with iterations performed with aid of the conjugate gradient 
method.
"""

# Python modules
import numpy as np
from numpy import inf, mean, copy, concatenate, empty, array, zeros, ones, float64, sqrt, dot, linspace, identity, nan, add, subtract
from numpy.linalg import norm
from numpy.random import randint
from scipy.linalg import solve
import sys
import warnings
from numba import njit

# Tensor Fox modules
import Alternating_Least_Squares as als
import Conversion as cnv
import MultilinearAlgebra as mlinalg


def dGN(T, X, Y, Z, R, init_error, options):
    """
    This function uses the Damped Gauss-Newton method to compute an approximation of T with rank r. An initial point to 
    start the iterations must be given. This point is described by the arrays X, Y, Z.    
    The Damped Gauss-Newton method is a iterative method, updating a point x at each iteration. The last computed x is 
    gives an approximate CPD in flat form, and from this we have the components to form the actual CPD. This program
    also gives some additional information such as the size of the steps (distance between each x computed), the
    absolute errors between the approximate and target tensor, and the path of solutions (the points x computed at each
    iteration are saved).

    Inputs
    ------
    T: float 3-D ndarray
    X: float 2-D ndarray of shape (m, R)
    Y: float 2-D ndarray of shape (n, R)
    Z: float 2-D ndarray of shape (p, R)
    R: int.
        The desired rank of the approximating tensor.
    maxiter: int
        Number of maximum iterations permitted. 
    tol: float
        Tolerance criterion to stop the iteration process. This value is used in more than one stopping criteria.
    symm: bool
    display: int
    
    Outputs
    -------
    X, Y, Z: 2-D ndarray
        The factor matrices of the CPD of T.
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration. 
    improv: float 1-D ndarray
        Improvement of the error at each iteration. More precisely, the difference between the relative error of the
        current iteration and the previous one.
    gradients: float 1-D ndarray
        Gradient of the error function at each iteration.
    stop: 0, 1, 2, 3, 4, 5 or 6
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
        7: no refinement was performed (this is not really a stopping condition, but it is necessary to indicate when
        the program can't give a stopping condition in the refinement stage).
    """

    # INITIALIZE RELEVANT VARIABLES 

    # Extract all variable from the class of options.
    init_damp = options.init_damp
    maxiter = options.maxiter
    tol = options.tol
    tol_step = options.tol_step
    tol_improv = options.tol_improv
    tol_grad = options.tol_grad
    symm = options.symm
    display = options.display
    low, upp, factor = options.constraints
    factors_norm = options.factors_norm
    inner_method, cg_maxiter, cg_factor, cg_tol = [options.inner_method, options.cg_maxiter, options.cg_factor, options.cg_tol]

    # Verify if some factor should be fixed or not. This only happens in the bi function.
    X_orig = []
    Y_orig = []
    Z_orig = []
    fix_mode = -1
    if type(X) == list:
        fix_mode = 0
        X_orig = copy(X[0])
        X = X[0]
    elif type(Y) == list:
        fix_mode = 1
        Y_orig = copy(Y[0])
        Y = Y[0]
    elif type(Z) == list:
        fix_mode = 2
        Z_orig = copy(Z[0])
        Z = Z[0]

    # Set the other variables.
    m, n, p = T.shape
    Tsize = norm(T)
    error = 1
    best_error = init_error
    stop = 5
    if type(init_damp) == list:
        damp = init_damp[0] 
    else:   
        damp = init_damp * mean(np.abs(T))
    const = 1 + int(maxiter / 10)

    # INITIALIZE RELEVANT ARRAYS

    x = concatenate((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
    y = zeros(R * (m + n + p), dtype=float64)
    step_sizes = empty(maxiter)
    errors = empty(maxiter)
    improv = empty(maxiter)
    gradients = empty(maxiter)
    best_X = copy(X)
    best_Y = copy(Y)
    best_Z = copy(Z)

    # Prepare data to use in each Gauss-Newton iteration.
    data = prepare_data(m, n, p, R)

    # Compute unfoldings.
    T1 = cnv.unfold(T, 1)
    T2 = cnv.unfold(T, 2)
    T3 = cnv.unfold(T, 3)
    T1_approx = empty(T1.shape, dtype=float64)

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
        # A = Dres(x) and b = -res(x).
        inner_parameters = damp, inner_method, cg_maxiter, cg_factor, cg_tol, low, upp, factor, symm, factors_norm, fix_mode
        T1_approx, X, Y, Z, x, y, grad, itn, residualnorm, error = compute_step(T, Tsize, T1, T2, T3, T1_approx, X, Y, Z,
                                                                               X_orig, Y_orig, Z_orig, data, x, y,
                                                                               inner_parameters, it)

        # Update best solution.
        if error < best_error:
            best_error = error
            best_X = copy(X)
            best_Y = copy(Y)
            best_Z = copy(Z)

        # Update damp. 
        damp = update_damp(damp, init_damp, old_error, error, residualnorm, it)

        # Save relevant information about the current iteration.
        errors[it] = error
        step_sizes[it] = norm(x - old_x)
        gradients[it] = norm(grad, inf)
        if it == 0:
            improv[it] = errors[it]
        else:
            improv[it] = np.abs(errors[it - 1] - errors[it])

        # Show information about current iteration.
        if display > 1:
            if display == 4:
                print('    ',
                      '{:^8}'.format(it + 1),
                      '| {:^10.5e}'.format(errors[it]),
                      '| {:^10.5e}'.format(step_sizes[it]),
                      '| {:^10.5e}'.format(improv[it]),
                      '| {:^11.5e}'.format(gradients[it]),
                      '| {:^15.5e}'.format(residualnorm),
                      '| {:^16}'.format(itn))
            else:
                print('   ',
                      '{:^9}'.format(it + 1),
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
            # Let const=1+int(maxiter/10). Comparing the average errors of const consecutive iterations prevents the
            # program to continue iterating when the error starts to oscillate.
            if it > 2*const and it % const == 0:
                mean1 = mean(errors[it - 2*const: it - const])
                mean2 = mean(errors[it-const: it])
                if mean1 - mean2 <= tol_improv:
                    stop = 4
                    break
            # Prevent blow ups.
            if error > max(1, Tsize ** 2) / tol:
                stop = 6
                break

                # SAVE LAST COMPUTED INFORMATION

    errors = errors[0:it + 1]
    step_sizes = step_sizes[0:it + 1]
    improv = improv[0:it + 1]
    gradients = gradients[0:it + 1]

    return best_X, best_Y, best_Z, step_sizes, errors, improv, gradients, stop


def compute_step(T, Tsize, T1, T2, T3, T1_approx, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters, it):
    """    
    This function uses the chosen inner method to compute the next step.
    """

    # Initialize first variables.
    R = X.shape[1]
    m, n, p = X.shape[0], Y.shape[0], Z.shape[0]
    damp, inner_method, cg_maxiter, cg_factor, cg_tol, low, upp, factor, symm, factors_norm, fix_mode = inner_parameters
    if type(inner_method) == list:
        inner_method = inner_method[it]

    # Call the inner method.
    if inner_method == 'cg' or inner_method == 'cg_static':
        if inner_method == 'cg':
            cg_maxiter = 1 + int(cg_factor * randint(1 + it ** 0.4, 2 + it ** 0.9))
        y, grad, itn, residualnorm = cg(T1, T2, T3, X, Y, Z, data, y, m, n, p, R, damp, cg_maxiter, cg_tol)

    elif inner_method == 'direct': 
        y, grad, itn, residualnorm = direct_solve(T, Tsize, T1_approx, T1, T2, T3, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters)

    elif inner_method == 'gd':
        y, grad, itn, residualnorm = gradient_descent(T, Tsize, T1_approx, T1, T2, T3, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters)

    elif inner_method == 'als':
        X, Y, Z = als.als_iteration(T1, T2, T3, X, Y, Z, fix_mode)
        x = concatenate((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
        y *= 0

    else:
        sys.exit("Wrong inner method name. Must be 'cg', 'cg_static', 'direct' or 'gd'.")

    # Update results.
    x = x + y

    # Compute new factors X, Y, Z.
    X, Y, Z = cnv.x2cpd(x, X, Y, Z)
    X, Y, Z = cnv.transform(X, Y, Z, low, upp, factor, symm, factors_norm)
    if fix_mode == 0:
        X = copy(X_orig)
    elif fix_mode == 1:
        Y = copy(Y_orig)
    elif fix_mode == 2:
        Z = copy(Z_orig)

    # Compute error.
    T1_approx = cnv.cpd2unfold1(T1_approx, [X, Y, Z])
    error = norm(T1 - T1_approx) / Tsize

    if inner_method == 'als':
        return T1_approx, X, Y, Z, x, y, [nan], '-', Tsize*error, error
    
    return T1_approx, X, Y, Z, x, y, grad, itn, residualnorm, error


def direct_solve(T, Tsize, T1_approx, T1, T2, T3, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters):
    # Give names to the arrays.
    Gr_X, Gr_Y, Gr_Z,\
        Gr_XY, Gr_XZ, Gr_YZ,\
        V_Xt, V_Yt, V_Zt,\
        V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,\
        Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,\
        Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,\
        Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,\
        GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,\
        BX1, BY1, BZ1,\
        BX2, BY2, BZ2,\
        BXv, BYv, BZv,\
        BX_plus, BY_plus, BZ_plus, Bv,\
        X_norms, Y_norms, Z_norms,\
        gamma_X, gamma_Y, gamma_Z, Gamma,\
        M, L, residual_cg, P, Q, z,\
        NX, NY, NZ, CX, DX, gX, CY, DY, gY, CZ, DZ, gZ, g = data

    damp, inner_method, cg_maxiter, cg_factor, cg_tol, low, upp, factor, symm, factors_norm, fix_mode = inner_parameters

    # Compute the values of all arrays.
    R = X.shape[1]
    m, n, p = T.shape
    Jf = jacobian(X, Y, Z, m, n, p, R) 
    H = hessian(Jf)
    L, X_norms, Y_norms, Z_norms = regularization(X, Y, Z, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, m, n, p, R)
    grad = -compute_grad(T1, T2, T3, X, Y, Z, NX, NY, NZ, AX, BX, CX, DX, Gr_YZ, gX, AY, BY, CY, DY, Gr_XZ, gY, AZ, BZ, CZ, DZ, Gr_XY, gZ, g)
    
    # Add regularization.
    for i in range(R*(m+n+p)):
        H[i, i] += damp*L[i]

    # Solve system.
    warnings.filterwarnings("ignore")
    try:
        y = solve(H, grad, sym_pos=True, check_finite=False)
    except np.linalg.LinAlgError:
        y, grad, itn, residualnorm = gradient_descent(T, Tsize, T1_approx, T1, T2, T3, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters)

    residualnorm = norm(dot(H, y) - grad)

    return y, grad, '-', residualnorm


def gradient_descent(T, Tsize, T1_approx, T1, T2, T3, X, Y, Z, X_orig, Y_orig, Z_orig, data, x, y, inner_parameters):
    # Give names to the arrays.
    Gr_X, Gr_Y, Gr_Z,\
        Gr_XY, Gr_XZ, Gr_YZ,\
        V_Xt, V_Yt, V_Zt,\
        V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,\
        Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,\
        Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,\
        Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,\
        GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,\
        BX1, BY1, BZ1,\
        BX2, BY2, BZ2,\
        BXv, BYv, BZv,\
        BX_plus, BY_plus, BZ_plus, Bv,\
        X_norms, Y_norms, Z_norms,\
        gamma_X, gamma_Y, gamma_Z, Gamma,\
        M, L, residual_cg, P, Q, z,\
        NX, NY, NZ, CX, DX, gX, CY, DY, gY, CZ, DZ, gZ, g = data

    damp, inner_method, cg_maxiter, cg_factor, cg_tol, low, upp, factor, symm, factors_norm, fix_mode = inner_parameters

    grad = compute_grad(T1, T2, T3, X, Y, Z, NX, NY, NZ, AX, BX, CX, DX, Gr_YZ, gX, AY, BY, CY, DY, Gr_XZ, gY, AZ, BZ, CZ, DZ, Gr_XY, gZ, g)

    # Test some values of alpha and keep the best.
    best_error = inf
    m, n, p = T.shape
    R = X.shape[1]
    alphas = 2**linspace(-32, 1, 16)

    for alpha in alphas:
        # Update x.
        temp_x = x - alpha*grad

        # Compute new factors X, Y, Z.
        temp_X, temp_Y, temp_Z = cnv.x2cpd(temp_x, X, Y, Z)
        X, Y, Z = cnv.transform(temp_X, temp_Y, temp_Z, low, upp, factor, symm, factors_norm)
        if fix_mode == 0:
            temp_X = copy(X_orig)
        elif fix_mode == 1:
            temp_Y = copy(Y_orig)
        elif fix_mode == 2:
            temp_Z = copy(Z_orig)

        # Compute error.
        T1_approx = cnv.cpd2unfold1(T1_approx, [temp_X, temp_Y, temp_Z])
        error = norm(T1 - T1_approx) / Tsize
        
        # Update best results.
        if error < best_error:
            best_error = error
            y = - alpha*grad

    residualnorm = Tsize*best_error

    return y, -grad, '-', residualnorm


def cg(T1, T2, T3, X, Y, Z, data, y, m, n, p, R, damp, maxiter, tol):
    """
    Conjugate gradient algorithm specialized to the tensor case.
    """

    maxiter = min(maxiter, R * (m + n + p))

    # Give names to the arrays.
    Gr_X, Gr_Y, Gr_Z,\
        Gr_XY, Gr_XZ, Gr_YZ,\
        V_Xt, V_Yt, V_Zt,\
        V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,\
        Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,\
        Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,\
        Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,\
        GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,\
        BX1, BY1, BZ1,\
        BX2, BY2, BZ2,\
        BXv, BYv, BZv,\
        BX_plus, BY_plus, BZ_plus, Bv,\
        X_norms, Y_norms, Z_norms,\
        gamma_X, gamma_Y, gamma_Z, Gamma,\
        M, L, residual_cg, P, Q, z,\
        NX, NY, NZ, CX, DX, gX, CY, DY, gY, CZ, DZ, gZ, g = data

    # Compute the values of all arrays.
    Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ = gramians(X, Y, Z,
                                                     Gr_X, Gr_Y, Gr_Z,
                                                     Gr_XY, Gr_XZ, Gr_YZ)
    L, X_norms, Y_norms, Z_norms = regularization(X, Y, Z,
                       X_norms, Y_norms, Z_norms,
                       gamma_X, gamma_Y, gamma_Z, Gamma,
                       m, n, p, R)
    M = precond(X_norms, Y_norms, Z_norms, L, M, damp, m, n, p, R)

    y *= 0

    # CG iterations.
    grad = -compute_grad(T1, T2, T3, X, Y, Z, NX, NY, NZ, Gr_X, Gr_Y, Gr_Z, CX, DX, Gr_YZ, gX, CY, DY, Gr_XZ, gY, CZ, DZ, Gr_XY, gZ, g)
    residual_cg = M * grad
    P = residual_cg
    residualnorm = dot(residual_cg.T, residual_cg)
    if residualnorm == 0.0:
        residualnorm = 1e-6
    
    for itn in range(maxiter):
        Q = M * P
        
        z = matvec(X, Y, Z,
                   Gr_X, Gr_Y, Gr_Z,
                   Gr_XY, Gr_XZ, Gr_YZ,
                   V_Xt, V_Yt, V_Zt,
                   V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,
                   Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,
                   Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,
                   Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,
                   GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,
                   BX1, BY1, BZ1,
                   BX2, BY2, BZ2,
                   BXv, BYv, BZv,
                   BX_plus, BY_plus, BZ_plus, Bv,
                   Q, m, n, p, R) + damp * L * Q
        z = M * z
        denominator = dot(P.T, z)
        if denominator == 0.0:
            denominator = 1e-6
        alpha = residualnorm / denominator
        y += alpha * P
        residual_cg -= alpha * z
        residualnorm_new = dot(residual_cg.T, residual_cg)
        beta = residualnorm_new / residualnorm
        residualnorm = residualnorm_new
        P = residual_cg + beta * P
        
        # Stopping criteria.
        if residualnorm < tol:
            break

    return M * y, grad, itn + 1, residualnorm


def compute_grad(T1, T2, T3, X, Y, Z, NX, NY, NZ, Gr_X, Gr_Y, Gr_Z, CX, DX, Gr_YZ, gX, CY, DY, Gr_XZ, gY, CZ, DZ, Gr_XY, gZ, g):
    """
    This function computes the gradient of the error function at (X, Y, Z).
    """

    # Initialize first variables.
    m, n, p = X.shape[0], Y.shape[0], Z.shape[0]
    R = X.shape[1]

    # X part.
    NX = mlinalg.khatri_rao(Z, Y, NX)
    dot(X, Gr_YZ, out=CX)
    dot(T1, NX, out=DX)
    subtract(CX, DX, gX)
    ggX = gX.T.reshape(m*R,)

    # Y part.
    NY = mlinalg.khatri_rao(Z, X, NY)
    dot(Y, Gr_XZ, out=CY)
    dot(T2, NY, out=DY)
    subtract(CY, DY, gY)
    ggY = gY.T.reshape(n*R,)

    # Z part.
    NZ = mlinalg.khatri_rao(Y, X, NZ)
    dot(Z, Gr_XY, out=CZ)
    dot(T3, NZ, out=DZ)
    subtract(CZ, DZ, gZ)
    ggZ = gZ.T.reshape(p*R,)

    concatenate((ggX, ggY, ggZ), out=g)

    return g


def prepare_data(m, n, p, R):
    """
    Initialize all necessary matrices to keep the values of several computations during the program.
    """

    # Gramians
    Gr_X = empty((R, R), dtype=float64)
    Gr_Y = empty((R, R), dtype=float64)
    Gr_Z = empty((R, R), dtype=float64)
    Gr_XY = empty((R, R), dtype=float64)
    Gr_XZ = empty((R, R), dtype=float64)
    Gr_YZ = empty((R, R), dtype=float64)

    # V_X^T, V_Y^T, V_Z^T
    V_Xt = empty((R, m), dtype=float64)
    V_Yt = empty((R, n), dtype=float64)
    V_Zt = empty((R, p), dtype=float64)

    # Initializations of matrices to receive the results of the computations.
    V_Xt_dot_X = empty((R, R), dtype=float64)
    V_Yt_dot_Y = empty((R, R), dtype=float64)
    V_Zt_dot_Z = empty((R, R), dtype=float64)
    Gr_Z_V_Yt_dot_Y = empty((R, R), dtype=float64)
    Gr_Y_V_Zt_dot_Z = empty((R, R), dtype=float64)
    Gr_X_V_Zt_dot_Z = empty((R, R), dtype=float64)
    Gr_Z_V_Xt_dot_X = empty((R, R), dtype=float64)
    Gr_Y_V_Xt_dot_X = empty((R, R), dtype=float64)
    Gr_X_V_Yt_dot_Y = empty((R, R), dtype=float64)
    GZY_plus_GYZ = empty((R, R), dtype=float64)
    GXZ_plus_GZX = empty((R, R), dtype=float64)
    GYX_plus_GXY = empty((R, R), dtype=float64)

    # Final blocks
    BX1 = empty((R, m), dtype=float64)
    BY1 = empty((R, n), dtype=float64)
    BZ1 = empty((R, p), dtype=float64)
    BX2 = empty((m, R), dtype=float64)
    BY2 = empty((n, R), dtype=float64)
    BZ2 = empty((p, R), dtype=float64)
    BX_plus = empty((m, R), dtype=float64)
    BY_plus = empty((n, R), dtype=float64)
    BZ_plus = empty((p, R), dtype=float64)
    BXv = empty(m * R, dtype=float64)
    BYv = empty(n * R, dtype=float64)
    BZv = empty(p * R, dtype=float64)
    Bv = empty(R*(m+n+p), dtype=float64)
    
    # Matrices for the diagonal block
    Gr_YZ_V_Xt = empty((R, m), dtype=float64)
    Gr_XZ_V_Yt = empty((R, n), dtype=float64)
    Gr_XY_V_Zt = empty((R, p), dtype=float64)

    # Matrices to use when constructing the Tikhonov matrix for regularization.
    X_norms = empty(R, dtype=float64)
    Y_norms = empty(R, dtype=float64)
    Z_norms = empty(R, dtype=float64)
    gamma_X = empty(R, dtype=float64)
    gamma_Y = empty(R, dtype=float64)
    gamma_Z = empty(R, dtype=float64)
    Gamma = empty(R * (m + n + p), dtype=float64)

    # Arrays to be used in the Conjugated Gradient.
    M = ones(R * (m + n + p), dtype=float64)
    L = ones(R * (m + n + p), dtype=float64)
    residual_cg = empty(R * (m + n + p), dtype=float64)
    P = empty(R * (m + n + p), dtype=float64)
    Q = empty(R * (m + n + p), dtype=float64)
    z = empty(R * (m + n + p), dtype=float64)

    # Arrays to be used in the compute_grad function.
    NX = empty((n*p, R), dtype=float64)
    NY = empty((m*p, R), dtype=float64)
    NZ = empty((m*n, R), dtype=float64)
    CX = empty((m, R), dtype=float64)
    DX = empty((m, R), dtype=float64)
    gX = empty((m, R), dtype=float64)
    CY = empty((n, R), dtype=float64)
    DY = empty((n, R), dtype=float64)
    gY = empty((n, R), dtype=float64)
    CZ = empty((p, R), dtype=float64)
    DZ = empty((p, R), dtype=float64)
    gZ = empty((p, R), dtype=float64)
    g = empty(R*(m+n+p), dtype=float64)

    data = [Gr_X, Gr_Y, Gr_Z,
            Gr_XY, Gr_XZ, Gr_YZ,
            V_Xt, V_Yt, V_Zt,
            V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,
            Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,
            Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,
            Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,
            GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,
            BX1, BY1, BZ1,
            BX2, BY2, BZ2,
            BXv, BYv, BZv,
            BX_plus, BY_plus, BZ_plus, Bv,
            X_norms, Y_norms, Z_norms,
            gamma_X, gamma_Y, gamma_Z, Gamma,
            M, L, residual_cg, P, Q, z,
            NX, NY, NZ, CX, DX, gX, CY, DY, gY, CZ, DZ, gZ, g]

    return data


def update_damp(damp, init_damp, old_error, error, residualnorm, it):
    """
    Update rule of the damping parameter for the dGN function.
    """

    if type(init_damp) == list:
        damp = init_damp[it] 
    else:   
        if old_error != residualnorm:
            gain_ratio = 2 * (old_error - error) / (old_error - residualnorm)
        else:
            gain_ratio = 1.0
        if gain_ratio < 0.75:
            damp = damp / 2
        elif gain_ratio > 0.9:
            damp = 1.5 * damp

    return damp


def gramians(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ):
    """ 
    Computes all Gramians matrices of X, Y, Z. Also it computes all Hadamard products between the different Gramians. 
    """

    dot(X.T, X, out=Gr_X)
    dot(Y.T, Y, out=Gr_Y)
    dot(Z.T, Z, out=Gr_Z)
    Gr_XY = mlinalg.hadamard(Gr_X, Gr_Y, Gr_XY)
    Gr_XZ = mlinalg.hadamard(Gr_X, Gr_Z, Gr_XZ)
    Gr_YZ = mlinalg.hadamard(Gr_Y, Gr_Z, Gr_YZ)

    return Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ


def matvec(X, Y, Z,
           Gr_X, Gr_Y, Gr_Z,
           Gr_XY, Gr_XZ, Gr_YZ,
           V_Xt, V_Yt, V_Zt,
           V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z,
           Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z,
           Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y,
           Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt,
           GZY_plus_GYZ, GXZ_plus_GZX, GYX_plus_GXY,
           BX1, BY1, BZ1,
           BX2, BY2, BZ2,
           BXv, BYv, BZv,
           BX_plus, BY_plus, BZ_plus, Bv,
           v, m, n, p, R):
    """
    Makes the matrix-vector computation (Df^T * Df)*v.
    """

    # Split v into three blocks, convert them into matrices and transpose them. 
    # With this we have the matrices V_X^T, V_Y^T, V_Z^T.
    V_Xt = v[0: m * R].reshape(R, m)
    V_Yt = v[m * R: R * (m + n)].reshape(R, n)
    V_Zt = v[R * (m + n): R * (m + n + p)].reshape(R, p)

    # Compute the products V_X^T*X, V_Y^T*Y, V_Z^T*Z
    dot(V_Xt, X, out=V_Xt_dot_X)
    dot(V_Yt, Y, out=V_Yt_dot_Y)
    dot(V_Zt, Z, out=V_Zt_dot_Z)

    # Compute the Hadamard products
    Gr_Z_V_Yt_dot_Y = mlinalg.hadamard(Gr_Z, V_Yt_dot_Y, Gr_Z_V_Yt_dot_Y)
    Gr_Y_V_Zt_dot_Z = mlinalg.hadamard(Gr_Y, V_Zt_dot_Z, Gr_Y_V_Zt_dot_Z)
    Gr_X_V_Zt_dot_Z = mlinalg.hadamard(Gr_X, V_Zt_dot_Z, Gr_X_V_Zt_dot_Z)
    Gr_Z_V_Xt_dot_X = mlinalg.hadamard(Gr_Z, V_Xt_dot_X, Gr_Z_V_Xt_dot_X)
    Gr_Y_V_Xt_dot_X = mlinalg.hadamard(Gr_Y, V_Xt_dot_X, Gr_Y_V_Xt_dot_X)
    Gr_X_V_Yt_dot_Y = mlinalg.hadamard(Gr_X, V_Yt_dot_Y, Gr_X_V_Yt_dot_Y)
    
    # Add intermediate blocks
    add(Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, out=GZY_plus_GYZ)
    add(Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, out=GXZ_plus_GZX)
    add(Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, out=GYX_plus_GXY)

    # Compute final products
    dot(X, GZY_plus_GYZ, out=BX2)
    dot(Y, GXZ_plus_GZX, out=BY2)
    dot(Z, GYX_plus_GXY, out=BZ2)
    
    # Diagonal block matrices
    dot(Gr_YZ, V_Xt, out=BX1)
    dot(Gr_XZ, V_Yt, out=BY1)
    dot(Gr_XY, V_Zt, out=BZ1)

    # Vectorize the matrices to have the final vectors
    add(BX1.T, BX2, out=BX_plus)
    add(BY1.T, BY2, out=BY_plus)
    add(BZ1.T, BZ2, out=BZ_plus)
    BXv = cnv.vec(BX_plus, BXv, m, R)
    BYv = cnv.vec(BY_plus, BYv, n, R)
    BZv = cnv.vec(BZ_plus, BZv, p, R)
    concatenate((BXv, BYv, BZv), out=Bv)

    return Bv


@njit(nogil=True)
def regularization(X, Y, Z, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, m, n, p, R):
    """
    Computes the Tikhonov matrix Gamma, where Gamma is a diagonal matrix designed specifically to make 
    Jf^T * Jf + Gamma diagonally dominant.
    """

    for r in range(R):
        X_norms[r] = norm(X[:, r])
        Y_norms[r] = norm(Y[:, r])
        Z_norms[r] = norm(Z[:, r])

    max_XY = np.max(X_norms * Y_norms)
    max_XZ = np.max(X_norms * Z_norms)
    max_YZ = np.max(Y_norms * Z_norms)
    max_all = max(max_XY, max_XZ, max_YZ)

    for r in range(R):
        gamma_X[r] = Y_norms[r] * Z_norms[r] * max_all
        gamma_Y[r] = X_norms[r] * Z_norms[r] * max_all
        gamma_Z[r] = X_norms[r] * Y_norms[r] * max_all

    for r in range(R):
        Gamma[r * m:(r + 1) * m] = gamma_X[r]
        Gamma[m * R + r * n:m * R + (r + 1) * n] = gamma_Y[r]
        Gamma[R * (m + n) + r * p:R * (m + n) + (r + 1) * p] = gamma_Z[r]

    return Gamma, X_norms, Y_norms, Z_norms


@njit(nogil=True)
def precond(X_norms, Y_norms, Z_norms, L, M, damp, m, n, p, R):
    """
    This function constructs a preconditioner in order to accelerate the Conjugate Gradient function. It is a diagonal
    preconditioner designed to make Dres.transpose*Dres + Gamma a unit diagonal matrix. Since the matrix is diagonally 
    dominant, the result will be close to the identity matrix. Therefore, it will have its eigenvalues clustered
    together.
    """
    for r in range(R):
        M[r * m:(r + 1) * m] = Y_norms[r]**2 * Z_norms[r]**2 + damp * L[r * m: (r + 1) * m]
        M[m * R + r * n:m * R + (r + 1) * n] = X_norms[r]**2 * Z_norms[r]**2 + damp * L[m * R + r * n: m * R + (r + 1) * n]
        M[R * (m + n) + r * p:R * (m + n) + (r + 1) * p] = X_norms[r]**2 * Y_norms[r]**2 + damp * L[R * (m + n) + r * p: R * (m + n) + (r + 1) * p]

    M = 1 / sqrt(M)
    return M


@njit(nogil=True)
def jacobian(X, Y, Z, m, n, p, r):
    """
    This function computes the Jacobian matrix Jf of the residual function. This is a dense mnp x r(m+n+p) matrix.
    """

    Jf = zeros((m*n*p, r*(m+n+p)))
    s = 0
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(r):
                    Jf[s, l*m + i] = -Y[j, l]*Z[k, l]
                    Jf[s, r*m + l*n + j] = -X[i, l]*Z[k, l]
                    Jf[s, r*(m+n) + l*p + k] = -X[i, l]*Y[j, l]
                s += 1
                        
    return Jf


def hessian(Jf):
    """
    Approximate Hessian matrix of the error function.
    """

    H = dot(Jf.T, Jf)
    return H
