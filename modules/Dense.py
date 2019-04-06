"""
Dense module

 - jacobian
 
 - Hessian

 - precond

 - eig_dist

 - plot_structures

 - dense_cpd

 - dense_dGN

 - dense_output_info

"""


import numpy as np
import sys
import scipy.io
from decimal import Decimal
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import TensorFox as tfx
import Construction as cnst
import Conversion as cnv
import Auxiliar as aux
import Display as disp
import Critical as crt


@njit(nogil=True)
def jacobian(X, Y, Z, m, n, p, r):
    """
    This function computes the Jacobian matrix Jf of the residual function. We can also
    write Jf = Dres with the notations used in the other modules. This is a dense 
    mnp x r(m+n+p) matrix.
    """

    Jf = np.zeros((m*n*p, r*(m+n+p)))
    s = 0
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(r):
                    Jf[s, l*m + i] = -Y[j,l]*Z[k,l]
                    Jf[s, r*m + l*n + j] = -X[i,l]*Z[k,l]
                    Jf[s, r*(m+n) + l*p + k] = -X[i,l]*Y[j,l]
                s += 1
                        
    return Jf


def Hessian(Jf):
    """
    Approximate Hessian matrix of the error function.
    """

    H = np.dot(Jf.T, Jf)
    return H


def precond(H, damp, m, n, p, r):
    """
    This function computes the preconditioner with respect to the regularized problem
    (A^T*A + mu*I)*x = A^T*b, where mu is the damping factor, A = Jf and b = -res. 
    Let M be this preconditioner, where M is diagonal with M(i,i) = A(i,i) + mu. 
    This matrix is introduced in the equation as 
                   M^(-1/2)*(A + mu*I)*M^(-1/2)*x = M^(-1/2)*A^T*b.
    
    """

    M = np.zeros((r*(m+n+p), r*(m+n+p)))
    
    for i in range(r*(m+n+p)):
        M[i,i] = 1/np.sqrt(H[i,i] + damp)
                    
    return M


def eig_dist(X, Y, Z, damp, m, n, p, r):
    """
    We want to use the conjugate gradient to solve the equation
                   M^(-1/2)*(A + mu*I)*M^(-1/2)*x = M^(-1/2)*A^T*b.
    In order to do this the spectrum of M^(-1/2)*(A + mu*I)*M^(-1/2) must have
    its eigenvalues clustered together in a few groups (only one group is best).
    This function shows the histogram of the equation without and with 
    preconditiong so we can compare.    
    """

    Jf = jacobian(X, Y, Z, m, n, p, r)
    H = Hessian(Jf)
    M = precond(H, damp, m, n, p, r)
    I = np.identity(r*(m+n+p))

    S, V = np.linalg.eigh(H + damp*I)
    plt.hist(S, bins=100)
    plt.title('Histogram of eigenvalues of H + damp*I')
    plt.show()
    print()

    S, V = np.linalg.eigh(np.dot(np.dot(M,(H + damp*I)),M))
    plt.title('Histogram of eigenvalues of H + damp*I with preconditioning')
    plt.hist(S, bins=100)
    plt.show()

    return


def plot_strucures(X, Y, Z, damp, m, n, p, r):
    """
    Jf is a sparse matrix with a special structure, and H has some sparse structure
    in its diagonal blocks. This function shows these structures.
    """
 
    Jf = jacobian(X, Y, Z, m, n, p, r)
    H = Hessian(Jf)

    # White is zero, black is nonzero.
    plt.imshow(Jf==0, cmap='gray')
    plt.title('Structure of Jacobian of the residual function')
    plt.show()

    plt.imshow(H==0, cmap='gray')
    plt.title('Structure of Hessian of the residual function')
    plt.show()

    return Jf, H


def dense_cpd(T, r, options=False, plot=False):
    """
    Dense version of cpd. Use it to research purposes. Setting plot to True makes the program to call
    the function eig_dist and plot the eigenvalue distribution of H + damp*I with and without preconditioning
    at each step of the dGN.
    """ 

    # Set options
    maxiter, tol, maxiter_refine, tol_refine, init, trunc_dims, level, refine, symm, low, upp, factor, display = aux.make_options(options)
        
    # Compute relevant variables and arrays.
    m_orig, n_orig, p_orig = T.shape
    m, n, p = m_orig, n_orig, p_orig
    T_orig = np.copy(T)
    Tsize = np.linalg.norm(T)
    tol = np.min([tol*m*n*p, 1e-1])
    tol_refine = np.min([tol_refine*m*n*p, 1e-1])
                   
    # Test consistency of dimensions and rank.
    aux.consistency(r, m, n, p, symm) 

    # Change ordering of indexes to improve performance if possible.
    T, ordering = aux.sort_dims(T, m, n, p)
    m, n, p = T.shape
    
    # COMPRESSION STAGE
    
    if display != 0:
        print('--------------------------------------------------------------------------------------------------------------') 
        print('Computing MLSVD of T')
    
    # Compute compressed version of T with the MLSVD. We have that T = (U1,U2,U3)*S.
    if display == 3:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, best_error = tfx.mlsvd(T, Tsize, r, trunc_dims, level, display)
    else:
        S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop = tfx.mlsvd(T, Tsize, r, trunc_dims, level, display)

    # When the tensor is symmetric we want S to have equal dimensions. 
    if symm:
        R = min(R1, R2, R3)
        R1, R2, R3 = R, R, R
        S = S[:R, :R, :R]
        U1, U2, U3 = U1[:, :R], U2[:, :R], U3[:, :R]
          
    if display != 0:
        if (R1, R2, R3) == (m, n, p):
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
    
    if display != 0:
        print('--------------------------------------------------------------------------------------------------------------')        
        if type(init) == list:
            print('Type of initialization: user')
        else:
            print('Type of initialization:', init)
        if display == 3:
            a = float('%.4e' % Decimal(rel_error))
            print('    Initial guess relative error =', a)   
    
    # DAMPED GAUSS-NEWTON STAGE 
    
    if display != 0:
        print('--------------------------------------------------------------------------------------------------------------') 
        print('Computing CPD of T')
   
    # Compute the approximated tensor in coordinates with the dGN method.
    X, Y, Z, step_sizes_main, errors_main, gradients_main, conds_main, stop_main = dense_dGN(S, X, Y, Z, r, maxiter, tol, symm, low, upp, factor, display, plot) 
    
    # REFINEMENT STAGE
    
    if refine:
        if display != 0:
            print('--------------------------------------------------------------------------------------------------------------') 
            print('Computing refinement of solution') 
        X, Y, Z, step_sizes_refine, errors_refine, gradients_refine, conds_refine, stop_refine = dense_dGN(S, X, Y, Z, r, maxiter_refine, tol_refine, symm, low, upp, factor, display, plot)
    else:
        step_sizes_refine = np.array([0])
        errors_refine = np.array([0]) 
        gradients_refine = np.array([0]) 
        conds_refine = np.array([0])
        stop_refine = 5 
    
    # FINAL WORKS

    # Go back to the original dimensions of T.
    X, Y, Z, U1_sort, U2_sort, U3_sort = aux.unsort_dims(X, Y, Z, U1, U2, U3, ordering)
       
    # Use the orthogonal transformations to obtain the CPD of T.
    X = np.dot(U1_sort,X)
    Y = np.dot(U2_sort,Y)
    Z = np.dot(U3_sort,Z)
    
    # Compute coordinate representation of the CPD of T.
    T_aux = np.zeros((m_orig, n_orig, p_orig), dtype = np.float64)
    temp = np.zeros((m_orig, r), dtype = np.float64, order='F')
    T_approx = cnv.cpd2tens(T_aux, X, Y, Z, temp, m_orig, n_orig, p_orig, r)
        
    # Normalize X, Y, Z to have column norm equal to 1.
    Lambda, X, Y, Z = aux.normalize(X, Y, Z, r)
    
    # Save and display final informations.
    output = dense_output_info(T_orig, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, gradients_main, gradients_refine, conds_main, conds_refine, mlsvd_stop, stop_main, stop_refine)

    if display != 0:
        print('==============================================================================================================')
        print('Final results')
        if refine:
            print('    Number of steps =', output.num_steps)
        else:
            print('    Number of steps =', output.num_steps-1)
        print('    Relative error =', output.rel_error)
        a = float( '%.6e' % Decimal(output.accuracy) )
        print('    Accuracy = ', a, '%')
    
    return Lambda, X, Y, Z, T_approx, output


def dense_dGN(T, X, Y, Z, r, maxiter, tol, symm, low, upp, factor, display, plot):
    """
    Same as the dGN function but this one works with dense matrices and because of that it
    is much slower. At each step this fucntion also computes amd store the condition number 
    of M^(-1/2)*(A + mu*I)*M^(-1/2), this is the main reason for using this function. To 
    study the behavior of the condition number during the computations. 
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
    # constant used in the third stopping condition
    const = 1 + int(maxiter/10)
    stop = 4
                    
    # INITIALIZE RELEVANT ARRAYS
    
    x = np.concatenate((X.flatten('F'), Y.flatten('F'), Z.flatten('F')))
    y = x
    step_sizes = np.zeros(maxiter)
    errors = np.zeros(maxiter)
    gradients = np.zeros(maxiter)
    conds = np.zeros(maxiter)
    # Create auxiliary arrays.
    I = np.identity(r*(m+n+p))
    T_aux = np.zeros((m, n, p), dtype = np.float64)
    temp = np.zeros((m, r), dtype = np.float64, order='F')
    X_aux = np.zeros((m, r), dtype = np.float64)
    Y_aux = np.zeros((n, r), dtype = np.float64)
    Z_aux = np.zeros((p, r), dtype = np.float64)
    X, Y, Z = cnv.x2cpd(x, X_aux, Y_aux, Z_aux, m, n, p, r)
    T_aux = cnv.cpd2tens(T_aux, X, Y, Z, temp, m, n, p, r)
    # res is the array with the residuals (see the residual function for more information).
    res = np.zeros(m*n*p, dtype = np.float64)
    g = np.zeros(r*(m+n+p), dtype = np.float64)
    g1 = np.zeros(r*(m+n+p), dtype = np.float64)
    y = np.zeros(r*(m+n+p), dtype = np.float64)

    # Prepare data to use in each Gauss-Newton iteration.
    data = crt.prepare_data(m, n, p, r)    
    data_rmatvec = crt.prepare_data_rmatvec(m, n, p, r)
        
    if display > 1:
        print('    Iteration | Rel Error  | Rel Error Diff |     ||g||    | Damp| #CG iterations')
    
    # START GAUSS-NEWTON ITERATIONS
    
    for it in range(0,maxiter):      
        # Update of all residuals at x. 
        res = cnst.residual(res, T, T_aux, m, n, p)
                                        
        # Keep the previous value of x and error to compare with the new ones in the next iteration.
        old_x = x
        old_error = error
        
        # cg_maxiter is the maximum number of iterations of the Conjugate Gradient. We obtain it randomly.
        cg_maxiter = np.random.randint(1 + it**0.4, 2 + it**0.9)
                             
        # Computation of the Gauss-Newton iteration formula to obtain the new point x + y, where x is the 
        # previous point and y is the new step obtained as the solution of min_y |Ay - b|, with 
        # A = Dres(x) and b = -res(x).         
        y, g, itn, residualnorm = tfx.cg(X, Y, Z, data, data_rmatvec, y, g, -res, m, n, p, r, damp, cg_maxiter, tol)   
              
        # Update point obtained by the iteration.         
        x = x + y
        
        # Compute X, Y, Z.
        X, Y, Z = cnv.x2cpd(x, X_aux, Y_aux, Z_aux, m, n, p, r)
        X, Y, Z = cnv.transform(X_aux, Y_aux, Z_aux, m, n, p, r, low, upp, factor, symm)
               
        # Compute error.
        T_aux = cnv.cpd2tens(T_aux, X, Y, Z, temp, m, n, p, r)
        error = np.linalg.norm(T - T_aux)

        # Matrix of preconditioned normal equations 
        Jf = jacobian(X, Y, Z, m, n, p, r) 
        H = Hessian(Jf) 
        M = precond(H, damp, m, n, p, r)
        B = np.dot(np.dot(M,(H + damp*I)),M)
        
        # Plot eigenvalue distribution of H + damp*I and B if requested.
        if plot:
            eig_dist(X, Y, Z, damp, m, n, p, r)   

        # Update best solution.
        if error < best_error:
            best_error = error
            best_X, best_Y, best_Z = X, Y, Z
                                                        
        # Update damp. 
        old_damp = damp
        damp = aux.update_damp(damp, old_error, error, residualnorm)
        
        # Save relevant information about the current iteration.
        step_sizes[it] = np.linalg.norm(x - old_x)   
        errors[it] = error
        gradients[it] = np.linalg.norm(g, np.inf)
        conds[it] = np.linalg.cond(B)
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
    conds = conds[0:it+1]
    
    return best_X, best_Y, best_Z, step_sizes, errors, gradients, conds, stop


def dense_output_info(T_orig, Tsize, T_approx, step_sizes_main, step_sizes_refine, errors_main, errors_refine, gradients_main, gradients_refine, conds_main, conds_refine, mlsvd_stop, stop_main, stop_refine):
    class info:
        rel_error = np.linalg.norm(T_orig - T_approx)/Tsize
        step_sizes = [step_sizes_main, step_sizes_refine]
        errors = [errors_main, errors_refine]
        errors_diff = [np.concatenate(([errors_main[0]], np.abs(errors_main[0:-1] - errors_main[1:]))), np.concatenate(([errors_refine[0]], np.abs(errors_refine[0:-1] - errors_refine[1:])))]
        gradients = [gradients_main, gradients_refine]
        conds = [conds_main, conds_refine]
        stop = [mlsvd_stop, stop_main, stop_refine]
        num_steps = np.size(step_sizes_main) + np.size(step_sizes_refine)
        accuracy = max(0, 100*(1 - rel_error))

        def stop_msg(self):
            # mlsvd_stop message
            print('MLSVD stop:')
            if self.stop[0] == 0:
                print('0 - Truncation was given manually by the user.')
            if self.stop[0] == 1:
                print('1 - User choose level = 4 so no truncation was done.')
            if self.stop[0] == 2:
                print('2 - When testing the truncations a big gap between singular values were detected and the program lock the size of the truncation.')
            if self.stop[0] == 3:
                print('3 - The program was unable to compress at the very first iteration. In this case the tensor singular values are equal or almost equal. The program stops the truncation process when this happens.')
            if self.stop[0] == 4:
                print('4 - Tensor probably is random or has a lot of noise.')
            if self.stop[0] == 5:
                print('5 - Overfit was found and the user will have to try again or try a smaller rank.')
            if self.stop[0] == 6:
                print('6 - The energy of the truncation is accepted because it is big enough.')
            if self.stop[0] == 7:
                print('7 - None of the previous conditions were satisfied and we keep the last truncation computed. This condition is only possible at the second stage.')
           
            # stop_main message
            print()
            print('Main stop:')
            if self.stop[1] == 0:
                print('0 - Steps are too small.')
            if self.stop[1] == 1:
                print('1 - The improvement in the relative error is too small.')
            if self.stop[1] == 2:
                print('2 - The gradient is close enough to 0.')
            if self.stop[1] == 3:
                print('3 - The average of the last k relative errors is too small, where k = 1 + int(maxiter/10).')
            if self.stop[1] == 4:
                print('4 - Limit of iterations was been reached.')
            if self.stop[1] == 6:
                print('6 - dGN diverged.')

            # stop_refine message
            print()
            print('Refinement stop:')
            if self.stop[2] == 0:
                print('0 - Steps are too small.')
            if self.stop[2] == 1:
                print('1 - The improvement in the relative error is too small.')
            if self.stop[2] == 2:
                print('2 - The gradient is close enough to 0.')
            if self.stop[2] == 3:
                print('3 - The average of the last k relative errors is too small, where k = 1 + int(maxiter/10).')
            if self.stop[2] == 4:
                print('4 - Limit of iterations was been reached.')
            if self.stop[2] == 5:
                print('5 - No refinement was performed.')
            if self.stop[2] == 6:
                print('6 - dGN diverged.')

            return 

    output = info()

    return output
