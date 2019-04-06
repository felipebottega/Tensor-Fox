"""
Critical Module
 
 This module is responsible for the most costly parts. Below we list all funtions presented in this module.

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

"""

import numpy as np
from numba import jit, njit, prange


@njit(nogil=True)
def kronecker(A, B):
    """
    Computes the Kronecker product between A and B. We must have 
    M.shape = (a1*b1, a2*b2), where A.shape = (a1, a2) and B.shape = (b1, b2). 
    """

    a1, a2 = A.shape
    b1, b2 = B.shape
    M = np.zeros((a1*b1, a2*b2), dtype = np.float64)
    
    for i in range(0, a1-1):
        for j in range(0, a2-1):
            M[i*b1 : (i+1)*b1, j*b2 : (j+1)*b2] = A[i, j]*B

    return M 


@njit(nogil=True, parallel=True)
def khatri_rao(A, B, M):
    """
    Computes the Khatri-Rao product between A and B. We must have 
    M.shape = (a1*b1, a2), where A.shape = (a1, a2) and B.shape = (b1, b2),
    with a2 == b2. 
    """

    a1, a2 = A.shape
    b1, b2 = B.shape
    
    for i in prange(0, a1):
        M[i*b1 : (i+1)*b1, :] = khatri_rao_inner_computations(A, B, M, i, b1, b2)

    return M 


@njit(nogil=True)
def khatri_rao_inner_computations(A, B, M, i, b1, b2):
    """
    Computes A[i, :]*B.
    """

    for k in range(0, b1):
        for j in range(0, b2):
            M[i*b1 + k, j] = A[i,j]*B[k,j]

    return M[i*b1 : (i+1)*b1, :] 


@njit(nogil=True)
def gramians(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ):
    """ 
    Computes all Gramians matrices of X, Y, Z. Also it computes all Hadamard
    products between the different Gramians. 
    """
    
    Gr_X = np.dot(X.T, X)
    Gr_Y = np.dot(Y.T, Y)
    Gr_Z = np.dot(Z.T, Z)
    Gr_XY = Gr_X*Gr_Y
    Gr_XZ = Gr_X*Gr_Z
    Gr_YZ = Gr_Y*Gr_Z
            
    return Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ


@njit(nogil=True, parallel=True)
def hadamard(A, B, M, r):
    """
    Computes M = A * B, where * is the Hadamard product. Since all Hadamard
    products in this context are between r x r matrices, we assume this
    without verifications.
    """
    
    for i in prange(0, r):
            M[i,:] = A[i, :]*B[i, :]

    return M 


@njit(nogil=True, parallel=True)
def vec(M, Bv, num_rows, r):
    """ 
    Take a matrix M with shape (num_rows, r) and stack vertically its columns
    to form the matrix Bv = vec(M) with shape (num_rows*r,).
    """
    
    for j in prange(0, r):
        Bv[j*num_rows : (j+1)*num_rows] = M[:,j]
        
    return Bv


@njit(nogil=True, parallel=True)
def vect(M, Bv, num_cols, r):
    """ 
    Take a matrix M with shape (r, num_cols) and stack vertically its rows
    to form the matrix Bv = vec(M) with shape (num_cols*r,).
    """
    
    for i in prange(0, r):
        Bv[i*num_cols : (i+1)*num_cols] = M[i,:]
        
    return Bv


def prepare_data(m, n, p, r):
    """
    Initialize all necessary matrices to keep the values of several computations
    during the program.
    """

    # Grammians
    Gr_X = np.zeros((r,r), dtype = np.float64)
    Gr_Y = np.zeros((r,r), dtype = np.float64)
    Gr_Z = np.zeros((r,r), dtype = np.float64)
    Gr_XY = np.zeros((r,r), dtype = np.float64)
    Gr_XZ = np.zeros((r,r), dtype = np.float64)
    Gr_YZ = np.zeros((r,r), dtype = np.float64)
    
    # V_X^T, V_Y^T, V_Z^T
    V_Xt = np.zeros((r,m), dtype = np.float64)
    V_Yt = np.zeros((r,n), dtype = np.float64)
    V_Zt = np.zeros((r,p), dtype = np.float64)

    # Initializations of matrices to receive the results of the computations.
    V_Xt_dot_X = np.zeros((r,r), dtype = np.float64)
    V_Yt_dot_Y = np.zeros((r,r), dtype = np.float64)
    V_Zt_dot_Z = np.zeros((r,r), dtype = np.float64)
    Gr_Z_V_Yt_dot_Y = np.zeros((r,r), dtype = np.float64)
    Gr_Y_V_Zt_dot_Z = np.zeros((r,r), dtype = np.float64)
    Gr_X_V_Zt_dot_Z = np.zeros((r,r), dtype = np.float64)
    Gr_Z_V_Xt_dot_X = np.zeros((r,r), dtype = np.float64)
    Gr_Y_V_Xt_dot_X = np.zeros((r,r), dtype = np.float64)
    Gr_X_V_Yt_dot_Y = np.zeros((r,r), dtype = np.float64)    
    X_dot_Gr_Z_V_Yt_dot_Y = np.zeros((r,r), dtype = np.float64)
    X_dot_Gr_Y_V_Zt_dot_Z = np.zeros((r,r), dtype = np.float64)
    Y_dot_Gr_X_V_Zt_dot_Z = np.zeros((r,r), dtype = np.float64)
    Y_dot_Gr_Z_V_Xt_dot_X = np.zeros((r,r), dtype = np.float64)
    Z_dot_Gr_Y_V_Xt_dot_X = np.zeros((r,r), dtype = np.float64)
    Z_dot_Gr_X_V_Yt_dot_Y = np.zeros((r,r), dtype = np.float64)
    
    # Matrices for the diagonal block
    Gr_YZ_V_Xt = np.zeros((m,r), dtype = np.float64)
    Gr_XZ_V_Yt = np.zeros((n,r), dtype = np.float64)
    Gr_XY_V_Zt = np.zeros((p,r), dtype = np.float64)
    
    # Final blocks
    B_X_v = np.zeros(m*r, dtype = np.float64)
    B_Y_v = np.zeros(n*r, dtype = np.float64)
    B_Z_v = np.zeros(p*r, dtype = np.float64)
    B_XY_v = np.zeros(m*r, dtype = np.float64)
    B_XZ_v = np.zeros(m*r, dtype = np.float64)
    B_YZ_v = np.zeros(n*r, dtype = np.float64)
    B_XYt_v = np.zeros(n*r, dtype = np.float64)
    B_XZt_v = np.zeros(p*r, dtype = np.float64) 
    B_YZt_v = np.zeros(p*r, dtype = np.float64)

    # Matrices to use when constructing the Tikhonov matrix for regularization.
    X_norms = np.zeros(r, dtype = np.float64)
    Y_norms = np.zeros(r, dtype = np.float64)
    Z_norms = np.zeros(r, dtype = np.float64)
    gamma_X = np.zeros(r, dtype = np.float64)
    gamma_Y = np.zeros(r, dtype = np.float64)
    gamma_Z = np.zeros(r, dtype = np.float64)
    Gamma = np.zeros(r*(m+n+p), dtype = np.float64)

    # Arrays to be used in the Conjugated Gradient.
    M = np.ones(r*(m+n+p), dtype = np.float64)
    L = np.ones(r*(m+n+p), dtype = np.float64)    
    residual_cg = np.zeros(r*(m+n+p), dtype = np.float64)
    P = np.zeros(r*(m+n+p), dtype = np.float64)
    Q = np.zeros(r*(m+n+p), dtype = np.float64)
    z = np.zeros(r*(m+n+p), dtype = np.float64)
    
    return Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, M, L, residual_cg, P, Q, z   


@njit(nogil=True)
def prepare_data_rmatvec(m, n, p, r):
    """
    This function creates several auxiliar matrices which will be used later 
    to accelerate matrix-vector products.
    """

    M_X = np.zeros((n*p, r), dtype = np.float64)
    
    M_Y = np.zeros((m*p, r), dtype = np.float64)
    
    M_Z = np.zeros((m*n,r), dtype = np.float64) 
    
    # B_X^T
    w_Xt = np.zeros(n*p, dtype = np.float64)
    Mw_Xt = np.zeros(r, dtype = np.float64)
    Bu_Xt = np.zeros(r*m, dtype = np.float64) 
    N_X = np.zeros((r, n*p), dtype = np.float64)
    
    # B_Y^T
    w_Yt = np.zeros(m*p, dtype = np.float64)
    Mw_Yt = np.zeros(r, dtype = np.float64)
    Bu_Yt = np.zeros(r*n, dtype = np.float64) 
    N_Y = np.zeros((r, m*p), dtype = np.float64)
    
    # B_Z^T
    w_Zt = np.zeros((p,m*n), dtype = np.float64)
    Bu_Zt = np.zeros(r*p, dtype = np.float64) 
    Mu_Zt = np.zeros(r, dtype = np.float64)
    N_Z = np.zeros((r, m*n), dtype = np.float64)
        
    return M_X, M_Y, M_Z, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z


@njit(nogil=True)
def update_data_rmatvec(X, Y, Z, M_X, M_Y, M_Z):
    """
    This function creates several auxiliar matrices which will be used later 
    to accelerate matrix-vector products.
    """

    M_X = -khatri_rao(Y, Z, M_X)

    M_Y = -khatri_rao(X, Z, M_Y)

    M_Z = -khatri_rao(X, Y, M_Z) 
    
    return M_X, M_Y, M_Z


@njit(nogil=True)
def matvec(X, Y, Z, Gr_X, Gr_Y, Gr_Z, Gr_XY, Gr_XZ, Gr_YZ, V_Xt, V_Yt, V_Zt, V_Xt_dot_X, V_Yt_dot_Y, V_Zt_dot_Z, Gr_Z_V_Yt_dot_Y, Gr_Y_V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, Gr_Z_V_Xt_dot_X, Gr_Y_V_Xt_dot_X, Gr_X_V_Yt_dot_Y, X_dot_Gr_Z_V_Yt_dot_Y, X_dot_Gr_Y_V_Zt_dot_Z, Y_dot_Gr_X_V_Zt_dot_Z, Y_dot_Gr_Z_V_Xt_dot_X, Z_dot_Gr_Y_V_Xt_dot_X, Z_dot_Gr_X_V_Yt_dot_Y, Gr_YZ_V_Xt, Gr_XZ_V_Yt, Gr_XY_V_Zt, B_X_v, B_Y_v, B_Z_v, B_XY_v, B_XZ_v, B_YZ_v, B_XYt_v, B_XZt_v, B_YZt_v, v, m, n, p, r): 
    """
    Makes the matrix-vector computation (Dres.transpose*Dres)*v. 
    """
     
    # Split v into three blocks, convert them into matrices and transpose them. 
    # With this we have the matrices V_X^T, V_Y^T, V_Z^T.
    V_Xt = v[0 : m*r].reshape(r, m)
    V_Yt = v[m*r : r*(m+n)].reshape(r, n)
    V_Zt = v[r*(m+n) : r*(m+n+p)].reshape(r, p)
       
    # Compute the products V_X^T*X, V_Y^T*Y, V_Z^T*Z
    V_Xt_dot_X = np.dot(V_Xt, X)
    V_Yt_dot_Y = np.dot(V_Yt, Y)
    V_Zt_dot_Z = np.dot(V_Zt, Z)
    
    # Compute the Hadamard products
    Gr_Z_V_Yt_dot_Y = hadamard(Gr_Z, V_Yt_dot_Y, Gr_Z_V_Yt_dot_Y, r)
    Gr_Y_V_Zt_dot_Z = hadamard(Gr_Y, V_Zt_dot_Z, Gr_Y_V_Zt_dot_Z, r)
    Gr_X_V_Zt_dot_Z = hadamard(Gr_X, V_Zt_dot_Z, Gr_X_V_Zt_dot_Z, r)
    Gr_Z_V_Xt_dot_X = hadamard(Gr_Z, V_Xt_dot_X, Gr_Z_V_Xt_dot_X, r)
    Gr_Y_V_Xt_dot_X = hadamard(Gr_Y, V_Xt_dot_X, Gr_Y_V_Xt_dot_X, r)
    Gr_X_V_Yt_dot_Y = hadamard(Gr_X, V_Yt_dot_Y, Gr_X_V_Yt_dot_Y, r)
    
    # Compute the final products
    X_dot_Gr_Z_V_Yt_dot_Y = np.dot(X, Gr_Z_V_Yt_dot_Y)
    X_dot_Gr_Y_V_Zt_dot_Z = np.dot(X, Gr_Y_V_Zt_dot_Z)
    Y_dot_Gr_X_V_Zt_dot_Z = np.dot(Y, Gr_X_V_Zt_dot_Z)
    Y_dot_Gr_Z_V_Xt_dot_X = np.dot(Y, Gr_Z_V_Xt_dot_X)
    Z_dot_Gr_Y_V_Xt_dot_X = np.dot(Z, Gr_Y_V_Xt_dot_X)
    Z_dot_Gr_X_V_Yt_dot_Y = np.dot(Z, Gr_X_V_Yt_dot_Y)
    
    # Diagonal block matrices
    Gr_YZ_V_Xt = np.dot(Gr_YZ, V_Xt)
    Gr_XZ_V_Yt = np.dot(Gr_XZ, V_Yt)
    Gr_XY_V_Zt = np.dot(Gr_XY, V_Zt)
    
    # Vectorize the matrices to have the final vectors
    B_X_v = vect(Gr_YZ_V_Xt, B_X_v, m, r)
    B_Y_v = vect(Gr_XZ_V_Yt, B_Y_v, n, r)
    B_Z_v = vect(Gr_XY_V_Zt, B_Z_v, p, r)
    B_XY_v = vec(X_dot_Gr_Z_V_Yt_dot_Y, B_XY_v, m, r)
    B_XZ_v = vec(X_dot_Gr_Y_V_Zt_dot_Z, B_XZ_v, m, r)
    B_YZ_v = vec(Y_dot_Gr_X_V_Zt_dot_Z, B_YZ_v, n, r)
    B_XYt_v = vec(Y_dot_Gr_Z_V_Xt_dot_X, B_XYt_v, n, r)
    B_XZt_v = vec(Z_dot_Gr_Y_V_Xt_dot_X, B_XZt_v, p, r) 
    B_YZt_v = vec(Z_dot_Gr_X_V_Yt_dot_Y, B_YZt_v, p, r)

    return np.concatenate((B_X_v + B_XY_v + B_XZ_v, B_XYt_v + B_Y_v + B_YZ_v, B_XZt_v + B_YZt_v + B_Z_v)) 


@njit(nogil=True)
def rmatvec(u, w_Xt, Mw_Xt, Bu_Xt, M_X, w_Yt, Mw_Yt, Bu_Yt, M_Y, w_Zt, Bu_Zt, Mu_Zt, M_Z, m, n, p, r):     
    """    
    Computes the matrix-vector product Dres.transpose*u.
    """
 
    "B_Xt"
    for i in range(0, m):
        w_Xt = u[i*n*p : (i+1)*n*p] 
        Mw_Xt = np.dot(w_Xt, M_X).transpose()
        Bu_Xt[i + m*np.arange(0, r)] = Mw_Xt

    "B_Yt"
    for j in range(0, n):
        for i in range(0, m):
            w_Yt[i*p : (i+1)*p] = u[j*p + i*n*p : (j+1)*p + i*n*p] 
        Mw_Yt = np.dot(w_Yt, M_Y).transpose()
        Bu_Yt[j + n*np.arange(0, r)] = Mw_Yt

    "B_Zt"
    w_Zt = u.reshape(m*n, p).transpose()
    for k in range(0, p):
        Mu_Zt = np.dot(w_Zt[k,:], M_Z).transpose()
        Bu_Zt[k + p*np.arange(0, r)] = Mu_Zt

    return np.hstack((Bu_Xt, Bu_Yt, Bu_Zt))


@njit(nogil=True)
def regularization(X, Y, Z, X_norms, Y_norms, Z_norms, gamma_X, gamma_Y, gamma_Z, Gamma, m, n, p, r):
    """
    Computes the Tikhonov matrix Gamma, where Gamma is a diagonal matrix designed
    specifically to make Dres.transpose*Dres + Gamma diagonally dominant.
    """
        
    for l in range(0, r):
        X_norms[l] = np.sqrt( np.dot(X[:,l], X[:,l]) )
        Y_norms[l] = np.sqrt( np.dot(Y[:,l], Y[:,l]) )
        Z_norms[l] = np.sqrt( np.dot(Z[:,l], Z[:,l]) )
    
    max_XY = np.max(X_norms*Y_norms)
    max_XZ = np.max(X_norms*Z_norms)
    max_YZ = np.max(Y_norms*Z_norms)
    max_all = max(max_XY, max_XZ, max_YZ)
        
    for l in range(0, r):
        gamma_X[l] = Y_norms[l]*Z_norms[l]*max_all
        gamma_Y[l] = X_norms[l]*Z_norms[l]*max_all
        gamma_Z[l] = X_norms[l]*Y_norms[l]*max_all
        
    for l in range(0, r):
        Gamma[l*m : (l+1)*m] = gamma_X[l]
        Gamma[m*r+l*n : m*r+(l+1)*n] = gamma_Y[l]
        Gamma[r*(m+n)+l*p : r*(m+n)+(l+1)*p] = gamma_Z[l]
        
    return Gamma


@njit(nogil=True)
def precond(X, Y, Z, L, M, damp, m, n, p, r):
    """
    This function constructs a preconditioner in order to accelerate the Conjugate Gradient fucntion.
    It is a diagonal preconditioner designed to make Dres.transpose*Dres + Gamma a unit diagonal matrix. Since 
    the matrix is diagonally dominant, the result will be close to the identity matrix. Therefore, it will be
    very well conditioned with its eigenvalues clustered together.
    """
    for l in range(0, r):
        M[l*m : (l+1)*m] = np.dot(Y[:,l], Y[:,l])*np.dot(Z[:,l], Z[:,l]) + damp*L[l*m : (l+1)*m] 
        M[m*r+l*n : m*r+(l+1)*n] = np.dot(X[:,l], X[:,l])*np.dot(Z[:,l], Z[:,l]) + damp*L[m*r+l*n : m*r+(l+1)*n] 
        M[r*(m+n)+l*p : r*(m+n)+(l+1)*p] = np.dot(X[:,l], X[:,l])*np.dot(Y[:,l], Y[:,l]) + damp*L[r*(m+n)+l*p : r*(m+n)+(l+1)*p]    
        
    M = 1/np.sqrt(M)
    return M
