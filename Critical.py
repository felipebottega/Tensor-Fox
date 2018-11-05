import numpy as np
from numba import jit, njit, prange


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
            M[i*b1 : (i+1)*b1 , :] = A[i, :]*B

    return M 


@njit(nogil=True)
def prepare_data(X, Y, Z, m, n, p, r):
    """
    This function creates several auxiliar matrices which will be used later 
    to accelerate matrix-vector products.
    """

    "B_X"
    w_X = np.zeros(r, dtype = np.float64)
    Mw_X = np.zeros(n*p, dtype = np.float64) 
    Bv_X = np.zeros(m*n*p, dtype = np.float64) 
    M_X = np.zeros((n*p, r), dtype = np.float64)
    M_X = -khatri_rao(Y, Z, M_X)

    "B_Y"
    w_Y = np.zeros(r, dtype = np.float64)
    Mw_Y = np.zeros(m*p, dtype = np.float64) 
    Bv_Y = np.zeros(m*n*p, dtype = np.float64)
    M_Y = np.zeros((m*p, r), dtype = np.float64)
    M_Y = -khatri_rao(X, Z, M_Y)

    "B_Z"
    w_Z = np.zeros(p*r, dtype = np.float64)
    Bv_Z = np.zeros(m*n*p, dtype = np.float64)
    M_Z = np.zeros((m*n,r), dtype = np.float64) 
    M_Z = -khatri_rao(X, Y, M_Z) 

    "B_X^T"
    w_Xt = np.zeros(n*p, dtype = np.float64)
    Mw_Xt = np.zeros(r, dtype = np.float64)
    Bu_Xt = np.zeros(r*m, dtype = np.float64) 
    N_X = np.zeros((r, n*p), dtype = np.float64)
    N_X = M_X.transpose()

    "B_Y^T"
    w_Yt = np.zeros(m*p, dtype = np.float64)
    Mw_Yt = np.zeros(r, dtype = np.float64)
    Bu_Yt = np.zeros(r*n, dtype = np.float64) 
    N_Y = np.zeros((r, m*p), dtype = np.float64)
    N_Y = M_Y.transpose()

    "B_Z^T"
    w_Zt = np.zeros((p,m*n), dtype = np.float64)
    Bu_Zt = np.zeros(r*p, dtype = np.float64) 
    Mu_Zt = np.zeros(r, dtype = np.float64)
    N_Z = np.zeros((r, m*n), dtype = np.float64)
    N_Z = M_Z.transpose()
    
    return w_X, Mw_X, Bv_X, M_X, w_Y, Mw_Y, Bv_Y, M_Y, w_Z, Bv_Z, M_Z, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z


@njit(nogil=True)
def matvec(X, Y, v, w_X, Mw_X, Bv_X, M_X, w_Y, Mw_Y, Bv_Y, M_Y, w_Z, Bv_Z, M_Z, m, n, p, r):  
    """    
    Computes the matrix-vector product Dres*v.
    """

    "B_X"
    for i in range(0, m):
        w_X = v[i + m*np.arange(0, r)] 
        Mw_X = np.dot(M_X, w_X)
        Bv_X[i*n*p : (i+1)*n*p] = Mw_X

    "B_Y"
    for j in range(0, n):
        w_Y = v[m*r + j + n*np.arange(0, r)] 
        Mw_Y = np.dot(M_Y, w_Y)
        for i in range(0, m):
            Bv_Y[j*p + i*n*p : (j+1)*p + i*n*p] = Mw_Y[i*p : (i+1)*p]

    "B_Z"
    w_Z = v[r*(m+n):]
    w_Z = w_Z.reshape(r, p).transpose()
    for k in range(0, p):
        Mw_Z = np.dot(M_Z, w_Z[k,:])
        Bv_Z[k + p*np.arange(0, m*n)] = Mw_Z

    return Bv_X + Bv_Y + Bv_Z


@njit(nogil=True)
def rmatvec(X, Y, u, w_Xt, Mw_Xt, Bu_Xt, N_X, w_Yt, Mw_Yt, Bu_Yt, N_Y, w_Zt, Bu_Zt, Mu_Zt, N_Z, m, n, p, r):     
    """    
    Computes the matrix-vector product Dres.transpose*v.
    """
 
    "B_Xt"
    for i in range(0, m):
        w_Xt = u[i*n*p : (i+1)*n*p] 
        Mw_Xt = np.dot(N_X, w_Xt)
        Bu_Xt[i + m*np.arange(0, r)] = Mw_Xt

    "B_Yt"
    for j in range(0, n):
        for i in range(0, m):
            w_Yt[i*p : (i+1)*p] = u[j*p + i*n*p : (j+1)*p + i*n*p] 
        Mw_Yt = np.dot(N_Y, w_Yt)
        Bu_Yt[j + n*np.arange(0, r)] = Mw_Yt

    "B_Zt"
    w_Zt = u.reshape(m*n, p).transpose()
    for k in range(0, p):
        Mu_Zt = np.dot(N_Z, w_Zt[k,:])
        Bu_Zt[k + p*np.arange(0, r)] = Mu_Zt

    return np.hstack((Bu_Xt, Bu_Yt, Bu_Zt))
