"""
Construction Module
 
 As we mentioned in the main module *Tensor Fox*, the module *Construction* is responsible for constructing the more complicated objects necessary to make computations. Between these objects we have the array of residuals, the derivative of the residuals, the starting points to begin the iterations and so on. Below we list all funtions presented in this module.
 
 - residual
 
 - residual_entries
 
 - residual_derivative_structure
 
 - concat
 
 - residual_derivative
 
 - start_point
 
 - smart_random
 
 - smart_sample
 
 - assign_values
 
 - truncation
 
 - truncate1
 
 - truncate2
""" 


import numpy as np
import sys
import scipy.io
import time
import matplotlib.pyplot as plt
from scipy import sparse
from numba import jit, njit, prange
import Conversion as cnv
import Auxiliar as aux


@njit(nogil=True,parallel=True)
def residual(res, T, Lambda, X, Y, Z, r, m, n, p):
    """
    This function computes (updates) the residuals between a 3-D tensor T in R^m⊗R^n⊗R^p
    and an approximation T_approx of rank r. The tensor T_approx is of the form
    T_approx = Lambda_1*X_1⊗Y_1⊗Z_1 + ... + Lambda_r*X_r⊗Y_r⊗Z_r, where
    X = [X_1, ..., X_r],
    Y = [Y_1, ..., Y_r],
    Z = [Z_1, ..., Z_r].
    
    The `residual map` is a map res:R^{r+r(m+n+p)}->R^{m*n*p}. For each i,j,k=0...n, the residual 
    r_{i,j,k} is given by res_{i,j,k} = T_{i,j,k} - sum_{l=1}^r Lambda_l*X_{il}*Y_{jl}*Z_{kl}.
    
    Inputs
    ------
    res: float 1-D ndarray with m*n*p entries 
        Each entry is a residual.
    T: float 3-D ndarray
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)
    r: int
    m,n,p: 
    
    Outputs
    -------
    res: float 1-D ndarray with m*n*p entries 
        Each entry is a residual.
    """   
    
    s = 0
    
    #Construction of the vector res = (res_{111}, res_{112}, ..., res_{mnp}).
    for i in prange(0,m):
        for j in range(0,n):
            for k in range(0,p):
                s = n*p*i + p*j + k
                res[s] = residual_entries(T, Lambda, X, Y, Z, r, m, n, p, i, j, k)
                            
    return res


@njit(nogil=True,cache=True)
def residual_entries(T, Lambda, X, Y, Z, r, m, n, p, i, j, k):
    """Computation of each individual residual in the function 'residual'."""
    
    s = 0.0
    for l in range(0,r):
        s += Lambda[l]*X[i,l]*Y[j,l]*Z[k,l]
        
    res_ijk = T[i,j,k] - s
        
    return res_ijk


@jit(nogil=True, parallel=True)
def residual_derivative_structure(r, m, n, p):
    """
    Initialization of the matrix Dres in sparse format, i.e., a triple (data,row,col) 
    such that data is a 1-D containing the nonzero values of Dres, row is a 1-D ndarray
    containing the corresponding rows index of the elements in data and col is a 1-D
    ndarray containing the corresponding columns index of the elements in data.
    Initially, all values of data are equal to zero. This function doesn't compute any
    actual Jacobian matrix, but only initializes its sparse structure for later. 
    
    Since the coordinates corresponding to the nonzero entries never changes, we compute 
    and store them with this function. Although data is storaged row by row, it is not CSR 
    format, because the ordering of the columns is not increasing. Consider this as a mix 
    of CSR and COO format. This construction consider the ordering of the derivatives, and 
    this ordering will construct the arrays data and col with a differente ordering of the 
    standard CSR format. The array datat_id is such that data[datat_id] is the array 
    corresponding to the entries of Dres.transpose in CSR format. Similarly, colt 
    gives the columns of Dres.transpose corresponding to data[datat_id]. 
    
    Inputs
    ------
    r: int
    m,n,p: int
    
    Outputs
    -------
    data: float 1-D ndarray of ones
    row: int 1-D ndarray
    col: int 1-D ndarray
    datat_id: int 1-D ndarray
    colt: int 1-D ndarray
    """   
    
    # Initialize output arrays.
    data = np.zeros(4*m*n*p*r, dtype = np.float64)
    row = np.zeros(4*m*n*p*r, dtype = np.int64)
    col = np.zeros(4*m*n*p*r, dtype = np.int64)
    datat_id = np.zeros(4*m*n*p*r, dtype = np.int64)
    colt = np.zeros(4*m*n*p*r, dtype = np.float64)
    
    # Initialize temporary arrays. We use Fortran ordering (columnwise) on the matrices 
    # because we will need to stack their columns, and this is performed faster if the
    # memory is columnwise continuous. 
    colt1 = np.zeros((m*n*p,r), dtype = np.int64, order='F')
    colt2 = np.zeros((n*p,r*m), dtype = np.int64, order='F')
    colt3 = np.zeros((m*p,r*n), dtype = np.int64, order='F')
    colt4 = np.zeros((m*n,r*p), dtype = np.int64, order='F')
    datat_id1 = np.zeros((m*n*p,r), dtype = np.int64, order='F')
    datat_id2 = np.zeros((n*p,r*m), dtype = np.int64, order='F')
    datat_id3 = np.zeros((m*p,r*n), dtype = np.int64, order='F')
    datat_id4 = np.zeros((m*n,r*p), dtype = np.int64, order='F')
    t1 = np.zeros(r*m*n*p, dtype = np.int64)
    t2 = np.zeros(r*m, dtype = np.int64)
    t3 = np.zeros(r*n, dtype = np.int64)
    t4 = np.zeros(r*p, dtype = np.int64)
    
    # Pre-declaring the variables s,i,j,k,l prevents Numba from crashing in the parallel for.
    s = i = j = k = l = current_row = 0
        
    for i in prange(0,m):
        for j in range(0,n):
            for k in range(0,p):
                current_row = n*p*i + p*j + k
                # This last loop for l computes the entries at the row ijk of Dres.
                for l in range(0,r):
                    s = 4*(n*p*r*i + p*r*j + r*k + l)
                    # Partial derivative with respect to Lambda.
                    row[s] = current_row
                    col[s] = l
                    # Partial derivative with respect to X.
                    row[s+1] = current_row
                    col[s+1] = r + l*m + i
                    # Partial derivative with respect to Y.
                    row[s+2] = current_row
                    col[s+2] = r + r*m + l*n + j
                    # Partial derivative with respect to Z.
                    row[s+3] = current_row
                    col[s+3] = r + r*m + r*n + l*p + k
                    
    # The increments on each ti below make these loops non parallelizable. That is why 
    # we have to repeate these loops again.
    for i in range(0,m):
        for j in range(0,n):
            for k in range(0,p):
                current_row = n*p*i + p*j + k
                for l in range(0,r):
                    s = 4*(n*p*r*i + p*r*j + r*k + l)
                    # Partial derivative with respect to Lambda.
                    colt1[t1[l*r + l], l] = current_row
                    datat_id1[t1[l*r + l], l] = s
                    t1[l*r + l] += 1
                    # Partial derivative with respect to X.
                    colt2[t2[l*m + i], l*m + i] = current_row
                    datat_id2[t2[l*m + i], l*m + i] = s+1
                    t2[l*m + i] += 1
                    # Partial derivative with respect to Y.
                    colt3[t3[l*n + j], l*n + j] = current_row
                    datat_id3[t3[l*n + j], l*n + j] = s+2
                    t3[l*n + j] += 1
                    # Partial derivative with respect to Z.
                    colt4[t4[l*p + k], l*p + k] = current_row
                    datat_id4[t4[l*p + k], l*p + k] = s+3
                    t4[l*p + k] += 1 
    
    # datat_id is such that data[datat_id] gives datat in the standard ordering, where
    # datat is the datat of Dres.transpose.
    datat_id = concat(datat_id1, datat_id2, datat_id3, datat_id4, r, m, n, p)
    colt = concat(colt1, colt2, colt3, colt4, r, m, n, p)
                
    return data, row, col, datat_id, colt


@njit(nogil=True, parallel=True) 
def concat(array1, array2, array3, array4, r, m, n, p):
    """
    Concatenate given inputs with respect to the columns of Dres. This function
    is used as an auxiliary function of residual_derivative_structure. It serves 
    to concatenate colt1, colt2, colt3, colt4 to obtain colt and to concatenate
    datat_id1, datat_id2, datat_id3, datat_id4 to obtain datat_id.
    """
    
    b = np.zeros(4*r*m*n*p, dtype = np.int64)
    
    for l in prange(0,r):
        low = l*m*n*p
        high = (l+1)*m*n*p
        b[low:high] = array1[:,l]
        
    for l in prange(0,r*m):
        low = r*m*n*p + l*n*p
        high = r*m*n*p + (l+1)*n*p
        b[low:high] = array2[:,l]
        
    for l in prange(0,r*n):
        low = 2*r*m*n*p + l*m*p
        high = 2*r*m*n*p + (l+1)*m*p
        b[low:high] = array3[:,l]
        
    for l in prange(0,r*p):
        low = 3*r*m*n*p + l*m*n 
        high = 3*r*m*n*p + (l+1)*m*n
        b[low:high] = array4[:,l]
        
    return b


@njit(nogil=True, parallel=True)
def residual_derivative(data, Lambda, X, Y, Z, r, m, n, p):
    """
    Computation of the nonzero entries of the Jacobian matrix Dres of the residuals 
    map at a particular point (Lambda,X,Y,Z). The matrix Dres is sparse, and that is
    why we only keep its nonzero entries. This matrix is updated several times during 
    the program through this function. 
    
    Inputs
    ------
    data: float 1-D ndarray 
        The nonzero entries of Dres.
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,r)
    Y: float 2-D ndarray of shape (n,r)
    Z: float 2-D ndarray of shape (p,r)
    r: int. 
        The desired rank of the approximating tensor.
    m,n,p: int
        The dimensions of the spaces.
    
    Outputs
    -------
    data: float 1-D ndarray 
        The nonzero entries of Dres. 
    """    
    
    for i in prange(0,m):
        for j in range(0,n):
            for k in range(0,p):
                for l in range(0,r):
                    s = 4*(n*p*r*i + p*r*j + r*k + l)
                    #Partial derivative with respect to Lambda.
                    data[s] = -X[i,l]*Y[j,l]*Z[k,l]
                    #Partial derivative with respect to X.
                    data[s+1] = -Lambda[l]*Y[j,l]*Z[k,l]
                    #Partial derivative with respect to Y.
                    data[s+2] = -Lambda[l]*X[i,l]*Z[k,l]
                    #Partial derivative with respect to Z.
                    data[s+3] = -Lambda[l]*X[i,l]*Y[j,l]
                    
    return data


def start_point(T, Tsize, S_trunc, U1_trunc, U2_trunc, U3_trunc, r, R1_trunc, R2_trunc, R3_trunc, init='smart_random'):
    """
    This function generates a starting point to begin the iterations of the
    Gauss-Newton method. There are three options:
        'fixed': for each values of rank and dimensions this option always
    generates the same starting point, which looks random. This is good when
    one want to change the code and compare performances.
        'random': each entry of Lambda, X, Y, Z are generated by the normal
    distribution with mean 0 and variance 1.
        'smart_random': generates a random starting point with a method which
    always guarantee a small relative error. Check the function 'smart' for 
    more details about this method.
    
    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
    S_trunc: float 3-D ndarray with shape R1_trunc x R2_trunc x R3_trunc
    U1_trunc: float 2-D ndarrays with shape R1_trunc x r
    U2_trunc: float 2-D ndarrays with shape R2_trunc x r
    U3_trunc: float 2-D ndarrays with shape R3_trunc x r
    r, R1_trunc, R2_trunc, R3_trunc: int
    
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray with shape m x r
    Y: float 2-D ndarray with shape n x r
    Z: float 2-D ndarray with shape p x r
    rel_err: float
        Relative error associate to the starting point. More precisely, it is the relative 
    error between T and (U1_trunc,U2_trunc,U3_trunc)*S_init, where S_init = (X,Y,Z)*Lambda.
    """
    
    if init == 'fixed': 
        Lambda = np.cos(np.arange(r)**2)/10
        X = np.sin(np.arange(R1_trunc*r))/10
        X = X.reshape(R1_trunc,r)
        Y = np.cos(np.arange(R2_trunc*r))/10
        Y = Y.reshape(R2_trunc,r)
        Z = np.tan(np.arange(R3_trunc*r))/10
        Z = Z.reshape(R3_trunc,r)    
        
    elif init == 'random':
        Lambda = np.random.randn(r)
        X = np.random.randn(R1_trunc,r)
        Y = np.random.randn(R2_trunc,r)
        Z = np.random.randn(R3_trunc,r)
        
    elif init == 'smart_random':
        Lambda, X, Y, Z = smart_random(S_trunc, r, R1_trunc, R2_trunc, R3_trunc)
        
    else:
        sys.exit('Error with `init` parameter.') 
    
    # Computation of relative error associated with the starting point given.
    T_aux = np.zeros(S_trunc.shape, dtype = np.float64)
    S_init = cnv.CPD2tens(T_aux, Lambda, X, Y, Z, r)
    T_init = aux.multilin_mult(S_init, U1_trunc, U2_trunc, U3_trunc, R1_trunc, R2_trunc, R3_trunc) 
    rel_err = np.linalg.norm(T - T_init)/Tsize
        
    return Lambda, X, Y, Z, rel_err


def smart_random(S_trunc, r, R1, R2, R3, samples=100):
    """
    100 samples of random possible initializations are generated and compared. We
    keep the closest to S_trunc. This method draws r points in S_trunc and generates
    a tensor with rank <= r from them. The distribution is such that it tries to
    maximize the energy of the sampled tensor, so the error is minimized.
    Althought we are using the variables named as R1, R2, R3, remember they refer to
    R1_trunc, R2_trunc, R3_trunc at the main function.
    
    Inputs
    ------
    S_trunc: 3-D float ndarray
    r: int
    R1, R2, R3: int
        The dimensions of the truncated tensor S_trunc.
    samples: int
        The number of tensors drawn randomly. Default is 100.
        
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,R1)
    Y: float 2-D ndarray of shape (n,R2)
    Z: float 2-D ndarray of shape (p,R3)
    """
    
    # Initialize auxiliary values and arrays.
    best_loss = np.inf
    S_truncsize = np.linalg.norm(S_trunc)
    T_aux = np.zeros(S_trunc.shape, dtype = np.float64)

    # Start search for a good initial point.
    for sample in range(0,samples):
        Lambda, X, Y, Z = smart_sample(S_trunc, r, R1, R2, R3)
        S_initial = cnv.CPD2tens(T_aux, Lambda, X, Y, Z, r)
        loss = np.linalg.norm(S_trunc - S_initial)/S_truncsize
        if loss < best_loss:
            best_loss = loss
            best_Lambda, best_X, best_Y, best_Z = Lambda, X, Y, Z

    return best_Lambda, best_X, best_Y, best_Z


@jit(nogil=True, parallel=True)
def smart_sample(S_trunc, r, R1, R2, R3):
    """
    We consider a distribution that gives more probability to smaller coordinates. This 
    is because these are associated with more energy. We choose a random number c1 in the 
    integer interval [0, R1 + (R1-1) + (R1-2) + ... + 1]. If 0 <= c1 < R1, we choose i = 1,
    if R1 <= c1 < R1 + (R1-1), we choose i = 2, and so on. The same goes for the other
    coordinates.
    Let S_{i_l,j_l,k_l}, l = 1...r, be the points chosen by this method. With them we form
    the tensor S_init = sum_{l=1}^r S_{i_l,j_l,k_l} e_{i_l} ⊗ e_{j_l} ⊗ e_{k_l}, which 
    should be close to S_trunc.
    
    Inputs
    ------
    S_trunc: 3-D float ndarray
    r: int
    R1, R2, R3: int
    samples: int
    
    Ouputs
    ------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray of shape (m,R1)
    Y: float 2-D ndarray of shape (n,R2)
    Z: float 2-D ndarray of shape (p,R3)
    """
    
    # Initialize arrays to construct initial approximate CPD.
    Lambda = np.zeros(r, dtype = np.float64)
    X = np.zeros((R1,r), dtype = np.float64)
    Y = np.zeros((R2,r), dtype = np.float64)
    Z = np.zeros((R3,r), dtype = np.float64)
    # Construct the upper bounds of the intervals.
    arr1 = R1*np.ones(R1, dtype = np.int64) - np.arange(R1)
    arr2 = R2*np.ones(R2, dtype = np.int64) - np.arange(R2)
    arr3 = R3*np.ones(R3, dtype = np.int64) - np.arange(R3)
    high1 = np.sum(arr1)
    high2 = np.sum(arr2)
    high3 = np.sum(arr3)

    # Arrays with all random choices.
    C1 = np.random.randint(high1, size=r)
    C2 = np.random.randint(high2, size=r)  
    C3 = np.random.randint(high3, size=r)

    # Update arrays based of the choices made.
    for l in prange(0,r):
        Lambda[l], X[:,l], Y[:,l], Z[:,l] = assign_values(S_trunc, Lambda, X, Y, Z, r, R1, R2, R3, C1, C2, C3, arr1, arr2, arr3, l) 
          
    return Lambda, X, Y, Z


@jit(nogil=True, cache=True)
def assign_values(S_trunc, Lambda, X, Y, Z, r, R1, R2, R3, C1, C2, C3, arr1, arr2, arr3, l):
    """
    For each l = 1...r, this function constructs l-th one rank term in the CPD of the
    initialization tensor, which is of the form S_trunc[i,j,k]*e_i ⊗ e_j ⊗ e_k for some
    i,j,k choosed through the random distribution described earlier.
    """
    
    for i in range(0,R1):
        if (np.sum(arr1[0:i]) <= C1[l]) and (C1[l] < np.sum(arr1[0:i+1])):
            X[i,l] = 1
            break
    for j in range(0,R2):
        if (np.sum(arr2[0:j]) <= C2[l]) and (C2[l] < np.sum(arr2[0:j+1])):
            Y[j,l] = 1
            break
    for k in range(0,R3):
        if (np.sum(arr3[0:k]) <= C3[l]) and (C3[l] < np.sum(arr3[0:k+1])):
            Z[k,l] = 1
            break    
            
    Lambda[l] = S_trunc[i,j,k]
    
    return Lambda[l], X[:,l], Y[:,l], Z[:,l]


def truncation(T, Tsize, S, U1, U2, U3, r, sigma1, sigma2, sigma3, energy):
    """
    This function computes an adequate truncation for the central tensor S of the
    HOSVD of T.
    There are three possibilities: 
        1) No truncation (energy == 100)
        2) Truncation by energy (1 <= energy < 100)
        3) Truncation by relative error (0 < energy < 1)
    The energy parameter is a double purpose parameter, acting as two different things
    depending on the value passed to it.
    When 1 <= energy < 100, the program uses the function 'truncate1', which computes
    the truncation with lowest energy bigger than the value 'energy'.
    When 0 < energy < 1, the program consider the value 'energy' as a relative error.
    Then it computes the smallest truncation S_trunc (and its respectives truncated
    matrices U1_trunc, U2_trunc, U3_trunc) such that the relative error between T and
    (U1_trunc,U2_trunc,U3_trunc)*S_trunc is lower than 'energy'.
    
    Inputs
    ------
    T: float 3-D ndarray
    Tsize: float
        norm of T.
    S: float 3-D ndarray
    U1, U2, U3: float 2-D ndarrays
    r: int 
    sigma1, sigma2, sigma3: float 1-D arrays
    energy: float
    
    Outputs
    -------
    S_trunc: float 3-D ndarray with shape R1_trunc x R2_trunc x R3_trunc
        Truncated central tensor.
    U1_trunc: float 2-D ndarrays with shape R1_trunc x r
    U2_trunc: float 2-D ndarrays with shape R2_trunc x r
    U3_trunc: float 2-D ndarrays with shape R3_trunc x r
    best_energy: float
        The energy of the truncation. The biggest is the energy, closer to T is the truncation
    R1_trunc, R2_trunc, R3_trunc: int
        The reduced dimensions obtained after truncating.
    rel_err: float
        Relative error associate to the truncation. More precisely, it is the relative error
    between T and (U1_trunc,U2_trunc,U3_trunc)*S_trunc.
    """
    
    if energy == 100:
        best_energy = 100
        
    elif (energy >= 1) and (energy < 100) :
        best_energy, R1_trunc, R2_trunc, R3_trunc = truncate1(sigma1, sigma2, sigma3, energy=energy) 
        
    elif (energy > 0) and (energy < 1):
        best_energy, R1_trunc, R2_trunc, R3_trunc = truncate2(T, S, U1, U2, U3, r, sigma1, sigma2, sigma3, rel_error=energy)
    
    else:
        sys.exit('Invalid energy value.')
        
    # Construct truncations of S, U1, U2, U3.    
    if best_energy == 100:
        S_trunc = S
        U1_trunc = U1
        U2_trunc = U2
        U3_trunc = U3  
        
    else:
        S_trunc = S[:R1_trunc,:R2_trunc,:R3_trunc]
        U1_trunc = U1[:,:R1_trunc]
        U2_trunc = U2[:,:R2_trunc]
        U3_trunc = U3[:,:R3_trunc]  
    
    # Computation of relative error associated with truncation. 
    T_trunc = aux.multilin_mult(S_trunc, U1_trunc, U2_trunc, U3_trunc, R1_trunc, R2_trunc, R3_trunc) 
    rel_err = np.linalg.norm(T - T_trunc)/Tsize
        
    return S_trunc, U1_trunc, U2_trunc, U3_trunc, best_energy, R1_trunc, R2_trunc, R3_trunc, rel_err


@njit(nogil=True, cache=True)
def truncate1(sigma1, sigma2, sigma3, energy=100):
    """
    sigma1, sigma2, sigma3 are the list of singular values of the unfoldings of T. Using
    this lists we start to truncating S with respect to some energy in the interval [1,100],
    where 100 means no truncation at all.
    Remember the energy associated to some truncation S_trunc is the value |S_trunc|/|S|*100.
    Given the value 'energy', this function searches for the truncation more energy than
    'energy' but as close as possible to 'energy'. It is the infimum of all truncations with
    more energy than 'energy'.
    
    Inputs
    ------
    sigma1, sigma2, sigma3: 1-D float ndarrays
    energy: float    
    """
    
    # Initialize values and arrays.
    best_energy = 100
    size1 = sigma1.shape[0]
    size2 = sigma2.shape[0]
    size3 = sigma3.shape[0]
    best_r1 = sigma1.shape[0]
    best_r2 = sigma2.shape[0]
    best_r3 = sigma3.shape[0]
    
    # Start to search for the best truncation based on energy.
    for r1 in range(1,size1): 
        for r2 in range(1,size2):
            for r3 in range(1,size3):
                # Compute the energy of the truncation.
                total_energy = (np.sum(sigma1[0:r1+1]) + np.sum(sigma2[0:r2+1]) + np.sum(sigma3[0:r3+1]))/(np.sum(sigma1) + np.sum(sigma2) + np.sum(sigma3))*100
                # Verify if the energy is good enough. 
                if (total_energy > energy) and (total_energy < best_energy):
                    best_energy = total_energy
                    best_r1 = r1+1
                    best_r2 = r2+1
                    best_r3 = r3+1 
                                          
    return best_energy, best_r1, best_r2, best_r3


@jit(nogil=True, cache=True)
def truncate2(T, S, U1, U2, U3, r, sigma1, sigma2, sigma3, rel_error=0.05):
    """
    We choose the smallest truncation with relative error < 0.05 (default) with 
    respect to the original tensor. More precisely, this relative error is given by
    |T - (U1_trunc, U2_trunc, U3_trunc)*S_trunc|/|T|.
    We inspect all truncations with more than 80% of the total energy.
    
    Inputs
    ------
    T: float 3-D ndarray
    S: float 3-D ndarray
    U1, U2, U3: float 2-D ndarrays
    r: int
    sigma1, sigma2, sigma3: float 1-D arrays
    rel_error: float
        We have 0 < rel_error < 1, with default of 0.05.
        
    Outputs
    -------
    best_energy: float
        The energy of the best truncation obtained.
    best_r1, best_r2, best_r3: int
        The dimensions of the best truncation obtained.    
    """
    
    # Initialize values and arrays.
    best_energy = 100
    R1 = sigma1.shape[0]
    R2 = sigma2.shape[0]
    R3 = sigma3.shape[0]
    best_r1 = R1
    best_r2 = R2
    best_r3 = R3
    best_dims = R1*R2*R3
    energy_range = np.arange(80,100,0.1)
    Tsize = np.sqrt(np.sum(T**2))
    
    # Start to search for the best truncation based on relative error.
    for energy in energy_range:   
        # Construct truncation with a certain energy.
        current_energy, R1_trunc, R2_trunc, R3_trunc = truncate1(sigma1, sigma2, sigma3, energy=energy)
        S_trunc = S[:R1_trunc,:R2_trunc,:R3_trunc]
        U1_trunc = U1[:,:R1_trunc]
        U2_trunc = U2[:,:R2_trunc]
        U3_trunc = U3[:,:R3_trunc]  
        # Compute the relative error of this truncation.
        T_trunc = aux.multilin_mult(S_trunc, U1_trunc, U2_trunc, U3_trunc, R1_trunc, R2_trunc, R3_trunc) 
        rel_err = np.sqrt(np.sum((T - T_trunc)**2))/Tsize
        max_r = min(R1_trunc*R2_trunc, R1_trunc*R3_trunc, R2_trunc*R3_trunc)
        # Verify if this truncation is good enough.
        if (rel_err < rel_error) and (R1_trunc*R2_trunc*R3_trunc < best_dims) and (r <= max_r):
            best_energy = current_energy
            best_dims = R1_trunc*R2_trunc*R3_trunc
            best_r1 = R1_trunc
            best_r2 = R2_trunc
            best_r3 = R3_trunc
            # It is not necessary to inspect all possible truncations. 
            # From 90% we accept the best computed truncation so far.
            if energy > 90:
                return best_energy, best_r1, best_r2, best_r3
           
    return best_energy, best_r1, best_r2, best_r3