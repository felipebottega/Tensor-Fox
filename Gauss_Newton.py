import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl
from scipy import sparse
from numba import jit, njit, prange

@njit(nogil=True,parallel=True)
def residuals(T,Lambda,X,Y,Z,r,n):
    """
    This function computes the residuals between a 3-D tensor T in R^{n+1}⊗R^{n+1}⊗R^{n+1}
    and an approximation S of rank r. The tensor S is of the form
    S = Lambda_1*X_1⊗Y_1⊗Z_1 + ... + Lambda_r*X_r⊗Y_r⊗Z_r, where
    X_l = (1, X_{l_1}, ..., X_{l_n}),
    Y_l = (1, Y_{l_1}, ..., Y_{l_n}),
    Z_l = (1, Z_{l_1}, ..., Z_{l_n}).
    
    The `residual map` is a map res:R^{n+1}->R. For each i,j,k=0...n, the residual r_{i,j,k} 
    is given by res_{i,j,k} = ( T_{i,j,k} - sum_{l=1}^r Lambda_l*X_{l_i}*Y_{l_j}*Z_{l_k} )^2.
    
    The entries X_{l_0} = 1, Y_{l_0} = 1, Z_{l_0} = 1 are not passed to the function. Thus,
    instead of passing X = (X_1,...,X_r), Y = (Y_1,...,Y_r), Z = (Z_1,...,Z_r) as vectors 
    with r*(n+1) entries, they are passed with r*n entries.
    
    Inputs
    ------
    T: float 3-D ndarray
    Lambda: float 1-D ndarray with r entries
    X: float 1-D ndarray with r*n entries
    Y: float 1-D ndarray with r*n entries
    Z: float 1-D ndarray with r*n entries
    r: int.
        The rank of the desired approximating tensor.
    n: int
        The dimension of the space minus 1.
    
    Outputs
    -------
    res: float 1-D ndarray with (n+1)**3 entries 
        Each entry is a residual.
    """    
    
    res = np.zeros((n+1)**3)
    augX = np.zeros(r*(n+1))
    augY = np.zeros(r*(n+1))
    augZ = np.zeros(r*(n+1))
    
    #The program constructs the augmented vectors by considering the 0-th entries, which are equal to 1.
    for l in prange(0,r):
        augX[l*(n+1)] = 1
        augX[l*(n+1)+1 : l*(n+1) + n+1] = X[l*n : l*n + n]
        augY[l*(n+1)] = 1
        augY[l*(n+1)+1 : l*(n+1) + n+1] = Y[l*n : l*n + n]
        augZ[l*(n+1)] = 1
        augZ[l*(n+1)+1 : l*(n+1) + n+1] = Z[l*n : l*n + n]
        
    #Construction of the vector r = (r_{000}, r_{001}, ..., r_{nnn}).
    for i in prange(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                res[(n+1)**2*i + (n+1)*j + k] = residuals_entries(T,Lambda,augX,augY,augZ,r,n,i,j,k)
                
    return res


@njit(nogil=True,cache=True)
def residuals_entries(T,Lambda,augX,augY,augZ,r,n,i,j,k):
    """Computation of each individual residual in the function residuals."""
    
    s = 0
    for l in range(0,r):
        s += Lambda[l]*augX[l*(n+1)+i]*augY[l*(n+1)+j]*augZ[l*(n+1)+k]
        
    res_entry = T[i,j,k] - s
        
    return res_entry


@njit(nogil=True,parallel=True)
def derivative_residuals(Lambda,X,Y,Z,r,n):
    """
    Computation of the nonzero entries of the Jacobian matrix Dres of the residuals 
    map at a particular point (Lambda,X,Y,Z). The matrix Dres is sparse, and that is
    why we only keep its nonzero entries. This matrix is computed several times
    during the program and since the coordinates corresponding to the nonzero 
    entries never changes, we compute them in another function which is called just 
    once.
    
    Inputs
    ------
    Lambda: float 1-D ndarray with r entries
    X: float 1-D ndarray with r*n entries
    Y: float 1-D array with r*n entries
    Z: float 1-D ndarray with r*n entries
    r: int. 
        The rank of the desired approximating tensor.
    n: int. 
        The dimension of the space minus 1.
    
    Outputs
    -------
    data: float 1-D ndarray 
        The nonzero entries of Dres.
    """    
    
    data = np.zeros(4*(n+1)**3*r, dtype = np.float64)
    augX = np.zeros(r*(n+1))
    augY = np.zeros(r*(n+1))
    augZ = np.zeros(r*(n+1))
    s = 0
        
    #The program constructs the augmented vectors by considering the 0-th entries, which are equal to 1.
    for l in prange(0,r):
        augX[l*(n+1)] = 1
        augX[l*(n+1)+1 : l*(n+1) + n+1] = X[l*n : l*n + n]
        augY[l*(n+1)] = 1
        augY[l*(n+1)+1 : l*(n+1) + n+1] = Y[l*n : l*n + n]
        augZ[l*(n+1)] = 1
        augZ[l*(n+1)+1 : l*(n+1) + n+1] = Z[l*n : l*n + n]
    
    #Computation of all entries of Dres.
    for i in range(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                for l in range(0,r):
                    #Partial derivative with respect to Lambda.
                    data[s] = -augX[l*(n+1) + i]*augY[l*(n+1) + j]*augZ[l*(n+1) + k]
                    s = s+1
                    #Partial derivative with respect to X.
                    if i != 0:
                        data[s] = -Lambda[l]*augY[l*(n+1) + j]*augZ[l*(n+1) + k]
                        s = s+1
                    #Partial derivative with respect to Y.
                    if j != 0:
                        data[s] = -Lambda[l]*augX[l*(n+1) + i]*augZ[l*(n+1) + k]
                        s = s+1
                    #Partial derivative with respect to Z.
                    if k != 0:
                        data[s] = -Lambda[l]*augX[l*(n+1) + i]*augY[l*(n+1) + j]
                        s = s+1    
    data = data[0:s]
    
    return data


@njit(nogil=True,cache=True)
def initialize(r,n):
    """
    Initialization of the matrix Dres in sparse format, i.e., a triple (data,row,col) 
    such that data is a 1-D containing the nonzero values of Dres, row is a 1-D ndarray
    containing the corresponding rows index of the elements in data and col is a 1-D
    ndarray containing the corresponding columns index of the elements in data.
    All initial values of data are equal to one. This function doesn't compute any
    actual Jacobian matrix, but only initializes its sparse structure for later.
    
    Inputs
    ------
    r: int
        The rank of the desired approximating tensor.
    n: int 
        The dimension of the space minus 1.
    
    Outputs
    -------
    data: float 1-D ndarray of ones
    row: int 1-D ndarray
    col: int 1-D ndarray
    """   
    
    row = np.zeros(4*(n+1)**3*r, dtype = np.int64)
    col = np.zeros(4*(n+1)**3*r, dtype = np.int64)
    data = np.zeros(4*(n+1)**3*r, dtype = np.float64)
    s = 0
        
    for i in range(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                for l in range(0,r):
                    #Partial derivative with respect to Lambda.
                    row[s] = (n+1)**2*i + (n+1)*j + k
                    col[s] = l
                    data[s] = 1
                    s = s+1
                    #Partial derivative with respect to X.
                    if i != 0:
                        row[s] = (n+1)**2*i + (n+1)*j + k
                        col[s] = r + l*n + i-1
                        data[s] = 1
                        s = s+1
                    #Partial derivative with respect to Y.
                    if j != 0:
                        row[s] = (n+1)**2*i + (n+1)*j + k
                        col[s] = r + r*n + l*n + j-1
                        data[s] = 1
                        s = s+1
                    #Partial derivative with respect to Z.
                    if k != 0:
                        row[s] = (n+1)**2*i + (n+1)*j + k
                        col[s] = r + 2*r*n + l*n + k-1
                        data[s] = 1
                        s = s+1    
    row = row[0:s]
    col = col[0:s]
    data = data[0:s]
    
    return(data,row,col)


@njit(nogil=True,parallel=True)
def point2tens(x,r,n):
    """
    Let x = [Lambda,X,Y,Z], where X,Y,Z are described as in the function residual,
    i.e., they are 1-D dnarrays with r*n entries each. This function complete these
    ndarrays by putting the additional ones and then constructs the 3-D tensor S
    associated.

    Inputs
    ------
    x: float 1-D ndarray with r+3*r*n entries
    r: int 
        The rank of the desired approximating tensor.
    n: int 
        The dimension of the space minus 1.
    
    Outputs
    -------
    S: float 3-D ndarray
    """   
    
    S = np.zeros((n+1, n+1, n+1))
    #The first entries of X,Y,Z are set equal to one. 
    X = np.ones(r*(n+1))
    Y = np.ones(r*(n+1))
    Z = np.ones(r*(n+1))
    Lambda = x[0:r]
    
    for l in prange(0,r):
        X[l*(n+1) + 1:(l+1)*(n+1)] = x[r + l*n:r + (l+1)*n]
        Y[l*(n+1) + 1:(l+1)*(n+1)] = x[r + r*n + l*n:r + r*n + (l+1)*n]
        Z[l*(n+1) + 1:(l+1)*(n+1)] = x[r + 2*r*n + l*n:r + 2*r*n + (l+1)*n]
    
    for i in prange(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):                
                S[i,j,k] = S_entries(Lambda,X,Y,Z,r,n,i,j,k)
         
    return S


@njit(nogil=True,cache=True)
def S_entries(Lambda,X,Y,Z,r,n,i,j,k):
    """Computation of each individual entry of S in the function point2tens."""
    
    s = 0
    for l in range(0,r):
        s += Lambda[l]*X[l*(n+1)+i]*Y[l*(n+1)+j]*Z[l*(n+1)+k]
            
    S_entry = s
    
    return S_entry


def gauss_newton(T,Lambda,X,Y,Z,r,n,maxit=500,tol=10**(-3)):
    """
    Starting at x = [Lambda,X,Y,Z], this function uses the Damped Gauss-Newton
    method to compute an approximation of T with rank r. The result is given in
    format both classical formats: as coordinates and as components to form the CPD.
    
    The Damped Gauss-Newton method is iterative, updating the point x at each iteration.
    The last computed x is of the form x = [Lambda,X,Y,Z], and from these we have
    the components to form the CPD of S (the approximating tensor). This program also
    gives some additional information such as the size of the steps (distance 
    between each x computed), the errors (distance between T and S at each iteration)
    and the path of solutions (the points x computed at each iteration are saved).

    Inputs
    ------
    T: float 3-D ndarray
    Lambda: Float 1-D ndarray with r entries
    X: float 1-D ndarray with r*n entries
    Y: float 1-D ndarray with r*n entries
    Z: float 1-D ndarray with r*n entries
    r: int 
        The rank of the desired approximating tensor.
    n: int 
        The dimension of the space minus 1.
    maxit: int
        Number of maximum iterations permitted. By default this function makes at
    most 500 iterations.
    tol: float
        Tolerance criterium to stop the iteration proccess. Let S^(k) be the approximating 
    tensor computed at the k-th iteration an x^(k) be the point computed at the k-th 
    iteration. If we have norm(T-S^(k))/norm(T) < tol or norm(x^(k+1) - x^(k)) < tol, then 
    the program stops. By default we have tol = 10**(-3).
    
    Outputs
    -------
    x: float 1-D ndarray with r+3*r*n entries 
        Each entry represents the components of the approximating tensor in the CPD form.
    S: float 3-D ndarray with (n+1)**3 entries 
        Each entry represents the coordinates of the approximating tensor in coordinate form.
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration. 
    xpath: float 2-D ndarray 
        Points computed at each iteration. The k-th row represents the point computed at the 
    k-th iteration. 
    """  
        
    S = np.zeros((n+1,n+1,n+1))
    x = np.concatenate((Lambda,X,Y,Z))
    step_sizes = np.zeros(maxit)
    errors = np.zeros(maxit)
    xpath = np.zeros((maxit,r+3*r*n))
    error = np.inf
    Tsize = np.linalg.norm(T)
    #Initialize the row and column indexes for sparse matrix creation.
    [data,row,col] = initialize(r,n)
    Dres = sparse.csr_matrix((data, (row, col)), shape=((n+1)**3,r+3*r*n))
    #u is the damping parameter.
    u = 1
    atol = 10**(-5)
    btol = 10**(-3)
    old_residualnorm = 0
    
    #Gauss-Newton iterations starting at x.
    for it in range(0,maxit):  
        #Computation of r(x) and Dres(x).
        res = residuals(T,Lambda,X,Y,Z,r,n)
        data = derivative_residuals(Lambda,X,Y,Z,r,n)
        Dres[row,col] = data
        
        #Sets the old values to compare with the new ones.
        old_x = x
        old_error = error
        
        #Computation of the Gauss-Newton iteration formula to obtain the new point x.
        #The vector a is the solution of min_y |Ay - b|, with A = Dres(x) and b = -res(x). 
        [y,istop,itn,residualnorm,auxnorm,Dres_norm,estimate_Dres_cond,ynorm] = ssl.lsmr(Dres,-res,u,atol,btol)
        x = x + y
        
        #Computation of the respective tensor S associated to x and its error.
        S = point2tens(x,r,n)
        error = np.linalg.norm(T-S)
                
        #Update the damping parameter. 
        g = 2*(old_error - error)/(old_residualnorm - residualnorm)
        if g < 0.25:
            u = 2*u
        elif g > 0.75:
            u = u/3
        old_residualnorm = residualnorm
        
        #Update the arrays with information about the iteration.
        step_sizes[it] = np.linalg.norm(x - old_x)   
        errors[it] = error
        xpath[it,:] = x
        #After 10 iterations, the program starts to verify if the size of the current step or the difference between the errors are smaller than tol.
        if it >= 10:
            errors_diff = 1/Tsize*np.abs(errors[it] - errors[it-1])
            if step_sizes[it] < tol or errors_diff < tol:
                break
        #Update the vectors L,X,Y,Z for the next iteration.
        Lambda = x[0:r]
        X = x[r:r+r*n]
        Y = x[r+r*n:r+2*r*n]
        Z = x[r+2*r*n:r+3*r*n]
        
    step_sizes = step_sizes[0:it+1]
    errors = errors[0:it+1]
    xpath = xpath[0:it+1,:]

    return(x,S,step_sizes,errors,xpath)


def gauss_newton_timing(T,Lambda,X,Y,Z,r,n,maxit=500,tol=10**(-3)):
    """
    This function does the same thing as the `gauss_newton` function, but with the 
    difference that it measures the computation time of several parts of the
    algorithm.

    Inputs
    ------
    T: float 3-D ndarray
    Lambda: Float 1-D ndarray with r entries
    X: float 1-D ndarray with r*n entries
    Y: float 1-D ndarray with r*n entries
    Z: float 1-D ndarray with r*n entries
    r: int 
        The rank of the desired approximating tensor.
    n: int 
        The dimension of the space minus 1.
    maxit: int
        Number of maximum iterations permitted. By default this function makes at
    most 500 iterations.
    tol: float
        Tolerance criterium to stop the iteration proccess. Let S^(k) be the approximating 
    tensor computed at the k-th iteration an x^(k) be the point computed at the k-th 
    iteration. If we have norm(T-S^(k))/norm(T) < tol or norm(x^(k+1) - x^(k)) < tol, then 
    the program stops. By default we have tol = 10**(-3).
    
    Outputs
    -------
    x: float 1-D ndarray with r+3*r*n entries 
        Each entry represents the components of the approximating tensor in the CPD form.
    S: float 3-D ndarray with (n+1)**3 entries 
        Each entry represents the coordinates of the approximating tensor in coordinate form.
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration. 
    xpath: float 2-D ndarray 
        Points computed at each iteration. The k-th row represents the point computed at the 
    k-th iteration. 
    sparse_time: float 1-D ndarray 
        The k-th entry is the computation time spent to update the sparse matrix Dres at the 
    k-th iteration.
    gauss_newton_time: float 1-D ndarray 
        The k-th entry is the computation time spent to compute the lsmr algorithm at the k-th 
    iteration. We call it gauss_newton because this is the principal part of the Damped 
    Gauss-Newton method.
    rest_time: float 1-D ndarray 
        The k-th entry is the computation time spent in all the other parts of the k-th iteration.
    """ 
    
    S = np.zeros((n+1,n+1,n+1))
    x = np.concatenate((Lambda,X,Y,Z))
    step_sizes = np.zeros(maxit)
    errors = np.zeros(maxit)
    xpath = np.zeros((maxit,r+3*r*n))
    error = np.inf
    Tsize = np.linalg.norm(T)
    #Initialize the row and column indexes for sparse matrix creation.
    [data,row,col] = initialize(r,n)
    Dres = sparse.csr_matrix((data, (row, col)), shape=((n+1)**3,r+3*r*n))
    #u is the damping parameter.
    u = 1
    atol = 10**(-5)
    btol = 10**(-3)
    old_residualnorm = 0
    
    sparse_time = np.zeros(maxit)
    gauss_newton_time = np.zeros(maxit)
    rest_time = np.zeros(maxit)
    
    #Gauss-Newton iterations starting at x.
    for it in range(0,maxit):  
        #Computation of r(x) and Dres(x).
        start = time.time()
        res = residuals(T,Lambda,X,Y,Z,r,n)
        data = derivative_residuals(Lambda,X,Y,Z,r,n)
        rest_time[it] = time.time() - start
        
        #Convert the matrix to a scipy sparse matrix.
        start = time.time()
        Dres[row,col] = data
        sparse_time[it] = time.time() - start
        
        #Sets the old values to compare with the new ones.
        start = time.time()
        old_x = x
        old_error = error
        rest_time[it] = rest_time[it] + time.time() - start
        
        start = time.time()
        #Computation of the Gauss-Newton iteration formula to obtain the new point x.
        #The vector a is the solution of min_y |Ay - b|, with A = Dr(x) and b = -res(x). 
        [y,istop,itn,residualnorm,auxnorm,Dres_norm,estimate_Dres_cond,ynorm] = ssl.lsmr(Dres,-res,u,atol,btol)
        x = x + y
        gauss_newton_time[it] = time.time() - start
        
        start = time.time()
        #Computation of the respective tensor S associated to x and its error.
        S = point2tens(x,r,n)
        error = np.linalg.norm(T-S)
                
        #Update the damping parameter. 
        g = 2*(old_error - error)/(old_residualnorm - residualnorm)
        if g < 0.25:
            u = 2*u
        elif g > 0.75:
            u = u/3
        old_residualnorm = residualnorm
        
        #Update the arrays with information about the iteration.
        step_sizes[it] = np.linalg.norm(x - old_x)   
                
        errors[it] = error
        xpath[it,:] = x
        #After 10 iterations, the program starts to verify if the size of the current step or the difference between the errors are smaller than tol.
        if it >= 10:
            errors_diff = 1/Tsize*np.abs(errors[it] - errors[it-1])
            if step_sizes[it] < tol or errors_diff < tol:
                break
        #Update the vectors L,X,Y,Z for the next iteration.
        Lambda = x[0:r]
        X = x[r:r+r*n]
        Y = x[r+r*n:r+2*r*n]
        Z = x[r+2*r*n:r+3*r*n]
        rest_time[it] = rest_time[it] + time.time() - start
        
    step_sizes = step_sizes[0:it+1]
    errors = errors[0:it+1]
    xpath = xpath[0:it+1,:]
    sparse_time = sparse_time[0:it+1]
    gauss_newton_time = gauss_newton_time[0:it+1]
    rest_time = rest_time[0:it+1]

    return(x,S,step_sizes,errors,xpath,sparse_time,gauss_newton_time,rest_time)


def low_rank(T,r,n,maxtrials=3,maxit=500,tol=10**(-3)):
    """
    This function searches for the best rank r approximation of T by making several
    calls to the gauss_newton function with random initial points. By defalt, the
    gauss_newton function is called 3 times and the best result is saved. The user 
    may choose the number of trials the program makes with the parameter maxtrials. 
    Also, the user may choose the maximum number of iterations at each trials. Finally, 
    the user may define the tolerance value to stop the iteration process. The parameter 
    tol is passed to each Gauss_Newton trial, we also use this parameter to stop the 
    program when |T-S|/|T| < tol or the improvement of |T-S|/|T| in some trial is less 
    than tol. 

    The first outputs of this function are essentially the best output of all functions 
    gauss_newton computed. The program returns the arrays $(\Lambda,X,Y,Z)$, including 
    the 0th entries (equal to 1). Look at the desciption of the function `x2CPD` for more 
    details. The latter outputs are general information about all the trials. These 
    informations are the following:

    * The total time spent in each trial.

    * The number of steps used in each trial.

    * The relative error |T-S|/|T| obtained in each trial.

    Inputs
    ------
    T: float 3-D ndarray
    r: int 
        The rank of the desired approximating tensor.
    n: int 
        The dimension of the space minus 1.
    maxtrials: int 
        Number of maximum number of times the program calls the gauss_newton function.
    maxit: int
        Number of maximum iterations permitted in the gauss_newton function. By default 
    this function makes at most 500 iterations.
    tol: float
        Tolerance criterium to stop the iteration proccess. Let S^(k) be the approximating 
    tensor computed at the k-th iteration an x^(k) be the point computed at the k-th 
    iteration. If we have norm(T-S^(k))/norm(T) < tol or norm(x^(k+1) - x^(k)) < tol, then 
    the program stops. By default we have tol = 10**(-3).
    
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray with r*(n+1) entries
    Y: float 2-D ndarray with r*(n+1) entries
    Z: float 2-D ndarray with r*(n+1) entries
    S: float 3-D ndarray
    error: float
        The value |T-S|, which is the error of the best trials. 
    step_sizes: float 1-D ndarray 
        Distance between the computed points at each iteration of the best trial.
    errors: float 1-D ndarray 
        Error of the computed approximating tensor at each iteration of the best trial.
    xpath: float 2-D ndarray 
        Points computed at each iteration. The k-th row represents the point computed at the 
    k-th iteration of the best trial. 
    times: float 1-D ndarray
        The total time spent in each trial. The i-th entry is the computation time of the
    i-th trial.
    steps: int 1-D ndarray
        The number of steps used in each trial. The i-th entry is the number of steps (number 
    of iterations) of the i-th trial.
    rel_errors: float 1-D ndarray 
        The relative error |T-S|/|T| obtained in each trial. The i-th entry is the relative 
    error of the approximation of the i-th trial.
    """ 
    
    times = np.zeros(maxtrials)
    steps = np.zeros(maxtrials)
    rel_errors = np.zeros(maxtrials)
    best_error = np.inf
    
    #Computation of the Frobenius norm of T.
    Tsize = np.linalg.norm(T)
    
    #At each trial the program generates a random starting point (L,X,Y,Z) to apply the Gauss-Newton method.
    for trial in range(0,maxtrials):
        Lambda = np.random.randn(r)
        X = np.random.randn(r*n)
        Y = np.random.randn(r*n)
        Z = np.random.randn(r*n)
        
        #Computation of one Gauss-Newton method starting at (L,X,Y,Z).
        start = time.time()
        [x,S,step_sizes,errors,xpath] = gauss_newton(T,Lambda,X,Y,Z,r,n,maxit,tol) 
        
        #Update the vectors with general information.
        times[trial] = time.time() - start
        steps[trial] = step_sizes.shape[0]
        rel_errors[trial] = 1/Tsize*errors[-1]
                
        #We use the subroutine "pocket" to save the best tensor S computed  so far.    
        error = errors[-1]
        if error < best_error:
            best_x = x
            best_S = S
            old_best_error = best_error
            best_error = error
            best_step_sizes = step_sizes
            best_errors = errors
            best_xpath = xpath
            #The search for better solutions stops when the best relative error is small enough or the improvement is irrelevant.
            if trial > 1:
                if best_error/Tsize < tol or np.abs(best_error - old_best_error) < Tsize*tol :
                    break
                
    #After everything, we rename all the information related to the best S, for convenience.
    [Lambda,X,Y,Z] = x2CPD(best_x,r,n)
    error = best_error
    step_sizes = best_step_sizes
    errors = best_errors
    xpath = best_xpath
    times = times[0:trial+1]
    steps = steps[0:trial+1]
    rel_errors = rel_errors[0:trial+1]

    return(Lambda,X,Y,Z,S,error,step_sizes,errors,xpath,times,steps,rel_errors)


@njit(nogil=True,parallel=True)
def x2CPD(x,r,n):
    """
    Given the point x = (x_1, \ldots, x_{r+3rn}), this function breaks it in parts, 
    in order to form the CPD of S. This program return the arrays (let ^T be the transpose)
    Lambda = [Lambda_1,...,Lambda_r]^T,
    X = [X_1,...,X_r]^T,
    Y = [Y_1,...,Y_r]^T,
    Z = [Z_1,...,Z_r]^T
    so we have that
    S = Lambda_1*X_1⊗Y_1⊗Z_1 + ... + Lambda_r*X_r⊗Y_r⊗Z_r, where
    X_l = (1, X_{l_1}, ..., X_{l_n}),
    Y_l = (1, Y_{l_1}, ..., Y_{l_n}),
    Z_l = (1, Z_{l_1}, ..., Z_{l_n}).
    
    Inputs
    ------
    x: float 1-D ndarray
    r: int 
        The rank of the approximating tensor.
    n: int 
        The dimension of the space minus 1.
        
    Outputs
    -------
    Lambda: float 1-D ndarray with r entries
    X: float 2-D ndarray with r*(n+1) entries
    Y: float 2-D ndarray with r*(n+1) entries
    Z: float 2-D ndarray with r*(n+1) entries
    """
    
    Lambda = np.zeros(r)
    X = np.ones((r,n+1))
    Y = np.ones((r,n+1))
    Z = np.ones((r,n+1))
    
    Lambda = x[0:r]
    for l in prange(0,r):
        X[l,:] = x[r + l*n:r + (l+1)*n]
        Y[l,:] = x[r + r*n + l*n:r + r*n + (l+1)*n]
        Z[l,:] = x[r + 2*r*n + l*n:r + 2*r*n + (l+1)*n]
        
    return(Lambda,X,Y,Z)


def rank(T):
    """
    This function computes several approximations of T for r = 1 \ldots n^2. We use 
    these computations to determine the (most probable) rank of T. The function also 
    returns an array `errors_per_rank` with the relative errors for the rank varying 
    from 1 to r+1, where r is the computed rank of T. It is relevant to say that the 
    value r computed can also be the `border rank` of T, not the actual rank. 

    The idea is that the minimum of \|T-S\|, for each rank r, stabilizes when S has 
    the same rank as T. This function also plots the graph of the errors so the user 
    are able to visualize the moment when the error stabilizes.
    
    Inputs
    ------
    T: float 3-D ndarray
    
    Outputs
    -------
    final_rank: int
        The computed rank of T.
    errors_per_rank: float 1-D ndarray
        The error |T-S| computed for each rank.    
    """
    
    #R is an upper bound for the rank.
    n = T.shape[0]-1
    R = n**2
    errors_per_rank = np.zeros(R)
    Tsize = np.linalg.norm(T)
    for r in range(1,R):
        maxit = 100
        tol = 10**(-3)
        Lambda = np.random.randn(r)
        X = np.random.randn(r*n)
        Y = np.random.randn(r*n)
        Z = np.random.randn(r*n)
        [x,S,step_sizes,errors,xpath] = gauss_newton(T,Lambda,X,Y,Z,r,n,maxit,tol)
        errors_per_rank[r-1] = errors[-1]/Tsize
        if r > 1:
            #Verification of the stabilization condition.
            if np.abs(errors_per_rank[r-1] - errors_per_rank[r-2]) < tol:
                break
     
    final_rank = r-1
    errors_per_rank = errors_per_rank[0:r] 
    
    print('R(T) =',r-1)
    print('|T-S|/|T| =',errors_per_rank[-2])
    plt.plot(range(1,r+1),np.log10(errors_per_rank))
    plt.plot(r-1,np.log10(errors_per_rank[-2]),marker = 'o',color = 'k')
    plt.title('Rank trials')
    plt.xlabel('r')
    plt.ylabel('$log10 \|T - S\|/|T|$')
    plt.grid()
    plt.show()
            
    return(final_rank,errors_per_rank)

