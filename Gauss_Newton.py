
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg as ssl
from numba import jit, njit, prange


# Consider the problem of approximating a tensor $T \in \mathbb{R}^{n+1} \otimes \mathbb{R}^{n+1} \otimes \mathbb{R}^{n+1}$ by a tensor of rank $r$ given by
# $$S = \sum_{\ell=1}^r \Lambda_\ell \cdot X_{\ell} \otimes Y_{\ell} \otimes Z_{\ell},$$
# where 
# $$X_{\ell} = (1, X_{\ell_1}, \ldots, X_{\ell_n}),$$
# $$Y_{\ell} = (1, Y_{\ell_1}, \ldots, Y_{\ell_n}),$$
# $$Z_{\ell} = (1, Z_{\ell_1}, \ldots, Z_{\ell_n}).$$
# 
# We do this by minimizing the error function
# $$\textbf{E}(\Lambda,X,Y,Z) = \frac{1}{2}\|T - S\|^2 = \frac{1}{2} \sum_{i,j,k=0}^n \left( T_{ijk} - \sum_{\ell=1}^r \Lambda_\ell \cdot X_{\ell_i} Y_{\ell_j} Z_{\ell_j} \right)^2 = \frac{1}{2} \sum_{i,j,k=0}^n r_{ijk}^2(\Lambda, X,Y,Z) = \frac{1}{2} \|\textbf{r}(\Lambda,X,Y,Z)\|^2,$$
# where
# $$\Lambda = (\Lambda_1, \ldots, \Lambda_r),$$
# $$X = (X_1, \ldots, X_r),$$
# $$Y = (Y_1, \ldots, Y_r),$$
# $$Z = (Z_1, \ldots, Z_r),$$
# and $\textbf{r} = (r_{000}, r_{001}, \ldots, r_{nnn})$ is the function of the residuals.
# 
# In the python function called *residuals* the program constructs $\textbf{r}(\Lambda,X,Y,Z)$ for a given $(\Lambda,X,Y,Z)$.

# In[2]:


@njit(nogil=True,parallel=True)
def residuals(T,L,X,Y,Z,r,n):
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
                s = 0
                for l in range(0,r):
                    s += L[l]*augX[l*(n+1)+i]*augY[l*(n+1)+j]*augZ[l*(n+1)+k]
                res[(n+1)**2*i + (n+1)*j + k] = T[i,j,k] - s
                
    return res


# In the python function *derivative_residuals* the program constructs the Jacobian matrix of $\textbf{r}$ at $(\Lambda,X,Y,Z)$.

# In[3]:


@njit(nogil=True,parallel=True)
def derivative_residuals(L,X,Y,Z,r,n):
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
    
    #Computation of all entries of Dr.
    for i in range(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                for l in range(0,r):
                    #Partial derivative with respect to Lambda.
                    data[s] = -augX[l*(n+1) + i]*augY[l*(n+1) + j]*augZ[l*(n+1) + k]
                    s = s+1
                    #Partial derivative with respect to X.
                    if i != 0:
                        data[s] = -L[l]*augY[l*(n+1) + j]*augZ[l*(n+1) + k]
                        s = s+1
                    #Partial derivative with respect to Y.
                    if j != 0:
                        data[s] = -L[l]*augX[l*(n+1) + i]*augZ[l*(n+1) + k]
                        s = s+1
                    #Partial derivative with respect to Z.
                    if k != 0:
                        data[s] = -L[l]*augX[l*(n+1) + i]*augY[l*(n+1) + j]
                        s = s+1
    
    data = data[0:s]
    
    return data


# The function *initialize* creates the arrays *data,row,col*, which are necessary for working with the sparse matrices **Dr**. Since the sparse structure of these matrices is always the same, the arrays *row,col* only need to be initialized one time.

# In[4]:


@njit(nogil=True,cache=True)
def initialize(r,n):
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


# The python function *point2tens* constructs the tensor $S = \sum_{\ell=1}^r \Lambda_\ell \cdot X_\ell \otimes Y_\ell \otimes Z_\ell$ from a point $x = (\Lambda,X,Y,Z)$.

# In[5]:


@njit(nogil=True,parallel=True)
def point2tens(x,r,n):
    S = np.zeros((n+1, n+1, n+1))
    #The first entries of X,Y,Z are set equal to one. 
    X = np.ones(r*(n+1))
    Y = np.ones(r*(n+1))
    Z = np.ones(r*(n+1))
    L = x[0:r]
    
    for l in prange(0,r):
        X[l*(n+1) + 1:(l+1)*(n+1)] = x[r + l*n:r + (l+1)*n]
        Y[l*(n+1) + 1:(l+1)*(n+1)] = x[r + r*n + l*n:r + r*n + (l+1)*n]
        Z[l*(n+1) + 1:(l+1)*(n+1)] = x[r + 2*r*n + l*n:r + 2*r*n + (l+1)*n]
    
    for i in prange(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                s = 0
                for l in range(0,r):
                    s += L[l]*X[l*(n+1)+i]*Y[l*(n+1)+j]*Z[l*(n+1)+k]
                S[i,j,k] = s
         
    return S


# For a given initial point $x^{(0)} = (\Lambda^{(0)},X^{(0)},Y^{(0)},Z^{(0)})$, the python function *gauss_newton* tries to minimize the residual function using the damped Gauss-Newton method. The user may choose the maximum number of iterations and the tolerance value to stop the iteration process. The parameter *tol* makes the iteration stops when $\|T-S\|/\|T\| < tol$ or $\|x^{(k+1)} - x^{(k)}\| < tol$. 
# 
# This function returns the approximating tensor $S$, the obtained error $\|T-S\|$, and the following additional information:
# 
# $\bullet$ The final point $x = (\Lambda,X,Y,Z) \in \mathbb{R}^{r+3rn}$ computed and used to construct the approximating tensor $S$.
# 
# $\bullet$ An array $[\|x^{(1)} - x^{(0)}\|, \|x^{(2)} - x^{(1)}\|, \ldots ]$ with the distance between the points in each iterarion.
# 
# $\bullet$ An array $[\kappa\left( D\textbf{r}(x^{(0)})^T D\textbf{r}(x^{(0)}) \right), \kappa\left( D\textbf{r}(x^{(1)})^T D\textbf{r}(x^{(1)}) \right), \ldots]$ with the condition numbers of $D\textbf{r}(x^{(k)})^T D\textbf{r}(x^{(k)})$ in each iteration.
# 
# $\bullet$ An array $[\|T-S^{(0)}\|, \|T-S^{(1)}\|, \ldots ]$ with the absolute errors in each iteration.
# 
# $\bullet$ An array $[x^{(0)}, x^{(1)}, \ldots]$ with the path of the points computed in each iteration.

# In[12]:


def gauss_newton(T,L,X,Y,Z,r,n,maxit=500,tol=10**(-3)):
    S = np.zeros((n+1,n+1,n+1))
    x = np.concatenate((L,X,Y,Z))
    step_sizes = np.zeros(maxit)
    errors = np.zeros(maxit)
    xpath = np.zeros((maxit,r+3*r*n))
    error = np.inf
    Tsize = np.linalg.norm(T)
    #Initialize the row and column indexes for sparse matrix creation.
    [data,row,col] = initialize(r,n)
    Dr = sparse.csr_matrix((data, (row, col)), shape=((n+1)**3,r+3*r*n))
    #u is the damping parameter.
    u = 1
    atol = 10**(-5)
    btol = 10**(-3)
    old_residualnorm = 0
    
    #Gauss-Newton iterations starting at x.
    for it in range(0,maxit):  
        #Computation of r(x) and Dr(x).
        res = residuals(T,L,X,Y,Z,r,n)
        data = derivative_residuals(L,X,Y,Z,r,n)
        Dr[row,col] = data
        
        #Sets the old values to compare with the new ones.
        old_x = x
        old_error = error
        
        #Computation of the Gauss-Newton iteration formula to obtain the new point x.
        #The vector a is the solution of min_y |Ay - b|, with A = Dr(x) and b = -res(x). 
        [y,istop,itn,residualnorm,auxnorm,Drnorm,estimateDrcond,ynorm] = ssl.lsmr(Dr,-res,u,atol,btol)
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
        L = x[0:r]
        X = x[r:r+r*n]
        Y = x[r+r*n:r+2*r*n]
        Z = x[r+2*r*n:r+3*r*n]
        
    step_sizes = step_sizes[0:it+1]
    errors = errors[0:it+1]
    xpath = xpath[0:it+1,:]

    return(x,S,step_sizes,errors,xpath)


# This function does the same thing as *gauss_newton*, but with the difference that it measures the times of several parts in the algorithm. 

# In[7]:


def gauss_newton_timing(T,L,X,Y,Z,r,n,maxit=500,tol=10**(-3)):
    S = np.zeros((n+1,n+1,n+1))
    x = np.concatenate((L,X,Y,Z))
    step_sizes = np.zeros(maxit)
    errors = np.zeros(maxit)
    xpath = np.zeros((maxit,r+3*r*n))
    error = np.inf
    Tsize = np.linalg.norm(T)
    #Initialize the row and column indexes for sparse matrix creation.
    [data,row,col] = initialize(r,n)
    Dr = sparse.csr_matrix((data, (row, col)), shape=((n+1)**3,r+3*r*n))
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
        #Computation of r(x) and Dr(x).
        start = time.time()
        res = residuals(T,L,X,Y,Z,r,n)
        data = derivative_residuals(L,X,Y,Z,r,n)
        rest_time[it] = time.time() - start
        
        #Convert the matrix to a scipy sparse matrix.
        start = time.time()
        Dr[row,col] = data
        sparse_time[it] = time.time() - start
        
        #Sets the old values to compare with the new ones.
        start = time.time()
        old_x = x
        old_error = error
        rest_time[it] = rest_time[it] + time.time() - start
        
        start = time.time()
        #Computation of the Gauss-Newton iteration formula to obtain the new point x.
        #The vector a is the solution of min_y |Ay - b|, with A = Dr(x) and b = -res(x). 
        [y,istop,itn,residualnorm,auxnorm,Drnorm,estimateDrcond,ynorm] = ssl.lsmr(Dr,-res,u,atol,btol)
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
        L = x[0:r]
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


# The python function *low_rank* tries to minimize the residual function calling several times the Gauss-Newton method. The user may choose the number of trials the program makes with the parameter *maxtrials*. Also, the user may choose the maximum number of iterations at each trials. Finally, the user may define the tolerance value to stop the iteration process. The parameter *tol* is passed to each Gauss_Newton trial, we also use this parameter to stop the program when $\|T-S\|/\|T\| < tol$ or the improvement of $\|T-S\|/\|T\|$ in some trial is less than *tol*. 
# 
# The first outputs of this function are essentially the best output of all functions *gauss_newton* computed. With respect to the point $x$, the program also returns the vectors $(\Lambda,X,Y,Z)$, including the 0th entries (equal to 1). This can be useful if the user wants to use the CPD form of $S$. The latter outputs are general information about all the trials. These informations are the following:
# 
# $\bullet$ The total time spent in each trial.
# 
# $\bullet$ The number of steps used in each trial.
# 
# $\bullet$ The relative error $\|T-S\|/\|T\|$ obtained in each trial.

# In[8]:


def low_rank(T,r,n,maxtrials,maxit=500,tol=10**(-3)):
    times = np.zeros(maxtrials)
    steps = np.zeros(maxtrials)
    rel_errors = np.zeros(maxtrials)
    best_X = np.ones(r*(n+1))
    best_Y = np.ones(r*(n+1))
    best_Z = np.ones(r*(n+1))
    best_error = np.inf
    
    #Computation of the Frobenius norm of T.
    Tsize = np.linalg.norm(T.flatten())
    
    #At each trial the program generates a random starting point (L,X,Y,Z) to apply the Gauss-Newton method.
    for trial in range(0,maxtrials):
        L = np.random.randn(r)
        X = np.random.randn(r*n)
        Y = np.random.randn(r*n)
        Z = np.random.randn(r*n)
        
        #Computation of one Gauss-Newton method starting at (L,X,Y,Z).
        start = time.time()
        [x,S,step_sizes,errors,xpath] = gauss_newton(T,L,X,Y,Z,r,n,maxit,tol) 
        
        #Update the vectors with general information.
        times[trial] = time.time() - start
        steps[trial] = step_sizes.shape[0]
        rel_errors[trial] = 1/Tsize*errors[-1]
                
        #We use the subroutine "pocket" to save the best tensor S computed  so far.    
        error = errors[-1]
        if error < best_error:
            best_x = x
            best_L = x[0:r]
            for l in range(0,r):
                best_X[l*(n+1) + 1:(l+1)*(n+1)] = x[r + l*n:r + (l+1)*n]
                best_Y[l*(n+1) + 1:(l+1)*(n+1)] = x[r + r*n + l*n:r + r*n + (l+1)*n]
                best_Z[l*(n+1) + 1:(l+1)*(n+1)] = x[r + 2*r*n + l*n:r + 2*r*n + (l+1)*n]
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
    x = best_x
    L = best_L
    X = best_X
    Y = best_Y
    Z = best_Z
    S = best_S
    error = best_error
    step_sizes = best_step_sizes
    errors = best_errors
    xpath = best_xpath
    times = times[0:trial+1]
    steps = steps[0:trial+1]
    rel_errors = rel_errors[0:trial+1]

    return(x,L,X,Y,Z,S,error,step_sizes,errors,xpath,times,steps,rel_errors)


# This function computes several approximations of $T$ for $r = 1 \ldots n^2$. We use these computations to determine the (most probable) rank of $T$. The function also returns an array *rank_errors* with the relative errors for the rank varying from 1 to $r+1$, where $r$ is the computed rank of $T$. It is relevant to say that the value $r$ computed can also be the *border rank* of $T$, not the actual rank. 
# 
# The idea is that the minimum of $\|T-S\|$, for each rank $r$, stabilizes when $S$ has the same rank as $T$. This function also plots the graph of the errors so the user are able to visualize the moment when the error stabilize.

# In[9]:


def rank(T):
    #R is an upper bound for the rank.
    n = T.shape[0]-1
    R = n**2
    rank_errors = np.zeros(R)
    Tsize = np.linalg.norm(T)
    for r in range(1,R):
        maxit = 100
        L = np.random.randn(r)
        X = np.random.randn(r*n)
        Y = np.random.randn(r*n)
        Z = np.random.randn(r*n)
        [x,S,step_sizes,errors,xpath] = gauss_newton(T,L,X,Y,Z,r,n,maxit)
        rank_errors[r-1] = errors[-1]/Tsize
        if r > 1:
            #Verification of the stabilization condition.
            if np.abs(rank_errors[r-1] - rank_errors[r-2]) < tol:
                break
           
    rank_errors = rank_errors[0:r] 
    
    print('R(T) =',r-1)
    print('|T-S|/|T| =',rank_errors[-2])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.plot(range(1,r+1),np.log10(rank_errors))
    plt.plot(r-1,np.log10(rank_errors[-2]),marker = 'o',color = 'k')
    plt.title('Rank trials')
    plt.xlabel('r')
    plt.ylabel('$log10 \|T - S\|/|T|$')
    plt.show()
            
    return(r-1,rank_errors)

