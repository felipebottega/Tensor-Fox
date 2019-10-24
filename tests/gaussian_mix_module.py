import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import scipy
import time
import TensorFox as tfx


def gen_parameters(d, k):
    """
    d is the dimension of the data and k is the number os clusters.
    This function generates k vectors u1, ..., uk (the means of each)
    cluster. These vectors form an orthonormal set. It also generates
    k values 0 < wi < 1 such that w1 + ... + wk = 1. Each wi is the
    probability of a random sample to be in cluster i. Finally, this 
    function also generates a small variance sigma^2 randomly.
    """
    
    M = np.random.randn(d, k)
    u, s, vt = np.linalg.svd(M)
    u = u[:, :k] 
        
    if np.linalg.matrix_rank(M) < k:
        print('Matrix is not of full rank. Try again')
        return 0, 0, 0
    
    # Randomly choose k points in interval (0,1) and use these points
    # to generate the wi's.
    points = np.random.uniform(0, 1, k)
    points = np.sort(points)
    points[-1] = 1
    w = np.zeros(k)
    w[0] = points[0]
    
    if w[0] == 0:
        print('First wi is equal to 0. Try again')
        return 0, 0, 0
    
    for i in range(1,k-1):
        w[i] = points[i] - points[i-1]
    w[-1] =  points[-1] - points[-2]
    w = np.sort(w)
        
    sigma = min(1e-6 + np.random.uniform(0, 0.1), 1 - 1e-6)
    
    return u, w, sigma        


def gen_samples(u, w, sigma, m):
    """
    Generates m samples xj such that P[xj is in cluster i] = wi.
    Once xj is in cluster i, it is a random vector with mean ui 
    and covariance matrix sigma^2*I. 
    """
    
    d, k = u.shape
    data = np.zeros((d, m))
    samples_per_cluster = np.zeros(k, dtype = np.int64)
    
    # Generate number of samples in each cluster
    for i in range(0, k):
        samples_per_cluster[i] = int(w[i]*m) 
    if np.sum(samples_per_cluster) < m:
        samples_per_cluster[-1] = m - np.sum(samples_per_cluster) + samples_per_cluster[-1]
    
    # Generate samples for each cluster i
    cov = sigma**2 * np.identity(d)
    num_samples = 0
    for i in range(k):
        mean = u[:,i]
        data[:, num_samples:num_samples + samples_per_cluster[i]] = np.random.multivariate_normal(mean, cov, samples_per_cluster[i]).T        
        num_samples += samples_per_cluster[i]
                
    return samples_per_cluster, data


def empirical(u, w, sigma, data, k, display):
    """
    Generates empirical (synthetic in this case) data from the
    parameters given. The parameter display is 0 (to not display)
    or any other number (to display). The parameters u, w, sigma,
    data and k are described in the previous functions.
    """
    
    d, m = data.shape 
    I = np.identity(d)
    
    # COMPUTE REAL DATA
    
    if display != 0:
        print('Computing real parameters...')
        print()
    
    # E[x] (mean)
    M1 = np.zeros(d)
    for i in range(k):
        M1 += w[i]*u[:,i]
        
    # M2 = E[x*x^T] - sigma^2 * I    
    M2 = np.zeros((d, d))
    for i in range(k):
        M2 += w[i] * np.outer(u[:,i], u[:,i])
    
    # M3 
    M3 = np.zeros((d, d, d))
    for i in range(k):
        M3 += w[i] * tensprod(u[:,i], u[:,i], u[:,i])  
    
    # Covariance matrix
    cov = M2 + sigma**2 * np.identity(d) - np.outer(M1, M1)
    
    # COMPUTE EMPIRICAL DATA
    
    if display != 0:
        print('Computing empirical parameters...')
        print()
    
    # Empirical mean (M1)
    M1_approx = 1/m * np.sum(data, axis=1)
    if display != 0:
        print('|M1_approx - M1|/|M1| =', np.linalg.norm(M1_approx - M1)/np.linalg.norm(M1))
        print()
    
    # Empirical covariance
    cov_approx = np.zeros((d, d))
    for j in range(m):
        cov_approx += np.outer(data[:,j], data[:,j])
    cov_approx = 1/m * cov_approx - np.outer(M1_approx, M1_approx)
    if display != 0:
        print('|cov_approx - cov|/|cov| =', np.linalg.norm(cov_approx - cov)/np.linalg.norm(cov))
        print()
    
    # Empirical variance
    s = np.linalg.svd(cov_approx, compute_uv=False)
    sigma_approx = np.sqrt(np.min(s))
    if display != 0:
        print('|sigma_approx - sigma|/|sigma| =', np.linalg.norm(sigma_approx - sigma)/np.linalg.norm(sigma))
        print()
    
    # Empirical M2
    M2_approx = np.zeros((d, d))
    for j in range(m):
        M2_approx += np.outer(data[:,j], data[:,j])
    M2_approx = 1/m * M2_approx - sigma_approx**2 * np.identity(d)
    if display != 0:
        print('|M2_approx - M2|/|M2| =', np.linalg.norm(M2_approx - M2)/np.linalg.norm(M2))
        print()
    
    # Empirical skewness (third moment)
    skew_approx = compute_skewness(data, d, m)
    
    # Empirical M3
    S_approx = np.zeros((d, d, d))    
    for i in range(d): 
        aux1 = tensprod(M1_approx, I[i,:], I[i,:])
        aux2 = tensprod(I[i,:], M1_approx, I[i,:])
        aux3 = tensprod(I[i,:], I[i,:], M1_approx)
        aux = aux1 + aux2 + aux3
        S_approx += aux
    M3_approx = skew_approx - sigma_approx**2 * S_approx
    if display != 0:
        print('|M3_approx - M3|/|M3| =', np.linalg.norm(M3_approx - M3)/np.linalg.norm(M3))    
        print()
    
    return M3_approx


def learn(w, u, sigma, M3_approx, k, options, trials, display=False):
    """ 
    Compute latent variables. 
    """
    
    # Compute CPD of rank k of M3_approx
    best_w_quality = np.inf
    best_u_quality = np.inf
    for i in range(trials):
        factors, output = tfx.cpd(M3_approx, k, options) 
        Lambda, factors = tfx.cnv.normalize(factors)
        X, Y, Z = factors
        
        Lambda, X = fix_parameters(Lambda, X, k)
        w_quality, u_quality = test_quality(Lambda, X, u, w)
         
        if w_quality < best_w_quality and u_quality < best_u_quality:
            best_w_quality = w_quality
            best_u_quality = u_quality
            print('trial', i+1, '  ', np.round(best_w_quality, 6), '  ', np.round(best_u_quality, 6))
            best_w, best_u, best_output = Lambda.copy(), X.copy(), output
                        
    # Show final results of the CPD computation
    if display == True:
        print()
        print('Relative error of CPD =', best_output.rel_error)
        print('Accuracy of CPD =', np.round(best_output.accuracy, 6), '%')
        print('w_quality =', best_w_quality)
        print('u_quality =', best_u_quality)
    
    return best_w, best_u, best_w_quality, best_u_quality, best_output


def fix_parameters(w_approx, u_approx, k):
    """
    Sort w_approx and fix it so that its entries are positive with sum 
    equal to 1. u_approx is updated accordingly.
    """
    
    idx = np.argsort(w_approx)
    w_approx = w_approx[idx]
    u_approx = u_approx[:, idx]
    
    for i in range(k):
        if w_approx[i] < 0:
            w_approx[i] = -w_approx[i]
            u_approx[:,i] = -u_approx[:,i]
            
    w_sum = np.sum(w_approx)
    if w_sum != 1.0:
        w_approx[-1] = w_approx[-1] + 1 - w_sum
    
    return w_approx, u_approx


def test_quality(w_approx, u_approx, u, w):
    """
    Now it is time to check the quality of the solution. Ideally we want 
    to have Lambda = w and X = Y = Z = u.    
    """
    
    w_quality = np.linalg.norm(w_approx - w)/np.linalg.norm(w)
    u_quality = np.linalg.norm(u_approx - u)/np.linalg.norm(u)
    
    return w_quality, u_quality


@njit(nogil=True)
def compute_skewness(data, d, m):
    """
    Use numba to accelerate the computation of the empirical skewness. 
    """
    
    skew = np.zeros((d, d, d), dtype = np.float64)
    
    for l in range(m):
        skew += tensprod(data[:,l], data[:,l], data[:,l])
                    
    skew = 1/m * skew
    
    return skew


@njit(nogil=True, parallel=True)
def tensprod(x, y, z):
    """
    Computes the tensor product x ⊗ y ⊗ z, where x, y, z are vector of same 
    dimension d.
    """
    
    d = x.size
    T = np.zeros((d, d, d))
    
    for i in prange(d):
        for j in range(d):
            for k in range(d):
                T[i,j,k] = x[i]*y[j]*z[k]
                
    return T
