"""
 Display Module
 
 This module contains function centered in computing and displaying general information about tensors and tensor spaces. Below we list all funtions presented in this module.
 
 - showtens
 
 - infotens
 
 - infospace
 
 - rank1_plot
 
 - rank1
"""


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import TensorFox as tf
import Auxiliar as aux


def showtens(T):
    """
    Let T = T_(i,j,k) be a tensor in coordinates. It is usual to consider that i is the
    row coordinate, j is the column coordinate and k is the section coordinate. But if 
    we just run the command print(T) we will see that numpy considers i as the section
    coordinate, j the row coordinate and k the column coordinate. This is not the usual
    way to consider tensors, and if we want to print T section by section (i.e., each
    frontal slice separately), this function does the job. 
    """
    
    for k in range(0,T.shape[2]):
        print(T[:,:,k])
        print()
        
    return


def infotens(T):
    """
    Given a tensor T, this function computes and shows several informations about this
    tensor. There are only print outputs. 
    Since this fucntion tries to estimate the rank of T, be aware that it may take a 
    long time to finish all computations. 
    """
    
    # Compute dimensions and norm of T.
    m, n, p = T.shape
    Tsize = np.linalg.norm(T)
    
    print('T is a tensor of dimensions',m,'x',n,'x',p)
    print()
    print('|T| =',Tsize)
    print()
    
    # Max and min entries of T.
    print('max(T) =',np.max(T))
    print()
    print('min(T) =',np.min(T))
    print()
    
    # Bounds on rank.
    R = min(m*n, m*p, n*p)
    print(1,'<= rank(T) <=',R)
    print()
    
    # Multilinear rank.
    S, multi_rank, U1, U2, U3, sigma1, sigma2, sigma3 = tf.hosvd(T)
    R1, R2, R3 = multi_rank
    print('multirank(T) =',R1,',',R2,',',R3)
    print()
    
    # Estimative of the rank of T.
    r, error_per_rank = tf.rank(T, display='none')
    print()
    
    # Condition number of T with respect to the rank R.
    condition_number = aux.cond(T,r) 
    print('cond(T) =',condition_number,' (condition number with respect to the rank',r,')')
    
    return


def infospace(m,n,p):
    """
    This function shows general information about the tensorial space
    R^m ⊗ R^n ⊗ R^p. At the moment we don't have too much to show. This
    is an on going work.
    """
    
    return


def rank1_plot(Lambda, X, Y, Z, r):
    """
    This function generates an image with the frontal sections of all rank one
    terms (in coordinates) of some CPD. It also saves the image in a file. The
    parameters may not fit all cases, so the user may prefer to change them
    manually.
    """

    rank1_sections = rank1(Lambda, X, Y, Z, r)
    
    plt.figure(figsize=(4*int(1+r/5),4*5))
    for l in range(0,r):
        ax = plt.subplot(int(1+r/5),5 ,l+1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(rank1_sections[:,:,l],cmap='Greys')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('rank1_sections.png')
    
    return


@njit(nogil=True, parallel=True)
def rank1(Lambda, X, Y, Z, r):
    """
    Compute each rank 1 term of the CPD given by Lambda, X, Y, Z. Them it
    converts these factors into a matrix, which is the frontal slice of the
    tensor in coordinates obtained by this rank 1 term. By doing this for all
    r terms, we have a tensor with r slices, each one cordesponding to a rank
    1 term of the original CPD.
    """

    # Obtain the dimensions of the tensor.
    m = X.shape[0]
    n = Y.shape[0]
    p = Z.shape[0]
    
    # Each frontal slice of rank1_sections is the coordinate representation of a
    # rank one term of the CPD given by (X,Y,Z)*Lambda.
    rank1_sections = np.zeros((m,n,r), dtype = np.float64)
    T_aux = np.zeros((m,n,p), dtype = np.float64)
   
    for l in prange(0,r):
        for i in range(0,m):
            for j in range(0,n):
                for k in range(0,p):
                    T_aux[i,j,k] = Lambda[l]*X[i,l]*Y[j,l]*Z[k,l]
        rank1_sections[:,:,l] = T_aux[:,:,l]
                
    return rank1_sections