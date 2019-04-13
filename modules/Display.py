"""
 Display Module
 
 This module contains function centered in computing and displaying general information about tensors and tensor spaces. Below we list all funtions presented in this module.
 
 - showtens
 
 - infotens

 - adjust
 
 - rank1_plot
 
 - rank1

 - rank_progress

"""


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import TensorFox as tfx
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
    
    dims = T.shape
    L = len(dims)

    if L == 3:
        for k in range(T.shape[2]):
            print(T[:,:,k])
            print()
    else:
        print(T)
        
    return


def infotens(T):
    """
    Given a tensor T, this function computes and shows several informations about this
    tensor. There are only print outputs. 
    Since this fucntion tries to estimate the rank of T, be aware that it may take a 
    long time to finish all computations. 
    """
    
    # Compute dimensions and norm of T.
    dims = T.shape
    L = len(dims)
    Tsize = np.linalg.norm(T)
    
    print('T is a tensor of dimensions', dims)
    print()
    print('|T| =', Tsize)
    print()
    
    # Max and min entries of T.
    print('max(T) =', np.max(T))
    print()
    print('min(T) =', np.min(T))
    print()
    print('E[T] =', np.mean(T))
    print()
    print('E[|T|] =', np.mean(np.abs(T)))
    print()
    
    # Bounds on rank.
    sorted_dims = np.sort(np.array(dims))
    R = np.prod(sorted_dims[1:])
    print(1, '<= rank(T) <=', R)
    print()

    # Show generic rank.
    R_gen = int(np.ceil( np.prod(sorted_dims)/(np.sum(sorted_dims) - L + 1) ))
    print('Generic rank of the tensor space of T =', R_gen)
    print()
    
    # Multilinear rank (only for third order tensors).
    if L == 3:
        trunc_dims = 0
        level = 2
        display = 3
        print('Computing multilinear rank...')
        print('------------------------------------')
        try:
            S, best_energy, R1, R2, R3, U1, U2, U3, sigma1, sigma2, sigma3, mlsvd_stop, rel_error = tfx.mlsvd(T, Tsize, R_gen, trunc_dims, level, display)
            print('Estimated multirank(T) =', R1, ',', R2, ',', R3)
            print('|T - (U1, U2, U3)*S|/|T| =', rel_error)
            print()
        except SystemExit:  
            print()      
    
    # Estimative of the rank of T.
    print('Computing rank...')
    r, error_per_rank = tfx.rank(T, plot=False)
    print()
    
    return


@njit(nogil=True)
def adjust(S, m, n, p):
    """
    A CPD of a rgb image will have aproximated values, not integers in the
    range [0, 255]. This function fix this problem. 
    """

    for i in range(m):
        for j in range(n):
            for k in range(p):
                S[i,j,k] = np.floor(S[i,j,k])
                if S[i,j,k] > 255:
                    S[i,j,k] = 255
                elif S[i,j,k] < 0: 
                    S[i,j,k] = 0
                    
    return S


@njit(nogil=True, parallel=True)
def rank1(Lambda, X, Y, Z, m, n, p, r, k):
    """
    Compute each rank 1 term of the CPD given by Lambda, X, Y, Z. Them it
    converts these factors into a matrix, which is the first frontal slice of the
    tensor in coordinates obtained by this rank 1 term. By doing this for all
    r terms, we have a tensor with r slices, each one representing a rank
    1 term of the original CPD.

    Inputs
    ------
    Lambda, X, Y, Z: float ndarray
        Their are the CPD of some tensor.
    m, n, p, r: int
    k: int
        Slice we want to compute.

    Outputs
    -------
    rank1_sections: 3-d float ndarray
        Each matrix rank1_sections[:,:,l] is the k-th section associated with the
    l-th factor in the CPD of some tensor. 
    """
    
    # Each frontal slice of rank1_sections is the coordinate representation of a
    # rank one term of the CPD given by (X,Y,Z)*Lambda.
    rank1_sections = np.zeros((m,n,r), dtype = np.float64)
    T_aux = np.zeros((m,n,p), dtype = np.float64)
   
    for l in prange(0,r):
        for i in range(0,m):
            for j in range(0,n):
                rank1_sections[i,j,l] = Lambda[l]*X[i,l]*Y[j,l]*Z[k,l]
                        
    return rank1_sections


def rank1_plot(Lambda, X, Y, Z, m, n, p, r, k=0, num_rows=5, num_cols=5, greys=True, rgb=False, save=False):
    """
    This function generates an image with the frontal sections of all rank one
    terms (in coordinates) of some CPD. It also saves the image in a file. 
    Warning: this functions uses a lot of memory.

    Inputs
    ------
    Lambda, X, Y, Z: float ndarray
        Their are the CPD of some tensor.
    m, n, p, r: int
    k: int
        Slice we want to visualize.
    num_rows, num_cols: int
        The dimensions of the grid of subplots. We recommend using squares grids in
    order to maximize the size of each subplot. Some blank squares may be left at 
    the end.
    greys: bool
        If True (default), it will show all slices in gray scale. Otherwise it
    will show the RGB evolution of the slices. In this case the parameter 'rgb'
    should be set to True.
    rgb: bool
        If True, it will show all the RGB evolution of the slices. Default is rgb=False.
    """

    sections = rank1(Lambda, X, Y, Z, m, n, p, r, k)
    l = 0
    count = 0
    
    while l < r:
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(30, 30), sharex='col', sharey='row')     
        for i in range(num_rows):
            for j in range(num_cols):
                ax[i,j].xaxis.set_major_locator(plt.NullLocator())
                ax[i,j].yaxis.set_major_locator(plt.NullLocator())
                if l < r:
                    if greys:
                        temp = np.zeros((m, n))
                        temp = sections[:,:,l] 
                        ax[i,j].imshow(temp, cmap='gray')
                    elif rgb:
                        temp = np.zeros((m, n, 3))
                        temp[:,:,k] = sections[:,:,l] 
                        ax[i,j].imshow(np.array(temp, dtype = np.uint8))
                    else:
                        return
                l += 1
    
        if save:
            name = 'fig_' + str(count) + '.png'
            plt.savefig(name)   
        count += 1
        
    return 


def rank_progress(Lambda, X, Y, Z, m, n, p, r, k=0, greys=True, rgb=False):
    """
    Plots the partial sums of rank one terms coresponding to the k-th slice of
    the CPD. The last image should match the original CPD. Use rgb=True only
    for tensors of the form (m, n, 3) enconding RGB format. The program will
    display the red rank one terms, then it will add the green rank one terms
    and then the blue rank one terms. This ordering may cause some distortions
    on the final image.

    Inputs
    ------
    Lambda, X, Y, Z: float ndarrays 
        Their are the CPD of some tensor.
    m, n, p, r: int
    k: int
        Slice we want to visualize.
    greys: bool
        If True (default), it will show all slices in gray scale. Otherwise it
    will show the RGB evolution of the slices. In this case the parameter 'rgb'
    should be set to True.
    rgb: bool
        If True, it will show all the RGB evolution of the slices. False is default.    
    """
    
    if greys:
        temp = np.zeros((m, n))
        sections = rank1(Lambda, X, Y, Z, m, n, p, r, k)
        for l in range(0,r):
            temp = temp + sections[:,:,l] 
            plt.imshow(temp, cmap='gray')
            name = 'factor_' + str(l+1) + '.png'
            plt.savefig(name) 
            plt.show()
            
    elif rgb:
        count = 0
        temp = np.zeros((m, n, 3))
        for color_choice in [0,1,2]:
            sections = rank1(Lambda, X, Y, Z, m, n, p, r, color_choice)
            for l in range(0,r):
                temp[:,:,color_choice] = temp[:,:,color_choice] + sections[:,:,l] 
                plt.imshow(np.array(temp, dtype = np.uint8))
                name = 'factor_' + str(count) + '.png'
                plt.savefig(name)
                plt.show()
                count += 1
                
    else:
        return
 
    return
