"""
 Display Module
 ==============
 This module contains function centered in computing and displaying general information about tensors and tensor spaces. 
"""

# Python modules
import numpy as np
from numpy import mean, var, array, sort, ceil, floor, zeros, uint8, uint64, prod, argsort
from numpy.linalg import norm
import time
import warnings
import IPython.display as ipd
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd

# Tensor Fox modules
import TensorFox.Auxiliar as aux
import TensorFox.Compression as cmpr
import TensorFox.MultilinearAlgebra as mlinalg
import TensorFox.TensorFox as tfx


def showtens(T):
    """
    Let T = T_(i,j,k) be a tensor in coordinates. It is usual to consider that i is the row coordinate, j is the column 
    coordinate and k is the section coordinate. But if we just run the command print(T) we will see that numpy considers
    i as the slice coordinate, j the row coordinate and k the column coordinate. This is not the usual way to consider
    tensors, and if we want to print T section by section (i.e., each frontal slice separately), this function does the
    job.
    """
    
    dims = T.shape
    L = len(dims)

    if L == 3:
        for k in range(T.shape[2]):
            print(T[:, :, k])
            print()
    else:
        print(T)
        
    return


def infotens(T):
    """
    Given a tensor T, this function computes and shows several information about this tensor. There are only print
    outputs. Since this function tries to estimate the rank of T, be aware that it may take a long time to finish all
    computations.
    """
    
    # Compute dimensions and norm of T.
    dims = T.shape
    L = len(dims)
    Tsize = norm(T)
    
    print('T is a tensor of dimensions', dims)
    print()
    print('|T| =', Tsize)
    print()
    
    # Max and min entries of T.
    print('max(T) =', np.max(T))
    print()
    print('min(T) =', np.min(T))
    print()
    print('mean(T) =', mean(T))
    print()
    print('mean(|T|) =', mean(np.abs(T)))
    print()
    print('var(T) =', var(np.abs(T)))
    print()
    
    # Histogram of the entries of T.
    plt.hist(T.flatten(), bins=50)
    plt.title('Tensor histogram')
    plt.xlabel('Tensor values')
    plt.ylabel('Quantity')
    plt.show()
    print()
    
    # Bounds on rank.
    sorted_dims = sort(array(dims))
    R = int(prod(sorted_dims[1:], dtype=uint64))
    print(1, '<= rank(T) <=', R)
    print()

    # Show generic rank.
    R_gen = int(ceil( int(prod(sorted_dims, dtype=uint64))/(np.sum(sorted_dims) - L + 1) ))
    print('Generic rank of the tensor space of T =', R_gen)
    print()
    
    # Multilinear rank.
    class options:
        display = 3

    options = aux.make_options(options)
    print('Computing multilinear rank...')
    print('------------------------------------')
    S, U, UT, sigmas, rel_error = cmpr.mlsvd(T, Tsize, R_gen, options)
    print('multirank(T) =', S.shape)
    print('|T - (U_1, ..., U_' + str(L) + ')*S|/|T| =', rel_error)
    print()
    
    # Estimate of the rank of T.
    print('Computing rank...')
    r, error_per_rank = tfx.rank(T, plot=False)
    print()
    
    return


def rank1_plot(X, Y, Z, m, n, R, k=0, num_rows=5, num_cols=5, greys=True, rgb=False, save=False):
    """
    This function generates an image with the frontal sections of all rank one terms (in coordinates) of some CPD. It
    also saves the image in a file.
    Warning: this functions uses a lot of memory.

    Inputs
    ------
    X, Y, Z: 2-D arrays
        Their are the CPD of some third order tensor.
    m, n, R: int
    k: int
        Slice we want to visualize.
    num_rows, num_cols: int
        The dimensions of the grid of subplots. We recommend using squares grids in order to maximize the size of each 
        subplot. Some blank squares may be left at the end.
    greys: bool
        If True (default), it will show all slices in gray scale. Otherwise it will show the RGB evolution of the
        slices.
        In this case the parameter 'rgb' should be set to True.
    rgb: bool
        If True, it will show all the RGB evolution of the slices. Default is rgb=False.
    """

    sections = mlinalg.rank1(X, Y, Z, m, n, R, k)
    r = 0
    count = 0
    
    while r < R:
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(30, 30), sharex='col', sharey='row')     
        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].xaxis.set_major_locator(plt.NullLocator())
                ax[i, j].yaxis.set_major_locator(plt.NullLocator())
                if r < R:
                    if greys:
                        temp = sections[:, :, r]
                        ax[i, j].imshow(temp, cmap='gray')
                    elif rgb:
                        temp = zeros((m, n, 3))
                        temp[:, :, k] = sections[:, :, r]
                        ax[i, j].imshow(array(temp, dtype=uint8))
                    else:
                        return
                r += 1
    
        if save:
            name = 'fig_' + str(count) + '.png'
            plt.savefig(name)   
        count += 1
        
    return 


def rank_progress(X, Y, Z, m, n, R, k=0, greys=True, rgb=False):
    """
    Plots the partial sums of rank one terms corresponding to the k-th slice of the CPD. The last image should match the
    original CPD. Use rgb=True only for tensors of the form (m, n, 3) encoding RGB format. The program will display the
    red rank one terms, then it will add the green rank one terms and then the blue rank one terms. This ordering may
    cause some distortions on the final image.

    Inputs
    ------
    X, Y, Z: 2-D arrays
        Their are the CPD of some third order tensor.
    m, n, p, R: int
    k: int
        Slice we want to visualize.
    greys: bool
        If True (default), it will show all slices in gray scale. Otherwise it will show the RGB evolution of the
        slices. In this case the parameter 'rgb' should be set to True.
    rgb: bool
        If True, it will show all the RGB evolution of the slices. False is default.    
    """
    
    if greys:
        temp = zeros((m, n))
        sections = mlinalg.rank1(X, Y, Z, m, n, R, k)
        for r in range(R):
            temp = temp + sections[:, :, r]
            plt.imshow(temp, cmap='gray')
            name = 'factor_' + str(r+1) + '.png'
            plt.savefig(name) 
            plt.show()
            
    elif rgb:
        count = 0
        temp = zeros((m, n, 3))
        for color_choice in [0, 1, 2]:
            sections = mlinalg.rank1(X, Y, Z, m, n, R, color_choice)
            for r in range(R):
                temp[:, :, color_choice] = temp[:, :, color_choice] + sections[:, :, r]
                plt.imshow(array(temp, dtype=uint8))
                name = 'factor_' + str(count) + '.png'
                plt.savefig(name)
                plt.show()
                count += 1
                
    else:
        return
 
    return


@njit(nogil=True)
def adjust(S, m, n, p):
    """
    A CPD of a rgb image will have approximated values, not integers in the range [0, 255]. This function fix this.
    """

    for i in range(m):
        for j in range(n):
            for k in range(p):
                S[i, j, k] = floor(S[i, j, k])
                if S[i, j, k] > 255:
                    S[i, j, k] = 255
                elif S[i, j, k] < 0:
                    S[i, j, k] = 0
                    
    return S


def test_tensors(tensors_list, options_list, trials, display):
    """
    Each element in the list 'tensors_list' is a tuple (name, T, R, thr) or (name, T, T_noise, R, thr). We have that
        - 'name' is any name for the tensor  
        - ' T is the tensor itself 
        - R is the rank
        - thr is a threshold for the relative error to be considered a success if smaller than thr.
    In the case we are working with noise, then T_noise is the noisy tensor. The CPD will be computed for T_noise but 
    the error will be computed for T.
    Each element of 'options_list' is a class of options to use in the respective CPD computation of the corresponding
    tensor in 'tensors_list'. More precisely, options_list[i] are the options to used for the COD of tensors_list[i].
    If the user want to use default options, just pass options_list as a list of False with same length of tensors_list.
    """
    
    warnings.filterwarnings('ignore')
    timings = np.zeros(len(tensors_list))
    errors_per_tensor = np.zeros(trials)
    errors_mean_good = np.zeros(len(tensors_list))
    errors_var_good = np.zeros(len(tensors_list))
    errors_mean_bad = np.zeros(len(tensors_list))
    errors_var_bad = np.zeros(len(tensors_list))
    num_good = np.zeros(len(tensors_list))
    num_bad = np.zeros(len(tensors_list))  
    names = []

    i = 0
    df = pd.DataFrame(columns=['Name',
                               'Method',
                               'Maxiter',
                               'Tol error',
                               'Tol step size',
                               'Tol improvement',
                               'Tol gradient',
                               'Initialization',
                               'Inner algorithm',
                               'Bi-CPD parameters',
                               '# Success',
                               '# Fail'])
    
    for element in tensors_list:
        if len(element) == 4:
            name = element[0]
            T = element[1]
            R = element[2]
            thr = element[3]
        elif len(element) == 5:
            name = element[0]
            T = element[1]
            T_noise = element[2]
            R = element[3]
            thr = element[4]
        else:
            print('Error in list of tensors.')
            break
        
        names.append(element[0])
        Tsize = np.linalg.norm(T)
        options = options_list[i]
            
        start = time.time()

        for t in range(trials):
            if len(element) == 4:
                factors, output = tfx.cpd(T, R, options) 
            else:
                factors, output = tfx.cpd(T_noise, R, options)
            T_approx = tfx.cnv.cpd2tens(factors)
            errors_per_tensor[t] = np.linalg.norm(T - T_approx)/Tsize  

        end = time.time()
        timings[i] = (end - start)/trials
    
        if np.sum(errors_per_tensor < thr) != 0:
            errors_mean_good[i] = errors_per_tensor[errors_per_tensor < thr].mean()
            errors_var_good[i] = errors_per_tensor[errors_per_tensor < thr].var()
            num_good[i] = np.sum(errors_per_tensor < thr)
        else:
            errors_mean_good[i] = None
            errors_var_good[i] = 0
            num_good[i] = None
        
        if np.sum(errors_per_tensor >= thr) != 0:
            errors_mean_bad[i] = errors_per_tensor[errors_per_tensor >= thr].mean()
            errors_var_bad[i] = errors_per_tensor[errors_per_tensor >= thr].var()
            num_bad[i] = np.sum(errors_per_tensor >= thr)
        else:
            errors_mean_bad[i] = None
            errors_var_bad[i] = 0
            num_bad[i] = None
            
        num_good_print = num_good[i]
        num_bad_print = num_bad[i]
        
        if np.isnan(num_good[i]):
            num_bad[i] = trials
            num_good_print = 0 
        elif np.isnan(num_bad[i]):
            num_good[i] = trials
            num_bad_print = 0
   
        method = output.options.method
        maxiter = output.options.maxiter
        tol = output.options.tol
        tol_step = output.options.tol_step
        tol_improv = output.options.tol_improv
        tol_grad = output.options.tol_grad
        init = output.options.initialization
        if type(init) == list:
            init = 'user'
        if output.options.method == 'als' or output.options.method == 'ttcpd':
            temp1 = ['', '', '']
        elif output.options.inner_method == 'cg' or output.options.inner_method == 'cg_static':
            temp1 = [output.options.inner_method, output.options.cg_factor, output.options.cg_tol]
        elif output.options.inner_method == 'gd':
            temp1 = [output.options.inner_method, '', '']
        else:
            temp1 = ['hybrid strategy', '', '', '']
        if output.options.method == 'ttcpd':
            temp2 = output.options.bi_method_parameters
        else:
            temp2 = '          '
        df.loc[i] = [name,
                     method,
                     maxiter,
                     tol, 
                     tol_step, 
                     tol_improv, 
                     tol_grad,
                     init,
                     [temp1[0], temp1[1], temp1[2]],
                     [temp2[0], temp2[1], temp2[2]],
                     int(num_good_print),
                     int(num_bad_print)]
        i += 1
        
        if display:
            ipd.clear_output()
            ipd.display(df)
            print()
        
    make_plots(names,
               trials,
               errors_mean_good, errors_var_good, num_good,
               errors_mean_bad, errors_var_bad, num_bad,
               timings)
    
    return errors_mean_good, errors_var_good, num_good, errors_mean_bad, errors_var_bad, num_bad, timings


def make_plots(names,
               trials,
               errors_mean_good, errors_var_good, num_good,
               errors_mean_bad, errors_var_bad, num_bad,
               timings):
    """
    After the function 'test_tensors' is finished, all data is passed to this function, which makes informative plots
    with the percentage of successes and failures, plus a plot with the average timings. These two functions combined
    are a great tool for modelling and hyperparameter grid search.
    """    
    
    plt.figure(figsize=[16, 6])

    plt.errorbar(
        names, errors_mean_bad, 
        yerr=errors_var_bad, 
        label="Fail", 
        fmt="rs", 
        linewidth=4, 
        elinewidth=2,
        ecolor='k',
        capsize=5,
        capthick=1
        )

    for i in range(len(names)):
        x, y = names[i], errors_mean_bad[i]
        plt.annotate(str(np.round(100*num_bad[i]/trials, 1)) + ' %',  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 20),  # distance from text to points (x,y)
                     ha='center')
    
    plt.errorbar(
        names, errors_mean_good, 
        yerr=errors_var_good, 
        label="Success", 
        fmt="bs", 
        linewidth=4, 
        elinewidth=2,
        ecolor='k',
        capsize=5,
        capthick=1
        )

    for i in range(len(names)):
        x, y = names[i], errors_mean_good[i]
        plt.annotate(str(np.round(100*num_good[i]/trials, 1)) + ' %',  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, -20),  # distance from text to points (x,y)
                     ha='center')

    plt.legend(numpoints=1, loc='center left')  
    plt.yscale('log')
    plt.ylim( (max(1e-16, np.nanmin(errors_mean_good)/10), max(1, np.nanmax(errors_mean_bad)*10)) )
    plt.grid()
    plt.show()
    
    plt.figure(figsize=[16, 6])
    plt.plot(names, timings, '--')
    plt.plot(names, timings, 'ks')
    plt.grid()
    plt.ylabel('Seconds (average)')
    plt.show()
    
    return


def show_options(output):
    """
    This function display all parameter options used in a previous call of the cpd function.
    """

    members_names = [attr for attr in dir(output.options) if not attr.startswith("__")]
    members = [getattr(output.options, attr) for attr in dir(output.options) if not attr.startswith("__")]
    L = len(members)
    for m in range(L):
        print(members_names[m], ':', members[m])

    return
