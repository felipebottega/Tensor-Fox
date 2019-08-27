import scipy.io as spio
import sys
import numpy as np
import h5py
import TensorFox as tfx


def loadmat(filename):
    """
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.    
    Source: 
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    return _check_keys(data)


def mat2list(X):
    """
    If the initialization method is a cell of arrays given by the user, some preparation is 
    necessary to transform the cell into a list. No shape verifications are made, so make sure
    that everything is correct. Furthermore, it is necessary to reverse the order of the list
    elements.
    """
    
    init = []
    for x in X:
        # Each x is a list where x[i] is the i-th row of a factor matrix.
        rows = len(x)
        cols = len(x[0])
        M = np.zeros((rows, cols))    
        i = 0
        for row in x:
            M[i, :] = row
            i += 1        
        init.append(M)

    init.reverse()
                
    return init


def loadstructure(filename):
    """
    This function loads the Matlab structure and converts it to a Python class called 'options',
    to be used in Tensor Fox.
    """
    
    # The original structure in Matlab should be called 'options'. If this is not the case, the 
    # program tries to use the last element of the dictionary opt (usually this is fine).
    opt = loadmat(filename)
    if 'options' in opt.keys():
        options_dict = opt['options']
    else:
        key = list(opt.keys())[-1]
        options_dict = opt[key]
    
    # Create class options for Tensor Fox.
    options = False
    options = tfx.aux.make_options(options)

    if 'maxiter' in options_dict.keys():
        options.maxiter = options_dict['maxiter']
    if 'tol' in options_dict.keys():
        options.tol = options_dict['tol']
    if 'tol_step' in options_dict.keys():
        options.tol_step = options_dict['tol_step']
    if 'tol_improv' in options_dict.keys():
        options.tol_improv = options_dict['tol_improv']
    if 'tol_grad' in options_dict.keys():
        options.tol_grad = options_dict['tol_grad']
    if 'method' in options_dict.keys():
        options.method = options_dict['method']
        
    if 'inner_method' in options_dict.keys():
        options.inner_method = options_dict['inner_method']
    if 'cg_maxiter' in options_dict.keys():
        options.cg_maxiter = options_dict['cg_maxiter']
    if 'cg_factor' in options_dict.keys():
        options.cg_factor = options_dict['cg_factor']
    if 'cg_tol' in options_dict.keys():
        options.cg_tol = options_dict['cg_tol']
        
    if 'bi_method' in options_dict.keys():
        options.bi_method = options_dict['bi_method']
    if 'bi_method_maxiter' in options_dict.keys():
        options.bi_method_maxiter = options_dict['bi_method_maxiter']
    if 'bi_method_tol' in options_dict.keys():
        options.bi_method_tol = options_dict['bi_method_tol']    
        
    if 'initialization' in options_dict.keys():
        initialization = options_dict['initialization']
        if type(initialization) == list:
            initialization = mat2list(initialization)
        options.initialization = initialization
    if 'trunc_dims' in options_dict.keys():
        options.trunc_dims = list(np.array(options_dict['trunc_dims'], dtype=np.int64))
    if 'tol_mlsvd' in options_dict.keys():
        options.tol_mlsvd = options_dict['tol_mlsvd']
    if 'init_damp' in options_dict.keys():
        options.init_damp = options_dict['init_damp']
    if 'refine' in options_dict.keys():
        options.refine = options_dict['refine']
    if 'symm' in options_dict.keys():
        options.symm = options_dict['symm']
    if 'low' in options_dict.keys():
        options.constraints[0] = options_dict['low']
    if 'upp' in options_dict.keys():
        options.constraints[1] = options_dict['upp']
    if 'factor' in options_dict.keys():
        options.constraints[2] = options_dict['factor']
    if 'factors_norm' in options_dict.keys():
        options.factors_norm = options_dict['factors_norm']
    if 'trials' in options_dict.keys():
        options.trials = options_dict['trials']
    if 'display' in options_dict.keys():
        options.display = options_dict['display']
    if 'epochs' in options_dict.keys():
        options.epochs = options_dict['epochs']
        
    return options


def loadarray(filename):
    """
    This functions loads Matlab arrays saved in the disk with .mat format. Does work for Matlab v7.3.
    """

    arrays = {}

    # Try to load with h5py. Works when Matlab saves a .mat with options '-v7.3', '-nocompression'.
    try:
        with h5py.File(filename, 'r') as f:
            for k, v in f.items():
                arrays[k] = np.array(v)
        T = arrays['T']

    # In the case the options above are not used, this part does the correct load.
    except OSError:
        T_matlab = spio.loadmat(filename)
        for x in T_matlab.keys():
            if isinstance(T_matlab[x], np.ndarray):
                break
        T = T_matlab[x]
        return T

                
    return T
