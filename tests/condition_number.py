def my_function():
    """
    >>> import numpy as np
    >>> import TensorFox as tfx
    
    Create a random rank-R tensor. Tensor Fox has a function for this purpose.
    >>> R = 5
    >>> dims = [20, 30, 40] 
    >>> T, orig_factors = tfx.gen_rand_tensor(dims, R)
    
    Compute the CPD.
    >>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, R)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    
    Compute backward error and condition number of the approximation.
    >>> backward_error = output.rel_error * np.linalg.norm(T)
    >>> condition_number = tfx.cond(factors)
    
    Verify condition number inequality.
    >>> forward_error, new_factors, idx = tfx.forward_error(orig_factors, factors)
    >>> print('forward error <= condition_number * backward_error', forward_error <= condition_number*backward_error)
    forward error <= condition_number * backward_error True
    """

    return 