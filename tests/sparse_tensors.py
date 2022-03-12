def my_function():
    """
    >>> import numpy as np
    >>> import TensorFox as tfx
    
    Generate sparse tensor.
    >>> dims = [10, 10, 10, 10]
    >>> R = 5 
    >>> nnz = 10
    >>> data, idxs, dims, factors = tfx.gen_rand_sparse_tensor(dims, R, nnz)
    >>> T = [data, idxs, dims]
    
    Compute the CPD of the tensor.
    >>> class options:
    ...     method = 'dGN'
    >>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, R, options)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    
     Generate the coordinate (dense) format of the approximation.
    >>> T_approx = tfx.cpd2tens(factors)

    Generate the dense format from the sparse representation.
    >>> T_dense = tfx.sparse2dense(data, idxs, dims)

    Compute the error.
    >>> rel_error = np.linalg.norm(T_dense - T_approx)/np.linalg.norm(T_dense)
    >>> print('|T - T_approx|/|T| < 1e-4', rel_error < 1e-4)
    |T - T_approx|/|T| < 1e-4 True
    """

    return 