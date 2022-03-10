def my_function():
    """
    >>> import numpy as np
    >>> import TensorFox as tfx
    
    Initialize dimensions of the tensor.
	>>> k = 2
	>>> dims = (k+1, k+1, k+1, k+1)
	>>> L = len(dims)

	Create four random factors matrices and from the factor matrices generate the respective tensor in coordinates.
	>>> orig_factors = []
    >>> for l in range(L):
    ...     M = np.random.randn(dims[l], k)
    ...     Q, R = np.linalg.qr(M)
    ...     orig_factors.append(Q)
	>>> T = tfx.cpd2tens(orig_factors)
	
	Compute the CPD of A using the tensor train method.
	>>> class options:    
	...	    pass		
	>>> options = tfx.make_options(options)    
	>>> options.method = 'ttcpd'
	>>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, k, options)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    >>> print('|T - T_approx|/|T| < 1e-14', best_error < 1e-14)
    |T - T_approx|/|T| < 1e-14 True
    
    Now let's see what happens when we set tol_mlsvd = [1e-6, -1] for the tensor. This choice means the program will 
    perform the high order compression using 10^(-6) as tolerance, and will not compress the intermediate third order tensors. 
    >>> options.tol_mlsvd = [1e-6, -1]
    >>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, k, options)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    >>> print('|T - T_approx|/|T| < 1e-14', best_error < 1e-14)
    |T - T_approx|/|T| < 1e-14 True
    
    # Now we use 5 epochs on the same tensor.
    >>> class options:    
    ...     pass		
	>>> options = tfx.make_options(options)    
    >>> options.epochs = 5
	>>> options.method = 'ttcpd'
	>>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, k, options)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    >>> print('|T - T_approx|/|T| < 1e-14', best_error < 1e-14)
    |T - T_approx|/|T| < 1e-14 True
    """

    return 