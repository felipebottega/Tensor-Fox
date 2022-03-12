def my_function():
    """
    >>> import numpy as np
    >>> import TensorFox as tfx
    
    Create and print the tensor, which is 2 x 2 x 2.
    >>> m = 2
    >>> T = np.zeros((m, m, m))
    >>> s = 0
    >>> for k in range(m):
    ...     for i in range(m):
    ...         for j in range(m):
    ...             T[i,j,k] = s
    ...             s += 1
    
    Compute the CPD of T (make 10 trials and keep the best one), assuming T has rank 3.
    >>> R = 3
    >>> best_error = 1
    >>> for i in range(10):
    ...     factors, output = tfx.cpd(T, R)
    ...     if output.rel_error < best_error:
    ...         best_error = output.rel_error
    >>> print('|T - T_approx|/|T| < 1e-14', best_error < 1e-14)
    |T - T_approx|/|T| < 1e-14 True
    
    Compute the CPD of T with user initialization.
    >>> class options:    # start creating an empty class
    ...    pass
    >>> options = tfx.make_options(options)    # update the class
    >>> X = np.ones((m, R))
    >>> Y = np.ones((m, R))
    >>> Z = np.ones((m, R))
    >>> options.initialization = [X,Y,Z]
    >>> factors, output = tfx.cpd(T, R, options)
    >>> print('|T - T_approx|/|T| < 0.14', output.rel_error < 0.14)
    |T - T_approx|/|T| < 0.14 True
    
    Compute the CPD of T with refinement.
    >>> options.display = 0
    >>> options.refine = True
    >>> factors, output = tfx.cpd(T, R, options)
    >>> print('|T - T_approx|/|T| < 0.14', output.rel_error < 0.14)
    |T - T_approx|/|T| < 0.14 True
    
    Compute the CPD with cg_static as the inner algorithm, with 3 iterations max and tolerance of 1e-7.
    >>> options.inner_method = 'cg_static'
    >>> options.cg_maxiter = 3
    >>> options.cg_tol = 1e-7
    >>> options.refine = False
    >>> factors, output = tfx.cpd(T, R, options)
    >>> print('|T - T_approx|/|T| < 0.14', output.rel_error < 0.14)
    |T - T_approx|/|T| < 0.14 True
    """

    return 