import numpy as np
import TensorFox as tfx


def gen_rand_tensor(dims, R):
    """
    This function generates a random rank-R tensor T of shape (dims[0], dims[1], ..., dims[L-1]), where
    L is the order of T. Each factor of T is a matrix of shape (dims[l], R). Let W[l] be the l-th factor
    matrix of T, then T = (W[0], W[1], ..., W[L-1])*I, where I is a diagonal R x R x... x R (L times)
    tensor and the parenthesis notation stands for the multilinear multiplication. In other words, one
    have that 
     T[i_{0}, ..., i_{L-1}] = sum_{r=0}^{R-1} W[0][i_{0}, r] * W[1][i_{1}, r] * ... * W[L-1][i_{L-1}, r]. 

    Input
    -----
    dims: tuple or list of ints
        The dimensions of the tensor
    R: int
        The rank of the tensor

    Output
    ------
    T: float ndarray with L dimensions
        The tensor in coordinate format
    orig_factors: list
        List of the factor matrices of T. We have that orig_factors[l] = W[l], as described above.
    """

    L = len(dims)
    orig_factors = []

    for l in range(L):
        M = np.random.randn(dims[l], r)
        orig_factors.append(M)

    T = np.zeros(dims)
    T = tfx.cnv.cpd2tens(T, orig_factors, dims) 

    return T, orig_factors
