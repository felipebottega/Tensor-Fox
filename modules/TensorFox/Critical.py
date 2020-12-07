"""
 Critical Module
 ===============
 This module is responsible for the most costly parts of Tensor Fox.
"""

import numpy as np
from numpy import dot, empty, zeros, float64
from numpy.linalg import norm
from numba import njit, prange


@njit(nogil=True, parallel=True)
def fastnorm(A, B):
    m, n = A.shape
    s = 0.0
    for i in prange(m):
        for j in range(n):
            s += (A[i, j] - B[i, j])**2
    s = np.sqrt(s)
    return s


def sparse_fastnorm(data, idxs, dims, factors):
    """
    This function computes the error between the nonzero entries in data and their corresponding approximations given
    by the factor matrices. The zero entries are not taken in account.
    """

    R = factors[0].shape[1]
    L = len(dims)
    nnz = len(idxs)

    data_approx_cols = zeros((nnz, R))
    data_approx_cols = sparse_fastnorm_computations(data_approx_cols, idxs, factors, R, L, nnz)
    data_approx_cols = np.sum(data_approx_cols, axis=1)
    s = norm(data - data_approx_cols)

    return s


@njit(nogil=True, parallel=True)
def sparse_fastnorm_computations(data_approx_cols, idxs, factors, R, L, nnz):
    for r in prange(R):
        for i in range(nnz):
            j = idxs[i]
            p = 1
            for l in range(L):
                p *= factors[l][j[l], r]
            data_approx_cols[i, r] += p

    return data_approx_cols


@njit(nogil=True, parallel=True)
def unfold1_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i2 in prange(I2):
        for i1 in range(I1):
            s = i2*I1 + i1
            Tl[:,s] = T[:, i1, i2]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i2 in prange(I2):
        for i0 in range(I0):
            s = i2*I0 + i0
            Tl[:,s] = T[i0, :, i2]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i1 in prange(I1):
        for i0 in range(I0):
            s = i1*I0 + i0
            Tl[:,s] = T[i0, i1, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i1 in range(I1):
                s = i3*I1*I2 + i2*I1 + i1
                Tl[:,s] = T[:, i1, i2, i3]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i0 in range(I0):
                s = i3*I0*I2 + i2*I0 + i0
                Tl[:,s] = T[i0, :, i2, i3]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i1 in range(I1):
            for i0 in range(I0):
                s = i3*I0*I1 + i1*I0 + i0
                Tl[:,s] = T[i0, i1, :, i3]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i2 in prange(I2):
        for i1 in range(I1):
            for i0 in range(I0):
                s = i2*I0*I1 + i1*I0 + i0
                Tl[:,s] = T[i0, i1, i2, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    s = i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                    Tl[:,s] = T[:, i1, i2, i3, i4]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i0 in range(I0):
                    s = i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                    Tl[:,s] = T[i0, :, i2, i3, i4]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                    Tl[:,s] = T[i0, i1, :, i3, i4]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i2 in range(I2):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                    Tl[:,s] = T[i0, i1, i2, :, i4]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                    Tl[:,s] = T[i0, i1, i2, i3, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        s = i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                        Tl[:,s] = T[:, i1, i2, i3, i4, i5]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i0 in range(I0):
                        s = i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                        Tl[:,s] = T[i0, :, i2, i3, i4, i5]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                        Tl[:,s] = T[i0, i1, :, i3, i4, i5]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        Tl[:,s] = T[i0, i1, i2, :, i4, i5]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        Tl[:,s] = T[i0, i1, i2, i3, :, i5]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        Tl[:,s] = T[i0, i1, i2, i3, i4, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            s = i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                            Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i0 in range(I0):
                            s = i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                            Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                            Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                s = i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i0 in range(I0):
                                s = i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :, i7]
    return Tl


@njit(nogil=True, parallel=True)
def unfold8_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    s = i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                    Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i0 in range(I0):
                                    s = i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                    Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :, i7, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold8_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, :, i8]
    return Tl


@njit(nogil=True, parallel=True)
def unfold9_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        s = i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                        Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i0 in range(I0):
                                        s = i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                        Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold8_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold9_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9]
    return Tl


@njit(nogil=True, parallel=True)
def unfold10_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            s = i10*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                            Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i0 in range(I0):
                                            s = i10*I0*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                            Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold8_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold9_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold10_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :, i10]
    return Tl


@njit(nogil=True, parallel=True)
def unfold11_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, :]
    return Tl


@njit(nogil=True, parallel=True)
def unfold1_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i2 in range(I2):
                                            for i1 in range(I1):
                                                s = i11*I1*I2*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                                Tl[:,s] = T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold2_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i2 in range(I2):
                                            for i0 in range(I0):
                                                s = i11*I0*I2*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                                Tl[:,s] = T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold3_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold4_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold5_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold6_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold7_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold8_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold9_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I7*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold10_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I8*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :, i10, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold11_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, :, i11]
    return Tl


@njit(nogil=True, parallel=True)
def unfold12_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                Tl[:,s] = T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, :]
    return Tl


@njit(nogil=True, parallel=True)
def foldback1_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i2 in prange(I2):
        for i1 in range(I1):
            s = i2*I1 + i1
            T[:, i1, i2] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i2 in prange(I2):
        for i0 in range(I0):
            s = i2*I0 + i0
            T[i0, :, i2] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order3(T, Tl, dims):
    I0, I1, I2 = dims
    for i1 in prange(I1):
        for i0 in range(I0):
            s = i1*I0 + i0
            T[i0, i1, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i1 in range(I1):
                s = i3*I1*I2 + i2*I1 + i1
                T[:, i1, i2, i3] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i0 in range(I0):
                s = i3*I0*I2 + i2*I0 + i0
                T[i0, :, i2, i3] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i3 in prange(I3):
        for i1 in range(I1):
            for i0 in range(I0):
                s = i3*I0*I1 + i1*I0 + i0
                T[i0, i1, :, i3] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order4(T, Tl, dims):
    I0, I1, I2, I3 = dims
    for i2 in prange(I2):
        for i1 in range(I1):
            for i0 in range(I0):
                s = i2*I0*I1 + i1*I0 + i0
                T[i0, i1, i2, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    s = i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                    T[:, i1, i2, i3, i4] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i0 in range(I0):
                    s = i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                    T[i0, :, i2, i3, i4] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                    T[i0, i1, :, i3, i4] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i4 in prange(I4):
        for i2 in range(I2):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                    T[i0, i1, i2, :, i4] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order5(T, Tl, dims):
    I0, I1, I2, I3, I4 = dims
    for i3 in prange(I3):
        for i2 in range(I2):
            for i1 in range(I1):
                for i0 in range(I0):
                    s = i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                    T[i0, i1, i2, i3, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        s = i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                        T[:, i1, i2, i3, i4, i5] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i0 in range(I0):
                        s = i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                        T[i0, :, i2, i3, i4, i5] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                        T[i0, i1, :, i3, i4, i5] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        T[i0, i1, i2, :, i4, i5] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i5 in prange(I5):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        T[i0, i1, i2, i3, :, i5] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order6(T, Tl, dims):
    I0, I1, I2, I3, I4, I5 = dims
    for i4 in prange(I4):
        for i3 in range(I3):
            for i2 in range(I2):
                for i1 in range(I1):
                    for i0 in range(I0):
                        s = i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                        T[i0, i1, i2, i3, i4, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            s = i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                            T[:, i1, i2, i3, i4, i5, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i0 in range(I0):
                            s = i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                            T[i0, :, i2, i3, i4, i5, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                            T[i0, i1, :, i3, i4, i5, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            T[i0, i1, i2, :, i4, i5, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            T[i0, i1, i2, i3, :, i5, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i6 in prange(I6):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            T[i0, i1, i2, i3, i4, :, i6] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order7(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6 = dims
    for i5 in prange(I5):
        for i4 in range(I4):
            for i3 in range(I3):
                for i2 in range(I2):
                    for i1 in range(I1):
                        for i0 in range(I0):
                            s = i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                            T[i0, i1, i2, i3, i4, i5, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                s = i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                T[:, i1, i2, i3, i4, i5, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i0 in range(I0):
                                s = i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                T[i0, :, i2, i3, i4, i5, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                T[i0, i1, :, i3, i4, i5, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                T[i0, i1, i2, :, i4, i5, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                T[i0, i1, i2, i3, :, i5, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                T[i0, i1, i2, i3, i4, :, i6, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i7 in prange(I7):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                T[i0, i1, i2, i3, i4, i5, :, i7] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback8_order8(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7 = dims
    for i6 in prange(I6):
        for i5 in range(I5):
            for i4 in range(I4):
                for i3 in range(I3):
                    for i2 in range(I2):
                        for i1 in range(I1):
                            for i0 in range(I0):
                                s = i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                T[i0, i1, i2, i3, i4, i5, i6, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    s = i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                    T[:, i1, i2, i3, i4, i5, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i0 in range(I0):
                                    s = i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                    T[i0, :, i2, i3, i4, i5, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                    T[i0, i1, :, i3, i4, i5, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, :, i4, i5, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, i3, :, i5, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, i3, i4, :, i6, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, i3, i4, i5, :, i7, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback8_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i8 in prange(I8):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, i3, i4, i5, i6, :, i8] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback9_order9(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8 = dims
    for i7 in prange(I7):
        for i6 in range(I6):
            for i5 in range(I5):
                for i4 in range(I4):
                    for i3 in range(I3):
                        for i2 in range(I2):
                            for i1 in range(I1):
                                for i0 in range(I0):
                                    s = i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                    T[i0, i1, i2, i3, i4, i5, i6, i7, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        s = i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                        T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i0 in range(I0):
                                        s = i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                        T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                        T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback8_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback9_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i9 in prange(I9):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback10_order10(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9 = dims
    for i8 in prange(I8):
        for i7 in range(I7):
            for i6 in range(I6):
                for i5 in range(I5):
                    for i4 in range(I4):
                        for i3 in range(I3):
                            for i2 in range(I2):
                                for i1 in range(I1):
                                    for i0 in range(I0):
                                        s = i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                        T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            s = i10*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                            T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i0 in range(I0):
                                            s = i10*I0*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                            T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                            T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback8_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback9_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback10_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i10 in prange(I10):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :, i10] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback11_order11(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = dims
    for i9 in prange(I9):
        for i8 in range(I8):
            for i7 in range(I7):
                for i6 in range(I6):
                    for i5 in range(I5):
                        for i4 in range(I4):
                            for i3 in range(I3):
                                for i2 in range(I2):
                                    for i1 in range(I1):
                                        for i0 in range(I0):
                                            s = i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                            T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback1_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i2 in range(I2):
                                            for i1 in range(I1):
                                                s = i11*I1*I2*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I1*I2*I3*I4*I5*I6*I7 + i7*I1*I2*I3*I4*I5*I6 + i6*I1*I2*I3*I4*I5 + i5*I1*I2*I3*I4 + i4*I1*I2*I3 + i3*I1*I2 + i2*I1 + i1
                                                T[:, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback2_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i2 in range(I2):
                                            for i0 in range(I0):
                                                s = i11*I0*I2*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I2*I3*I4*I5*I6*I7 + i7*I0*I2*I3*I4*I5*I6 + i6*I0*I2*I3*I4*I5 + i5*I0*I2*I3*I4 + i4*I0*I2*I3 + i3*I0*I2 + i2*I0 + i0
                                                T[i0, :, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback3_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i3 in range(I3):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I3*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I3*I4*I5*I6*I7 + i7*I0*I1*I3*I4*I5*I6 + i6*I0*I1*I3*I4*I5 + i5*I0*I1*I3*I4 + i4*I0*I1*I3 + i3*I0*I1 + i1*I0 + i0
                                                T[i0, i1, :, i3, i4, i5, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback4_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i4 in range(I4):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I4*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I4*I5*I6*I7 + i7*I0*I1*I2*I4*I5*I6 + i6*I0*I1*I2*I4*I5 + i5*I0*I1*I2*I4 + i4*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, :, i4, i5, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback5_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i5 in range(I5):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I5*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I5*I6*I7 + i7*I0*I1*I2*I3*I5*I6 + i6*I0*I1*I2*I3*I5 + i5*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, :, i5, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback6_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i6 in range(I6):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I6*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I6*I7 + i7*I0*I1*I2*I3*I4*I6 + i6*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, :, i6, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback7_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i7 in range(I7):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I7*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I7 + i7*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, :, i7, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback8_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i8 in range(I8):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I8*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I8 + i8*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, i6, :, i8, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback9_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i9 in range(I9):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I9*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I7*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, i6, i7, :, i9, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback10_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i10 in range(I10):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I8*I10 + i10*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, i6, i7, i8, :, i10, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback11_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i11 in prange(I11):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i11*I0*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, :, i11] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def foldback12_order12(T, Tl, dims):
    I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 = dims
    for i10 in prange(I10):
        for i9 in range(I9):
            for i8 in range(I8):
                for i7 in range(I7):
                    for i6 in range(I6):
                        for i5 in range(I5):
                            for i4 in range(I4):
                                for i3 in range(I3):
                                    for i2 in range(I2):
                                        for i1 in range(I1):
                                            for i0 in range(I0):
                                                s = i10*I0*I1*I2*I3*I4*I5*I6*I7*I8*I9 + i9*I0*I1*I2*I3*I4*I5*I6*I7*I8 + i8*I0*I1*I2*I3*I4*I5*I6*I7 + i7*I0*I1*I2*I3*I4*I5*I6 + i6*I0*I1*I2*I3*I4*I5 + i5*I0*I1*I2*I3*I4 + i4*I0*I1*I2*I3 + i3*I0*I1*I2 + i2*I0*I1 + i1*I0 + i0
                                                T[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, :] = Tl[:,s]
    return T


@njit(nogil=True, parallel=True)
def tt_error_order3(T, G0, G1, G2, dims, L):
    a, b, c = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                A = G0[i0, :]
                B = dot(A, G1[:, i1, :])
                T_approx[i0, i1, i2] = dot(B, G2[:, i2])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order4(T, G0, G1, G2, G3, dims, L):
    a, b, c, d = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    A = G0[i0,:]
                    B = dot(A, G1[:,i1,:])
                    C = dot(B, G2[:,i2,:])
                    T_approx[i0,i1,i2,i3] = dot(C, G3[:,i3])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order5(T, G0, G1, G2, G3, G4, dims, L):
    a, b, c, d, e = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        A = G0[i0,:]
                        B = dot(A, G1[:,i1,:])
                        C = dot(B, G2[:,i2,:])
                        D = dot(C, G3[:,i3,:])
                        T_approx[i0,i1,i2,i3,i4] = dot(D, G4[:,i4])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order6(T, G0, G1, G2, G3, G4, G5, dims, L):
    a, b, c, d, e, f = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            A = G0[i0,:]
                            B = dot(A, G1[:,i1,:])
                            C = dot(B, G2[:,i2,:])
                            D = dot(C, G3[:,i3,:])
                            E = dot(D, G4[:,i4,:])
                            T_approx[i0,i1,i2,i3,i4,i5] = dot(E, G5[:,i5])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order7(T, G0, G1, G2, G3, G4, G5, G6, dims, L):
    a, b, c, d, e, f, g = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                A = G0[i0,:]
                                B = dot(A, G1[:,i1,:])
                                C = dot(B, G2[:,i2,:])
                                D = dot(C, G3[:,i3,:])
                                E = dot(D, G4[:,i4,:])
                                F = dot(E, G5[:,i5,:])
                                T_approx[i0,i1,i2,i3,i4,i5,i6] = dot(F, G6[:,i6])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order8(T, G0, G1, G2, G3, G4, G5, G6, G7, dims, L):
    a, b, c, d, e, f, g, h = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                for i7 in range(h):
                                    A = G0[i0,:]
                                    B = dot(A, G1[:,i1,:])
                                    C = dot(B, G2[:,i2,:])
                                    D = dot(C, G3[:,i3,:])
                                    E = dot(D, G4[:,i4,:])
                                    F = dot(E, G5[:,i5,:])
                                    G = dot(F, G6[:,i6,:])
                                    T_approx[i0,i1,i2,i3,i4,i5,i6,i7] = dot(G, G7[:,i7])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order9(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, dims, L):
    a, b, c, d, e, f, g, h, i = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                for i7 in range(h):
                                    for i8 in range(i):
                                        A = G0[i0,:]
                                        B = dot(A, G1[:,i1,:])
                                        C = dot(B, G2[:,i2,:])
                                        D = dot(C, G3[:,i3,:])
                                        E = dot(D, G4[:,i4,:])
                                        F = dot(E, G5[:,i5,:])
                                        G = dot(F, G6[:,i6,:])
                                        H = dot(G, G7[:,i7,:])
                                        T_approx[i0,i1,i2,i3,i4,i5,i6,i7,i8] = dot(H, G8[:,i8])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order10(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, dims, L):
    a, b, c, d, e, f, g, h, i, j = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                for i7 in range(h):
                                    for i8 in range(i):
                                        for i9 in range(j):
                                            A = G0[i0,:]
                                            B = dot(A, G1[:,i1,:])
                                            C = dot(B, G2[:,i2,:])
                                            D = dot(C, G3[:,i3,:])
                                            E = dot(D, G4[:,i4,:])
                                            F = dot(E, G5[:,i5,:])
                                            G = dot(F, G6[:,i6,:])
                                            H = dot(G, G7[:,i7,:])
                                            I = dot(H, G8[:,i8,:])
                                            T_approx[i0,i1,i2,i3,i4,i5,i6,i7,i8,i9] = dot(I, G9[:,i9])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order11(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, dims, L):
    a, b, c, d, e, f, g, h, i, j, k = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                for i7 in range(h):
                                    for i8 in range(i):
                                        for i9 in range(j):
                                            for i10 in range(k):
                                                A = G0[i0,:]
                                                B = dot(A, G1[:,i1,:])
                                                C = dot(B, G2[:,i2,:])
                                                D = dot(C, G3[:,i3,:])
                                                E = dot(D, G4[:,i4,:])
                                                F = dot(E, G5[:,i5,:])
                                                G = dot(F, G6[:,i6,:])
                                                H = dot(G, G7[:,i7,:])
                                                I = dot(H, G8[:,i8,:])
                                                J = dot(I, G9[:,i9,:])
                                                T_approx[i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10] = dot(J, G10[:,i10])

    return T_approx


@njit(nogil=True, parallel=True)
def tt_error_order12(T, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, dims, L):
    a, b, c, d, e, f, g, h, i, j, k, m = dims
    T_approx = empty(dims, dtype=float64)

    for i0 in prange(a):
        for i1 in range(b):
            for i2 in range(c):
                for i3 in range(d):
                    for i4 in range(e):
                        for i5 in range(f):
                            for i6 in range(g):
                                for i7 in range(h):
                                    for i8 in range(i):
                                        for i9 in range(j):
                                            for i10 in range(k):
                                                for i11 in range(m):
                                                    A = G0[i0,:]
                                                    B = dot(A, G1[:,i1,:])
                                                    C = dot(B, G2[:,i2,:])
                                                    D = dot(C, G3[:,i3,:])
                                                    E = dot(D, G4[:,i4,:])
                                                    F = dot(E, G5[:,i5,:])
                                                    G = dot(F, G6[:,i6,:])
                                                    H = dot(G, G7[:,i7,:])
                                                    I = dot(H, G8[:,i8,:])
                                                    J = dot(I, G9[:,i9,:])
                                                    K = dot(J, G10[:,i10,:])
                                                    T_approx[i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11] = dot(K, G11[:,i11])

    return T_approx


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order3(U, data, S, dims):
    a, b, c = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                s = 0
                for k in range(nnz):
                    s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * data[i0][k])
                S[i0, i1, i2] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order4(U, data, S, dims):
    a, b, c, d = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    s = 0
                    for k in range(nnz):
                        s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * data[i0][k])
                    S[i0, i1, i2, i3] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order5(U, data, S, dims):
    a, b, c, d, e = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        s = 0
                        for k in range(nnz):
                            s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * data[i0][k])
                        S[i0, i1, i2, i3, i4] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order6(U, data, S, dims):
    a, b, c, d, e, f = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            s = 0
                            for k in range(nnz):
                                s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * data[i0][k])
                            S[i0, i1, i2, i3, i4, i5] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order7(U, data, S, dims):
    a, b, c, d, e, f, g = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                s = 0
                                for k in range(nnz):
                                    s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * data[i0][k])
                                S[i0, i1, i2, i3, i4, i5, i6] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order8(U, data, S, dims):
    a, b, c, d, e, f, g, h = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                for i7 in range(h):
                                    i7 = (i7 + i0) % h
                                    s = 0
                                    for k in range(nnz):
                                        s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * U[7][i7,k] * data[i0][k])
                                    S[i0, i1, i2, i3, i4, i5, i6, i7] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order9(U, data, S, dims):
    a, b, c, d, e, f, g, h, i = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                for i7 in range(h):
                                    i7 = (i7 + i0) % h
                                    for i8 in range(i):
                                        i8 = (i8 + i0) % i
                                        s = 0
                                        for k in range(nnz):
                                            s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * U[7][i7,k] * U[8][i8,k] * data[i0][k])
                                        S[i0, i1, i2, i3, i4, i5, i6, i7, i8] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order10(U, data, S, dims):
    a, b, c, d, e, f, g, h, i, j = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                for i7 in range(h):
                                    i7 = (i7 + i0) % h
                                    for i8 in range(i):
                                        i8 = (i8 + i0) % i
                                        for i9 in range(j):
                                            i9 = (i9 + i0) % j
                                            s = 0
                                            for k in range(nnz):
                                                s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * U[7][i7,k] * U[8][i8,k] * U[9][i9,k] * data[i0][k])
                                            S[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order11(U, data, S, dims):
    a, b, c, d, e, f, g, h, i, j, m = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                for i7 in range(h):
                                    i7 = (i7 + i0) % h
                                    for i8 in range(i):
                                        i8 = (i8 + i0) % i
                                        for i9 in range(j):
                                            i9 = (i9 + i0) % j
                                            for i10 in range(m):
                                                i10 = (i10 + i0) % m
                                                s = 0
                                                for k in range(nnz):
                                                    s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * U[7][i7,k] * U[8][i8,k] * U[9][i9,k] * U[10][i10,k] * data[i0][k])
                                                S[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10] = s

    return S


@njit(nogil=True, parallel=True)
def sparse_multilin_mult_order12(U, data, S, dims):
    a, b, c, d, e, f, g, h, i, j, m, n = dims
    nnz = len(data[0])

    for i0 in prange(a):
        for i1 in range(b):
            i1 = (i1 + i0) % b
            for i2 in range(c):
                i2 = (i2 + i0) % c
                for i3 in range(d):
                    i3 = (i3 + i0) % d
                    for i4 in range(e):
                        i4 = (i4 + i0) % e
                        for i5 in range(f):
                            i5 = (i5 + i0) % f
                            for i6 in range(g):
                                i6 = (i6 + i0) % g
                                for i7 in range(h):
                                    i7 = (i7 + i0) % h
                                    for i8 in range(i):
                                        i8 = (i8 + i0) % i
                                        for i9 in range(j):
                                            i9 = (i9 + i0) % j
                                            for i10 in range(m):
                                                i10 = (i10 + i0) % m
                                                for i11 in range(n):
                                                    i11 = (i11 + i0) % n
                                                    s = 0
                                                    for k in range(nnz):
                                                        s += (U[0][i0,k] * U[1][i1,k] * U[2][i2,k] * U[3][i3,k] * U[4][i4,k] * U[5][i5,k] * U[6][i6,k] * U[7][i7,k] * U[8][i8,k] * U[9][i9,k] * U[10][i10,k] * U[11][i11,k] * data[i0][k])
                                                    S[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11] = s

    return S
