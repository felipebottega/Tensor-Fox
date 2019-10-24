import numpy as np
import TensorFox as tfx


def create(m, n, p, r, c, var):
    """
    This function generates tensors with double bottlenecks.
    m, n, p are the dimensions of the tensor and r is the rank.
    0 < c < 1 is the collinearity level.
    The higher is the collinearity degree, lower will be the rank of the tensor. 
    Caution is adviced to not generating tensor with small c and big r.
    """
    
    T = np.zeros((m, n, p))
    
    M_X = np.random.randn(m, r)
    Q_X, R_X = np.linalg.qr(M_X)
    
    M_Y = np.random.randn(n, r)
    Q_Y, R_Y = np.linalg.qr(M_Y)
    
    M_Z = np.random.randn(p, r)
    Q_Z, R_Z = np.linalg.qr(M_Z)


    X = np.zeros((m, r))
    Y = np.zeros((n, r))
    Z = np.zeros((p, r))
    for l in range(r):
        if l==0 or l==1:
            X[:, l] = Q_X[:, 0] + c*Q_X[:, l]
            Y[:, l] = Q_Y[:, 1] + c*Q_Y[:, l]
            Z[:, l] = Q_Z[:, 2] + c*Q_Z[:, l]
        else:
            X[:, l] = Q_X[:, l]
            Y[:, l] = Q_Y[:, l]
            Z[:, l] = Q_Z[:, l]
    
    T = tfx.cnv.cpd2tens([X, Y, Z])    
    T_noise = T + var*np.random.randn(m, n, p)

    return T, T_noise
