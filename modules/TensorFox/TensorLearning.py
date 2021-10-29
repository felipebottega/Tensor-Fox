import numpy as np
from numpy import prod, uint64
import sys
import matplotlib.pyplot as plt
from numba import njit, prange
import TensorFox.TensorFox as tfx


# PREPROCESSING


def normalization(X, method='mean_normalization'):
    """
    Normalize data by shifting all data by a certain constant, and dividing all data by a other certain constant. The
    program returns the normalized data X = (X - mu)/sigma. In the case the user want to return to the original format,
    set X = sigma*X + mu.

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    method: string
        There are 4 available methods of feature scaling.
        min_max_normalization: X = (X - min(X)) / (max(X) - min(X))
        mean_normalization: X = (X - mean(X)) / (max(X) - min(X))
        standardization: X = (X - mean(X)) / sqrt(var(X))
        unit_length: X = X / norm(X), where norm() is the Frobenius norm

    Outputs
    -------
    X_new: 2D array
        Transformed data after normalization, that is, X_new = (X - mu)/sigma.
    mu: 2D array
        The value which X is subtracted.
    sigma: float
        The value which X will be divided. In the case max_unit and var_unit are False, then sigma = 1.
    """

    if method == 'min_max_normalization':
        mu = np.min(X)
        sigma = np.max(X) - np.min(X)
    elif method == 'mean_normalization':
        mu = np.mean(X)
        sigma = np.max(X) - np.min(X)
    elif method == 'standardization':
        mu = np.mean(X)
        sigma = np.sqrt(np.var(X))
    elif method == 'unit_length':
        mu = 0
        sigma = np.linalg.norm(X)

    X_new = (X - mu)/sigma

    return X_new, mu, sigma


def pca(X, p):
    """
    Make principal component analysis of the data. In the case the user want to return to the original format, set
    X = dot(X, U.T) .

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    p: Float
        Float between 0 and 1. The program compress the data so that 100*p % is retained. For example, if p = 0.90, then
        90 % of the data is retained after compression.
        
    Outputs
    -------
    X_new: 2D array
        Transformed data after PCA.
    U: 2D array
        Left singular vectors of the SVD of the covariance matrix of X.
    """

    # Number of samples and size of inputs
    num_samples, n = X.shape
    # Covariance matrix of X
    cov = (1 / num_samples) * np.dot(X.T, X)
    # SVD of covariance matrix
    U, S, Vt = np.linalg.svd(cov)

    # Compute size k <= n of compression
    total_energy = np.sum(S)
    for k in range(n):
        energy = np.sum(S[0:k])
        ratio = energy / total_energy
        if ratio > p:
            break

    # Compute compressed inputs
    U = U[:, :k]
    X_new = np.dot(X, U)

    return X_new, U


def prepare_data(X, p):
    """
    Before the starting the training stage, you will need to prepare your data, that is, make the PCA, construct the one
    hot encoded classes, adding bias, and so on. This function and the function normalize_data are close since both
    prepare the data for training.

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    p: Float
        Float between 0 and 1. The program compress the data so that 100*p % is retained. For example, if p = 0.90, then
        90 % of the data is retained after compression.
        
    Outputs
    -------
    X_bias: 2D array
        Train data prepared to be used for training.
    U: 2D array
        Left singular vectors of the SVD of the covariance matrix of X.
    """

    # PCA of inputs
    X_new, U = pca(X, p)
    num_samples, n = X_new.shape

    # Add bias
    aux = np.ones((num_samples, n + 1))
    for i in range(num_samples):
        aux[i, 1:] = X_new[i, :]
    X_bias = aux

    return X_bias, U


def hot_encoded_target(Y):
    """
    If the labels are not hot encoded, then this function is mandatory, otherwise we get errors. We are using one hot
    encoded outputs of the form [1,0,0,...,0], [0,1,0,...,0], ..., [0,0,...,0,1]. To make the conversion the program
    assumes the classes (the entries of Y) are numbers, from 0 to m-1 (so we have m classes).

    Inputs
    ------
    Y: 1D array
        Each entry i of Y correspond to the class of the ith input (ith row of X).

    Outputs
    -------
    Y_hot_encoded: 2D array
        Train data classes with hot encoding.
    """

    num_samples = Y.size
    m = int(max(Y)) + 1
    Y_hot_encoded = np.zeros((num_samples, m))
    I = np.identity(m)

    for i in range(num_samples):
        for j in range(m):
            if Y[i] == j:
                Y_hot_encoded[i, :] = I[j, :]

    return Y_hot_encoded


def norm_pca(X_new, U, mu, sigma):
    """
    After normalizing and performing pca over the train dataset, we may be interested in making predictions about some
    new inputs. In any case, normalizing and compressing in the same way it is necessary in order to get meaningful
    results.

    Inputs
    ------
    X_new: 2D array
        Input data in 2D array format. Each row correspond to a single sample of the data.
    U: 2D array
         Left singular vectors of the SVD of the covariance matrix of the train dataset.
    mu: 2D array
        Mean of the train dataset.
    sigma: float
        The value in which the dataset was divided.

    Outputs
    -------
    X_new: 2D array
        Input data after transformations.
    """

    X_new = (X_new - mu) / sigma
    X_new = np.dot(X_new, U)

    return X_new


# CPD LEARNING


def init_W(m, L, n, R):
    """
    Generate initial weights to start iterating. We have that W[k] = [W1,...,WL] is a multidimensional array of shape
    L x n x R. For each tensor Tk of the model we have that

                Tk = sum_{r=1}^R W[k, 0, :, r] ⊗ W[k, 1, :, r] ⊗ ... ⊗ W[k, L-1, :, r].

    Note that each W[k, l, :, :] is a n x R factor matrix of the CPD of Tk, in other words, W[k] is the set of the
    factor matrices of Tk. Each tensor Tk has shape n x n x ... x n (L times) and has rank <= R.

    Inputs
    ------
    m: int
        Number of tensors of the model, which must coincide with the dimension of the output space.
    L: int
        Order of the tensors.
    n: int
        Dimension of the input space.
    R: int
        Rank of the tensors.

    Outputs
    -------
    W: 4D array
    """

    W = np.zeros((m, L, n, R))

    for k in range(m):
        for l in range(L):
            W[k, l, :, :] = np.random.uniform(-1, 1, size=(n, R))

    return W


def dot_products(x, W, dropout):
    """
    Function to compute all the dot products between an input x and the weights.

    Inputs
    ------
    x: 1D array
        A single input sample.
    W: L-D array
        Coordinates of all tensors Tk of the model (see function init_W for more details).
    dropout:
        Probability to use or not dropout (see function cpd_train for more details).

    Outputs
    -------
    dot_prods: 3D array
        Each value dot_prods[k, l, r] is the dot product between x and the weight vector corresponding to the r-th
        column of the l-th factor matrix of the k-th tensor.
    """

    # Initialize array to receive all dot products
    m, L, n, R = W.shape
    dot_prods = np.ones((m, L, R))

    # Compute all the dot products.
    lst = [W[k, :, :, r] for k in range(m) for r in range(R)]
    W_tmp = np.concatenate(lst)
    Wx = np.dot(W_tmp, x)
    Wx = Wx.reshape(m, R, L)
    for k in range(m):
        dot_prods[k, :, :] = Wx[k, :, :].T

    # The above is an optimized form of the calculations commented below.
    #for k in range(m):
    #    for r in range(R):
    #        dot_prods[k, :, r] = np.dot(W[k, :, :, r], x)

    # Apply dropout.
    if dropout < 1:
        dropout_decision = np.random.uniform(0, 1, size=(m, R))
        dropout_decision = dropout_decision <= dropout
        for k in range(m):
            for r in np.arange(R)[~dropout_decision[k, :]]:
                dot_prods[k, :, r] *= 0

    return dot_prods


@njit(nogil=True)
def h(W, dot_prods):
    """
    Function to compute the hypothesis function h(x). This is defined by

                h(x) = f(T1(x,...,x),...,Tm(x,...,x)) = ( f(T1(x,...,x)) ,..., f(Tm(x,...,x)) ),

    where f is the activation function (in this case we are using the sigmoid function). Note that h(x) is a
    m-dimensional vector.

    Inputs
    ------
    W: 4D array
        Coordinates of all tensors Tk of the model (see function init_W for more details).
    dot_prods: 3D array.
        Dot products between the weights and x (see function dot_products for more details).

    Outputs
    -------
    hx: 1D array
        Evaluation of h(x).
    dhx: 1D array
        Evaluation of the derivative of h at x, that is, h'(x).
    """

    m, L, n, R = W.shape
    # Initialize vector to receive T(x,...,x)
    hx = np.zeros(m, dtype=np.float64)

    # Compute Tk(x,...,x) for all k=1...m
    for k in range(m):
        for r in range(R):
            temp = 1
            for l in range(L):
                temp *= dot_prods[k, l, r]
            hx[k] += temp

    # Compute (f'(T1(x,...,x),...,f'(Tm(x,...,x))
    dhx = df(hx)
    # Compute f(T1(x,...,x),...,Tm(x,...,x))
    hx = f(hx)

    return hx, dhx


@njit(nogil=True)
def f(z):
    """
    Non-linear (activation) function f at a point z(we are using the sigmoid function here).

    Inputs
    ------
    z: 1D array

    Outputs
    -------
    fz: 1D array
    """

    fz = 1/(1 + np.exp(-z))
    return fz


@njit(nogil=True)
def df(z):
    """
    Derivative of the non-linear (activation) function f at a point z.

    Inputs
    ------
    z: 1D array

    Outputs
    -------
    dfz: 1D array
    """

    dfz = f(z)*(1 - f(z))
    return dfz


@njit(nogil=True, parallel=True)
def grad(x, y, W, dot_prods, Lambda):
    """
    Let J(W) be the cost function of the problem, and grad(J) be the gradient of J. Then this function constructs
    grad(J)_{k, l, i, r} with respect to a pair input-output (x, y). In the end, the partial derivative of J with
    respect to W[k, l, i, r] is the sum of grad(J)_{k, l, i, r}(x, y) for all pairs (x, y).

    Inputs
    ------
    x: 1D array
    y: 1D array
    W: 4D array
    dot_prods: 3D array
    Lambda: float
        Regularization parameter for the cost function. We must have Lambda >= 0.

    Outputs
    -------
    grad_J: 4D array
    """

    m, L, n, R = W.shape
    # Initialize gradient vector associated to J at x
    grad_J = np.zeros((m, L, n, R))
    # Compute hypothesis and its derivative at x
    hx, dhx = h(W, dot_prods)

    for k in prange(m):
        for l in range(L):
            for i in range(n):
                for r in range(R):
                    grad_J[k, l, i, r] = compute_grad(x, y, W, dot_prods, hx, dhx, Lambda, L, k, l, i, r)

    return grad_J


@njit(nogil=True)
def compute_grad(x, y, W, dot_prods, hx, dhx, Lambda, L, k, l, i, r):
    """
    Auxiliar function to make inner computations of function grad.
    """

    term1 = Lambda * W[k, l, i, r]
    term2 = hx[k] - y[k]
    term3 = dhx[k]

    temp = 1
    for lt in range(L):
        if lt != l:
            temp *= dot_prods[k, lt, r]
    term4 = x[i] * temp

    return term1 + term2*term3*term4


def update(alpha, W, grad_J, constraint):
    """
    Compute the update stage W = W - alpha*grad_J for the weights in the gradient descent algorithm.

    Inputs
    ------
    alpha: float
        Step parameter of the gradient descent algorithm. We must have alpha > 0.
    W: 4D array
    grad_J: 4D array
        See the description of grad_J in the function grad.
    constraint: float
        This paramater limits the size of the weights with the update formula W = constraint * W/max(abs(W)). This way
        the largest magnitude of W is always equal to this parameter.

    Outputs
    -------
    W: 4D array
        Update of W.
    """

    W = W - alpha * grad_J

    if constraint != np.inf:
        W = constraint * W / np.max(np.abs(W))

    return W


def find_class(x, W):
    """
    Given a set of weights W and a new input x, this function computes the predicted class of x.

    Inputs
    ------
    x: 1D array
        New input for which the program will try to predict its class.
    W:
        Set of weights after the learning stage.

    Outputs
    -------
    x_class: int
        At the moment the program is limited to classification problems. So each prediction is a number (of a class).
    """

    dot_prods = dot_products(x, W, 1)
    hx, dhx = h(W, dot_prods)
    x_class = np.argmax(hx)

    return x_class


def cpd_train(X, Y, X_val, Y_val, W, alpha=0.01, alpha_decay=0.5, Lambda=0.1, epochs=10, batch=1, constraint=np.inf, dropout=1, display=True):
    """
    Function to start the training stage.

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    Y: 1D array
        Each entry i of Y correspond to the original class of the ith input (ith row of X).
    X_val: 2D array
        Piece of train dataset used as validation input set. If no validation is intended, set X_val to nan.
    Y_val: 1D array
        Piece of train dataset used as validation target set.
    W: 4D array
        Initial set of weights (see function init_W for more details).
    alpha: float
        Step parameter of the gradient descent algorithm. We must have alpha > 0.
    alpha_decay: float
        The more the program is closer to the optimum, the more it is necessary to take smaller steps. In this case it
        is interesting to decrease alpha gradually. At each epoch we update alpha with alpha = alpha_decay * alpha.
        Default is alpha_decay = 0.5.
    Lambda: float
        Regularization parameter for the cost function. We must have Lambda >= 0. Default is Lambda = 0.1.
    epochs: int
        Desired number of epochs to use in the training stage. Default is 10 epochs.
    batch: int
        Size of the batch in the learning algorithm. After passing through batch inputs and accumulating their costs,
        the corresponding gradient is used to make the next step. Default is 1 batch.
    constraint: float
        Large weights size can be a sign of an unstable network. Ona may force the magnitude of all weights to be below
        a specified value, given by this parameter. At each iteration the program applies the update formula
        W = constraint * W/max(abs(W)). This way the largest magnitude of W is always equal to this parameter. Default
        is constraint = np.inf, which means to not use constraints.
    dropout: float
        This value should be between 1 (no dropout is used) and 0 (drop is used for all nodes). At each iteration the
        program decides, for each node, if it will be used or not, based on this probability. Default is dropout = 1.
    display: bool
        If set to True (default), the program displays some relevant results and plots about the training stage.

    Outputs
    -------
    W: 4D array
        Optimized set of weights.
    accuracy: list
        Evolution of the accuracy (0 % to 100 %) at each iteration.
    cost_function: list
        Evolution of the cost function at each iteration.
    success: int
        Total number of successes after the training.
    """

    # Verify if the parameter dropout is consistent.
    if dropout < 0 or dropout > 1:
        sys.exit('The parameter dropout has to be between 0 ans 1.')
    elif dropout == 0:
        sys.exit('The parameter dropout has to be bigger than 0.')

    # Number of samples.
    num_samples, n = X.shape

    # Transform target into hot encoded form.
    Y_hot_encoded = hot_encoded_target(Y)

    # Identity matrix with size of inputs minus bias.
    I = np.identity(n-1)

    # Create vectors to visualize some information later.
    accuracy = []
    accuracy_val = []
    cost_function = []

    for ep in range(epochs):
        count = 0
        success = 0
        cum_cost = 0
        cum_grad = np.zeros(W.shape)

        for j in range(num_samples):
            x = X[j, :]
            y = Y_hot_encoded[j, :]
            dot_prods = dot_products(x, W, dropout)
            grad_J = grad(x, y, W, dot_prods, Lambda)
            cum_grad += grad_J

            # Update weights after seeing batch inputs.
            if (j % batch == 0) and (j >= batch):
                W = update(alpha, W, cum_grad/batch, constraint)
                cum_grad = 0*cum_grad
            elif j == num_samples-1:
                W = update(alpha, W, cum_grad/batch, constraint)

            # Predict class of current input.
            x_class = find_class(x, W)
            if x_class == Y[j]:
                success += 1

            # Save the value of the cost function with respect to this individual input.
            hx, dhx = h(W, dot_prods)
            cost = (1 / 2) * np.linalg.norm(hx - y) ** 2
            cum_cost += cost

            # Display progress bar.
            if j % (num_samples//80) == 0 and j >= (num_samples//80):
                count += 1
                s = "Epoch " + str(ep+1) + ": [" + min(80, count) * "=" + min((80-count), 80) * " " + "]" + \
                    " " + str(np.round(100*j/num_samples, 1)) + "%"
                sys.stdout.write('\r' + s)

        # Display progress bar at 100%.
        s = "Epoch " + str(ep+1) + ": [" + min(80, count) * "=" + min((80-count), 80) * " " + "]" + " " + "100.0% / acc=" + str(round(100 * success/num_samples, 2)) + '%'
        sys.stdout.write('\r' + s)

        # Update relevant information.
        acc = 100 * success/num_samples
        accuracy.append(100 * success/num_samples)
        cost_function.append(cum_cost/num_samples)

        # Update alpha.
        alpha = alpha_decay * alpha

        # The X_val dataset is supposed to be already normalized.
        if np.sum(np.isnan(X_val)) == 0:
            accuracy_val.append(cpd_test(X_val, Y_val, W, I))
        print()

    if display:
        disp_results(accuracy, accuracy_val, cost_function, W, epochs)

    return W, accuracy, accuracy_val, cost_function, success


def disp_results(accuracy, accuracy_val, cost_function, W, epochs):
    """
    Display results of the training stage.

    Inputs
    ------
    accuracy: list
        Evolution of the accuracy (0 % to 100 %) at each iteration.
    accuracy_val list
        Evolution of the validation accuracy (0 % to 100 %) at each iteration.
    cost_function: list
        Evolution of the cost function at each iteration.
    W: 4D array
        Initial set of weights (see function init_W for more details).
    epochs: int
        Number desired of epochs to use in the training stage.
    """

    m, L, n, R = W.shape
    print()
    print('Dimensions of input:', n)
    print('Dimensions of target:', m)
    print('Rank:', R)
    print('Number of factor matrices:', L)
    print('Number of epochs:', epochs)
    print('Number of weights:', ((L - 1) * n + m) * R)
    print('Accuracy over training dataset:', np.round(accuracy[-1], 3), '%')
    if len(accuracy_val) > 0:
        print('Accuracy over validation dataset:', np.round(accuracy_val[-1], 3), '%')
    print()

    plt.figure(figsize=[10, 4])
    plt.plot(np.arange(1, epochs+1), accuracy, markersize=2, label='train')
    if len(accuracy_val) > 0:
        plt.plot(np.arange(1, epochs + 1), accuracy_val, markersize=2, label='validation')
    plt.title('Learning Curve')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

    plt.figure(figsize=[10, 4])
    plt.plot(np.arange(1, epochs+1), cost_function, 'r', markersize=2)
    plt.title('Cost function')
    plt.grid()
    plt.xlabel('Epoch')
    plt.show()

    return


def cpd_test(X_test, Y_test, W, U, mu=0, sigma=1):
    """
    After the training stage it is time to validate the model with the test dataset. This function is responsible for
    that part. The default values for mu and sigma assume that no normalization was performed. Additionally, if no PCA
    was also performed, pass the matrix U as np.identity(X.shape[1]).

    Inputs
    ------
    X_test: 2D array
        Test data in 2D array format. Each row correspond to a single sample of the test data.
    Y_test:
        Each entry i of Y_test correspond to the class of the ith input of the test dataset (ith row of X_test).
    W: 4D array
        Weights obtained after the training stage.
    U: 2D array
        Left singular vectors of the SVD of the covariance matrix of the normalized X_test. The compressed data is given
        by dot(X_test_new, U), where X_test_new is the normalized version of X_test.
    mu, sigma: float
        mu and sigma are such the normalized data is given by (X_test - mu)/sigma.
        
    Outputs
    -------
    accuracy: float
        Accuracy of the model over the test dataset.
    """
    
    predictions = cpd_predictions(X_test, W, U, mu=mu, sigma=sigma)
    
    num_samples = X_test.shape[0]
    success = 0
    for j in range(num_samples):
        x_class = predictions[j]
        y = Y_test[j]
        if x_class == y:
            success += 1

    accuracy = 100 * success/num_samples

    return accuracy


def cpd_predictions(X_test, W, U, mu=0, sigma=1):
    """
    Given set of new inputs X_test, this function makes the corresponding predictions. The default values for mu and
    sigma assume that no normalization was performed. Additionally, if no PCA was also performed, pass the matrix U as
    np.identity(X.shape[1]).

    Inputs
    ------
    X_test: 2D array
        Test data in 2D array format. Each row correspond to a single sample of the test data.
    W: 4D array
        Weights obtained after the training stage.
    U: 2D array
        Left singular vectors of the SVD of the covariance matrix of the normalized X_test. The compressed data is given
        by dot(X_test_new, U), where X_test_new is the normalized version of X_test.
    mu, sigma: float
        mu and sigma are such the normalized data is given by (X_test - mu)/sigma.
        
    Outputs
    -------
    predictions: 1D array
        Predictions of the classes of each sample in X_test.
    """

    # Transform data to compressed form. This is necessary since the weights W were obtained over compressed data.
    X_test_new = norm_pca(X_test, U, mu, sigma)
    num_samples, n = X_test_new.shape

    # Add bias
    aux = np.ones((num_samples, n + 1))
    aux[:, 1:] = X_test_new
    X_bias = aux

    # Create array of predictions, which are integers from 0 to m-1.
    predictions = np.zeros(num_samples, dtype=np.int64)

    for j in range(num_samples):
        x = X_bias[j, :]
        x_class = find_class(x, W)
        predictions[j] = x_class

    return predictions


def simplify_model(W, r, options=False):
    """
    With the set of weights W, obtained after training and testing, we can construct the corresponding coordinate
    tensors and compute a rank-r CPD for them. The idea is to simplify the model, using less rank-1 terms.

    Inputs
    ------
    W: 4D array
        Weights obtained after the training stage.
    r: int
        Desired rank.
    options: class
        Set of options to be used in the CPD computation (see the Tensor Fox documentation for more details).
        
    Outputs
    -------
    new_W: float 4D-ndarray
        New model based in rank-r approximations. An array with the same shape as W, except for the last dimension.
    errors: list
        Each errors[k] is the relative error between the rank-r approximation and Tk.
    """

    # Initialize variables.
    m, L, n, R = W.shape
    new_W = np.zeros((m, L, n, r))
    errors = []

    for k in range(m):
        # Construct list of factor matrices for Tk.
        factors = []
        for l in range(L):
            factors.append(W[k, l, :, :])
        # Convert factor matrices to coordinates format.
        Tk = tfx.cnv.cpd2tens(factors)
        # Compute rank-r approximation for Tk.
        factors, output = tfx.cpd(Tk, r, options)
        errors.append(output.rel_error)
        for l in range(L):
            new_W[k, l, :, :] = factors[l]

    return new_W, errors


# MLSVD LEARNING


def create_sets(X, Y, num_samples, p, var_type, display):
    """
    This function separates the inputs by class in a list called samples_per_class. If the number of inputs per 
    class is no equal, the program randomly select some inputs to be repeated in a certain class. This is done 
    so that all classes have the same number of elements. 
    There is the option to add noise to these repeated inputs, so we can have a simple method of data augmentation
    to apply.

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    Y: 1D array
        Each entry i of Y correspond to the class of the ith input (ith row of X).
    num_samples: int
        The desired number of samples per class. This number should be equal or greater than the current maximum number
    of samples in a class.
    p: float
        Proportion of the noise with respect to the original data. For example, if p=0.1, then the noise added has
        size less or equal than 10% of the size of the original data.  
    var_type: str
        This variable indicates the type or perturbation. In both cases we use uniform distribution to pick the
        perturbations. The only possibilities considered are 'int' and 'float'. We remark that you should use float 
        whenever the data is normalized.
    display: bool
        If True (default), the function shows the number of inputs for each class.
        
    Outputs
    -------
    X_new, Y_new: arrays
        New versions of X and Y, maybe with more data.
    inputs: list
        Each element of this list is a list with all inputs with same class.
    """

    total_num_samples = X.shape[0]
    num_classes = int(max(Y)) + 1
    X_new = []
    Y_new = []
    inputs = [[] for i in range(num_classes)]

    # Create list with all inputs separated by class.
    for i in range(total_num_samples):
        x = X[i, :]
        y = int(Y[i])
        inputs[y].append(x)

    # Counting of inputs for each class.
    samples_per_class = []
    for i in range(num_classes):
        samples_per_class.append(len(inputs[i]))
        if display:
            print('Inputs of class', i, '=', len(inputs[i]))

    # Make each class to have the same number of inputs.
    if num_samples is not None:
        max_class = max(max(samples_per_class), num_samples)
    else:
        max_class = max(samples_per_class)

    for i in range(num_classes):
        c = samples_per_class[i]
        diff = max_class - c
        if diff > 1:
            for j in range(diff):
                idx = np.random.randint(c)
                x = inputs[i][idx]

                # Apply data augmentation.
                xp = np.random.uniform(low=min(x), high=1+max(x), size=x.shape[0])
                if var_type == 'float':
                    x_new = x + p*xp
                elif var_type == 'int':
                    x_new = x + (p*xp).astype(int)               
                inputs[i].append(x_new)
          
                # Update X_new and Y_new.
                X_new.append(x_new)
                Y_new.append(i)             
                
    # Convert X_new and Y_new to arrays.
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    X_new = np.concatenate([X, X_new])
    Y_new = np.concatenate([Y, Y_new])

    if display:
        print()
        print('After fixing number of inputs per class:')
        for i in range(num_classes):
            print('Inputs of class', i, '=', len(inputs[i]))

    return X_new, Y_new, inputs


def data2tens(X, Y, num_samples=None, p=0, var_type='float', display=True):
    """
    Given the input dataset X and the target dataset Y, this function separates the inputs by class so that each
    class has the same number of inputs m. If n is the dimension of the inputs and each class has p inputs, the
    tensor created has shape m x n x p. Each frontal slice contains all inputs corresponding to a certain class.

    Inputs
    ------
    X: 2D array
        Train data in 2D array format. Each row correspond to a single sample of the data.
    Y: 1D array
        Each entry i of Y correspond to the class of the ith input (ith row of X).
    num_samples: int
        The desired number of samples per class. This number should be equal or greater than the current maximum number
    of samples in a class.
    p: float
        Proportion of the noise with respect to the original data. For example, if p=0.1, then the noise added has
        size less or equal than 10% of the size of the original data. Default is p=0, which means to not perturb. 
    var_type: str
        This variable indicates the type or perturbation. In both cases we use uniform distribution to pick the
        perturbations. The only possibilities considered are 'int' and 'float'. Default is 'float'. We remark that you 
        should use float whenever the data is normalized.
    display: bool
        If True (default), the function shows the number of inputs for each class.
        
    Outputs
    -------
    T: 3D array
        Each vector T[i, :, k] is the ith input of the kth class.
    X_new, Y_new: arrays
        New versions of X and Y, maybe with more data.
    """

    # Create list with inputs organized by class.
    X, Y, inputs = create_sets(X, Y, num_samples, p, var_type, display)
    num_inputs = len(inputs[0])
    input_size = inputs[0][0].size
    num_classes = len(inputs)

    # Create tensor.
    T = np.zeros((num_inputs, input_size, num_classes))
    for i in range(num_inputs):
        for k in range(num_classes):
            T[i, :, k] = np.array(inputs[k][i])

    return T, X, Y


def mlsvd_train(T, r, options=False):
    """
    Compute MLSVD (Multilinear Singular Value Decomposition) of T, a tensor of shape m x n x p. More precisely, we have
    that T = (U1, U2, U3)*S (in fact, S is truncated so this is an approximation), where S is a R1 x R2 x R3 tensor (we
    have that R3 = p, that is, the number of classes can't change), S_energy is the total energy retained after
    truncating the original MLSVD, U1 is a m x R1 matrix, U2 is a n x R2 matrix, U3 is a p x R3 matrix, sigma1 is a
    vector with the first R1 singular values of T1 (first unfolding), sigma2 is a vector with the first R2 singular
    values of T2, sigma3 is a vector with the first R3 singular values of T3, rel_error = | T - (U1, U2, U3)*S | / |T|
    is the relative error of the truncated MLSVD.

    Inputs
    ------
    T: 3D array
        Each vector T[i, :, k] is the ith input of the kth class.
    r: int
        Rank used to compute the MLSVD. See the function mlsvd in Tensor Fox for more details about this parameter. If a
        specific truncation trunc_dims is passed, the parameter r is irrelevant.
    options: class
        Options to the computation of the MLSVD. See more in the Tensor Fox documentation. If set to False, the default 
        options are used.
        
    Outputs
    -------
    F: 3D array
        Each vector F[i, :, k] is the compressed ith input of the kth class. We have F = (I, I, U3)*S =
        = (U1.T, U2.T, I)*T, where is the identity matrix.
    U2: 2D array
        Second orthogonal matrix of the MLSVD of T.
    success: bool
        success is False only if the last dimension of T was trancated. In this case the proccess must be repeated.
    """

    print('Training model...')
    m, n, p = T.shape
    num_samples, num_classes = m, p
    print('Original shape:', T.shape)

    # Set options
    options = tfx.aux.make_options(options)

    Tsize = np.linalg.norm(T)
    if options.display == 3:
        S, U, UT, sigmas, rel_error = tfx.cmpr.mlsvd(T, Tsize, r, options)       
    else:
        S, U, UT, sigmas = tfx.cmpr.mlsvd(T, Tsize, r, options)

    R1, R2, R3 = S.shape
    U1, U2, U3 = U

    # Sometimes the MLSVD may truncate the last dimension p. We have to repeat the training in this case.
    if R3 != num_classes:
        print('The number of classes was truncated. This cannot happen. Try again, but if the problem persists, we '
              'recommend to manually pass the truncation with the parameter `trunc_dims`.')
        success = False
        F, U2 = 0, 0
    else:
        success = True
        S1 = tfx.cnv.unfold(S, 1)
        F = tfx.mlinalg.multilin_mult([np.identity(R1), np.identity(R2), U3], S1, S.shape)
        print('MLSVD shape:', F.shape)
        if options.display == 3:
            print('Error:', rel_error)
        print('Working with', np.round(100*int(prod(F.shape, dtype=uint64))/int(prod(T.shape, dtype=uint64)), 4), '% of the original size')
        print()

    return F, U2, success


def mlsvd_test(X_test, Y_test, F, U2, p, mu=0, sigma=1):
    """
    After the training stage it is time to validate the model with the test dataset. This function is responsible for
    that part.

    Inputs
    ------
    X_test: 2D array
        Test data in 2D array format. Each row correspond to a single sample of the test data.
    Y_test:
        Each entry i of Y_test correspond to the class of the ith input of the test dataset (ith row of X_test).
    F: 3D array
        Same tensor of mlsvd_train function.
    U2: 2D array
        Same matrix of mlsvd_train function.
    p: Float
        Float between 0 and 1. The program compress the data so that 100*p % is retained. For example, if
        p = 0.90, then 90 % of the data is retained after compression. This compression is used in each slice 
        of F to make the predictions.
    mu, sigma: float
        mu and sigma are such the normalized data is given by (X_test - mu)/sigma.
        
    Outputs
    -------
    accuracy: float
        Accuracy of the model over the test dataset.
    """
    
    predictions = mlsvd_predictions(X_test, F, U2, p, mu=mu, sigma=sigma)
    
    num_samples = X_test.shape[0]
    success = 0
    for j in range(num_samples):
        x_class = predictions[j]
        y = Y_test[j]
        if x_class == y:
            success += 1
    
    accuracy = 100 * success/num_samples

    return accuracy


def mlsvd_predictions(X_test, F, U2, p, mu=0, sigma=1):
    """
    In a more geometric interpretation, first we consider the space generated by the columns associated to the digit d,
    then we check the distance between z_new and this space. This distance is the norm associated with the solution of
    the least squares problem. After choosing d, this choice means z_new is more close to the space associated to the
    digit d.

    Inputs
    ------
    X_test: 2D array
        Test data in 2D array format. Each row correspond to a single sample of the test data.
    F: 3D array
        Same tensor of mlsvd_train function.
    U2: 2D array
        Same matrix of mlsvd_train function.
    p: Float
        Float between 0 and 1. The program compress the data so that 100*p % is retained. For example, if
        p = 0.90, then 90 % of the data is retained after compression. This compression is used in each slice 
        of F to make the predictions.
    mu, sigma: float
        mu and sigma are such the normalized data is given by (X_test - mu)/sigma.
        
    Outputs
    -------
    predictions: 1D array
        Predictions of the classes of each sample in X_test.
    """

    print('Computing predictions...')
    X_test_new = (X_test - mu)/sigma
    num_samples, num_classes = X_test_new.shape[0], F.shape[2]
    predictions = np.zeros(num_samples, dtype=np.int64)

    # Compute truncated SVD of slices of F.
    F_slices_svd = []
    for d in range(num_classes):
        U, S, Vt = np.linalg.svd(F[:, :, d])        
        # Truncate U
        n = S.size
        total_energy = np.sum(S)
        for k in range(n):
            energy = np.sum(S[0:k])
            ratio = energy / total_energy
            if ratio > p:
                break
        F_slices_svd.append(Vt[:k, :].T)

    # Make predictions.    
    for i in range(num_samples):
        z = X_test_new[i, :]
        z_new = np.dot(U2.T, z)
        best_dist = np.inf
        for d in range(num_classes):
            x = np.dot(F_slices_svd[d].T, z_new)
            # Distance between z' and the space associated to the digit d.
            dist = np.linalg.norm(np.dot(F_slices_svd[d], x) - z_new)
            if dist < best_dist:
                best_class = d
                best_dist = dist

        # Save results.
        predictions[i] = best_class

    print('Finished')

    return predictions

