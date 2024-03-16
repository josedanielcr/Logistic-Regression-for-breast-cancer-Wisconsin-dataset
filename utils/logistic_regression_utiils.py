import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(x, y, w, b):
    """
    Computes the cost over all examples
    Args:
        x : (ndarray Shape (m,n)) features, mXn features
        y : (ndarray Shape (m,))  target values
        w : (ndarray Shape (n,))  weights of the model
        b : (scalar)              bias parameter of the model
    Returns:
        total_cost : (scalar) cost
    """

    m, n = x.shape

    # calculate the prediction for each example
    z = np.dot(x, w) + b
    f_w_b = sigmoid(z)

    # loss for each model prediction
    total_loss = -y * np.log(f_w_b) - (1 - y) * np.log(1 - f_w_b)

    # calculate the total cost
    return np.sum(total_loss) / m


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for logistic regression
    Args:
        x : (ndarray Shape (m,n)) features, mXn features
        y : (ndarray Shape (m,))  target values
        w : (ndarray Shape (n,))  weights of the model
        b : (scalar)              bias parameter of the model
    Returns
        dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
        dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """

    m, n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0

    for i in range(m):
        # prediction and sigmoid
        z_wb = np.dot(x[i], w) + b
        f_wb = sigmoid(z_wb)

        # calculate error
        error = f_wb - y[i]
        dj_db += error

        for j in range(n):
            dj_dw[j] += error * x[i, j]

    # Average
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def compute_gradient_descent(x, y, w, b, learning_rate, num_iterations):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iterations gradient steps with learning rate alpha

    Args:
        x :    (ndarray Shape (m, n) data, m examples by n features
        y :    (ndarray Shape (m,))  target value
        w : (ndarray Shape (n,))  Initial values of parameters of the model
        b : (scalar)              Initial value of parameter of the model
        learning_rate : (float)              Learning rate
        num_iterations : (int)            number of iterations to run gradient descent

    Returns:
        w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
        b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    m = len(x)
