import numpy as np

from utils import logistic_regression_utils
from utils.logistic_regression_utils import sigmoid


def execute(x_train, y_train):
    initial_w = 0.01 * (np.random.rand(5) - 0.5)
    initial_b = -8
    num_iterations = 10000
    learning_rate = 0.00005

    print("starting bias: " + str(initial_b))
    print("starting weights: " + str(initial_w))

    w, b = logistic_regression_utils.compute_gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate, num_iterations)

    print("after logistic regression bias: " + str(b))
    print("after logistic regression weights: " + str(w))

    return w, b


def predict(x, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
        x : (ndarray Shape (m,n)) features, mXn features
        w : (ndarray Shape (n,))  weights of the model
        b : (scalar)              bias parameter of the model

    Returns:
        p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """

    return logistic_regression_utils.predict(x, w, b)
