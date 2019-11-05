import numpy as np

def load_linear_example1():
    """
    >>> X, Y = load_linear_example1()
    >>> print(X[0])
    [1 4]

    # testing (cont.)
    >>> import regression
    >>> model = regression.LinearRegression()
    >>> print(model.x)
    None

    # testing (cont.)
    >>> import importlib
    >>> importlib.reload(regression)
    <module 'regression' from '/Users/jaejinan/PycharmProjects/exp4/regression.py'>
    >>> model = regression.LinearRegression()
    >>> model.fit(X, Y)
    >>> model.theta
    array([5.30412371, 0.49484536])
    """

    X = np.array([[1,4],[1,8],[1,13],[1,17]])
    Y = np.array([7,10,11,14])

    return X,Y