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

    # testing (cont.)
    >>> importlib.reload(regression)
    <module 'regression' from '/Users/jaejinan/PycharmProjects/exp4/regression.py'>
    >>> model = regression.LinearRegression()
    >>> model.fit(X, Y)
    >>> model.predict(X)
    array([ 7.28350515,  9.2628866 , 11.7371134 , 13.71649485])

    # testing (cont.)
    >>> importlib.reload(regression)
    <module 'regression' from '/Users/jaejinan/PycharmProjects/exp4/regression.py'>
    >>> model = regression.LinearRegression()
    >>> model.fit(X, Y)
    >>> model.score(X, Y)
    1.2474226804123705
    """


    X = np.array([[1,4],[1,8],[1,13],[1,17]])
    Y = np.array([7,10,11,14])

    return X,Y

def load_nonlinear_example1():
    X = np.array([[1, 0.0], [1, 2.0], [1, 3.9], [1, 4.0]])
    Y = np.array([4.0, 0.0, 3.0, 2.0])
    return X, Y
def polynomial2_features(input):
    """
    # testing
    >>> import datasets
    >>> X, Y = datasets.load_nonlinear_example1()
    >>> ex_X = datasets.polynomial2_features(X)
    >>> ex_X
    array([[ 1.  ,  0.  ,  0.  ],
           [ 1.  ,  2.  ,  4.  ],
           [ 1.  ,  3.9 , 15.21],
           [ 1.  ,  4.  , 16.  ]])
    >>> Y
    array([4., 0., 3., 2.])
    """
    poly2 = input[:,1:]**2
    return np.c_[input, poly2]

def polynomial3_features(input):
    """
    # testing
    >>> import datasets
    >>> X, Y = datasets.load_nonlinear_example1()
    >>> ex_X = datasets.polynomial3_features(X)
    >>> ex_X
    array([[ 1.   ,  0.   ,  0.   ,  0.   ],
           [ 1.   ,  2.   ,  4.   ,  8.   ],
           [ 1.   ,  3.9  , 15.21 , 59.319],
           [ 1.   ,  4.   , 16.   , 64.   ]])
    >>> Y
    array([4., 0., 3., 2.])
    """
    poly2 = input[:,1:]**2
    poly3 = input[:,1:]**3
    return np.c_[input, poly2, poly3]