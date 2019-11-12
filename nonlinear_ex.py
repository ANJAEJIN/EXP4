import numpy as np
import datasets
import regression
import matplotlib.pyplot as plt

for _alpha in {0,0.1,0.5,1.0,10.0}:
    X, Y = datasets.load_nonlinear_example1()
    ex_X = datasets.polynomial3_features(X)

    model = regression.RidgeRegression(alpha=_alpha)
    model.fit(ex_X, Y)

    samples = np.arange(0, 4, 0.1)
    x_samples = np.c_[ np.ones(len(samples)), samples ]
    ex_x_samples = datasets.polynomial3_features(x_samples)

    plt.plot(samples, model.predict(ex_x_samples),label='alpha = '+str(_alpha))

plt.scatter(X[:, 1], Y)
plt.legend()

plt.show()