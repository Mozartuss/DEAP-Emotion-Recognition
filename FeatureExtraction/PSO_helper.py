import warnings
from datetime import datetime

import humanize
import numpy as np
# Create an instance of the classifier
from pyswarms.discrete import BinaryPSO
from sklearn.svm import SVC

warnings.filterwarnings('ignore', 'Solver terminated early.*')

classifier = SVC(max_iter=1000, gamma='auto', random_state=10, kernel="rbf")
x_data = None
y_data = None


# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = x_data
    else:
        X_subset = x_data[:, m == 1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y_data)
    P = (classifier.predict(X_subset) == y_data).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
         + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j


def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    start = datetime.now()
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    end = datetime.now()
    print(humanize.precisedelta(start - end, minimum_unit="seconds", format="%d"))
    return np.array(j)


def pso_main(x, y, options, dimensions, n_particles, n_processes, iterations):
    global x_data, y_data
    x_data = x
    y_data = y
    optimizer = BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(f, iters=iterations, n_processes=n_processes)
    return cost, pos
