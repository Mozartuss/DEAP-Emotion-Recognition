import warnings

import numpy as np
from numpy.random import rand
# error rate
from sklearn.svm import SVC

warnings.filterwarnings('ignore', 'Solver terminated early.*')


def error_rate(x, opts):
    # parameters
    fold = opts['fold']
    xt = fold['xt']
    yt = fold['yt']
    xv = fold['xv']
    yv = fold['yv']

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt.reshape(num_train)  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv.reshape(num_valid)  # Solve bug
    # Training
    mdl = SVC(max_iter=1000, gamma='auto', random_state=10, kernel="rbf")
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    acc = np.sum(yvalid == ypred) / num_valid
    error = 1 - acc

    return error


# Error rate & Feature size
def Fun(x, opts):
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        error = error_rate(x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]

    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    return V, Vmax, Vmin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def jfs(xtrain, opts, resulting_channels):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor

    N = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']

        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position & velocity
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(Xbin[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]

        # Store result
        curve[0, t] = fitG.copy()
        # print("Iteration:", t + 1)
        # print("Best (PSO):", curve[0, t])
        t += 1

        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1 = rand()
                r2 = rand()
                V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d])
                # Boundary
                V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                # Update position
                X[i, d] = X[i, d] + V[i, d]
                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

    # Best feature subset
    x_best = Xgb[0].argsort()[-resulting_channels:][::-1]
    num_feat = len(x_best)
    # Create dictionary
    pso_data = {'sf': x_best, 'c': curve, 'nf': num_feat, "df": Xgb[0]}

    return pso_data
