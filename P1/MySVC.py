#!/usr/bin/env python3

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class MySVC():
    """
        SVC with a simplified version of SMO.
    """
    def __init__(self, C=1.0, gamma='scale', tol=0.0001, max_iter=100, seed = 1234):
        # Assignment of the hyper-parameters.
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.intercept = None
        self.seed = seed


    def fit(self, X, y):
        # Constants.
        self.n_pat = X.shape[0]
        self.n_dim = X.shape[1]
        # Options for gamma (for compatibility with sklean).
        if (self.gamma == 'scale'):
            self.gamma = 1.0 / (self.n_dim * X.var())
        if (self.gamma == 'auto'):
            self.gamma = 1.0 / self.n_dim

        # Initialization of the dual coefficients (named "a" instead of "alpha" for simplicity).
        self.a = np.zeros(self.n_pat)

        self.X = X
        self.y = y

        # Pre-calculate all RBG configurations. As rbf(i,j) = rbf(j,i), only the "upper side
        # of the matrix" is computed.
        rbf = [[
            rbf_kernel(X[i].reshape(1, -1), X[j].reshape(1, -1))[0][0]
            for j in range(i, self.n_pat)
        ] for i in range(self.n_pat)]

        # Create a lambda function to access correct position in the upper side matrix given
        # "careless" indexes.
        RBF = lambda i, j: rbf[min(i, j)][max(i, j) - min(i, j)]

        # Loop over the iterations.
        for it in range(self.max_iter):
            # Safe the old value as it is used for convergence criteria.
            a_old = self.a

            # Compute all indexes are are going to be altered
            I = np.arange(self.n_pat)
            np.random.seed(self.seed)
            J = [np.random.choice(I[I != i]) for i in I]

            # As k does only depend on the RBF, it may be computed outside the loop.
            k = np.array([[2 * RBF(i, j) - RBF(i, i) - RBF(j, j) for i in I]
                          for j in J])

            # Set null values to a small quantity.
            k[k == 0] = 1e-16

            # Loop over the indexes.
            for i, j in zip(I, J):

                # Compute E(X_i, y_i) and E(X_j, y_j)
                Ei = sum(self.a * y * [RBF(k, i)
                                       for k in range(self.n_pat)]) - y[i]
                Ej = sum(self.a * y * [RBF(k, j)
                                       for k in range(self.n_pat)]) - y[j]

                # Compute d
                d = y[j] * (Ej - Ei) / k[i][j]

                # Compute bounds
                L, H = self._compute_bounds(i, j, y[i] != y[j])

                # Update of the corresponding a[i] and a[j] values.
                aj_new = np.minimum(np.maximum(self.a[j] + d, L), H)
                self.a[i] = self.a[i] - y[i] * y[j] * (aj_new - self.a[j])
                self.a[j] = aj_new

            # Check of the stopping conditions.
            if np.linalg.norm(self.a - a_old) < self.tol:
                break

        # Storage of the obtained parameters and computation of the intercept (complete).
        self.intercept = np.mean([
            y[k] - sum(self.a * y * [RBF(i, k) for i in range(self.n_pat)])
            for k in range(self.n_pat)
        ])

        return self

    def _compute_bounds(self, i, j, condition):
        # Compute lower bound
        L = max(0, self.a[j] - self.a[i]) if condition else max(
            0, self.a[j] + self.a[i] - self.C)

        # Computer upper bound
        H = min(self.C, self.C - self.a[i] + self.a[j]) if condition else min(
            self.C, self.a[j] + self.a[i] - self.C)

        return L, H

    def decision_function(self, X_test):
        # Computation of the decision function over X (complete).
        if self.intercept is None:
            print("[Error]: Model is not trained.")
        else:
            return [
                sum(self.a * self.y * [
                    rbf_kernel(self.X[i].reshape(1, -1), x.reshape(1, -1))[0][0]
                    for i in range(self.n_pat)
                ]) + self.intercept for x in X_test
            ]

    def predict(self, X_test):
        # Computation of the predicted class over X (complete).
        if self.intercept is None:
            print("[Error]: Model is not trained.")
        else:
            return np.sign(self.decision_function(X_test))

    def score(self, X_test, y_test):
        if self.intercept is None:
            print("[Error]: Model is not trained.")
        else:
            return np.sum(self.predict(X_test) == y_test)/len(y_test)
