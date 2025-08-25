import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import gamma


# Logistic chaotic initialization
def initialization_logistic(pop, dim, lb, ub):
    Positions = np.zeros((pop, dim))
    for i in range(pop):
        x0 = np.random.rand()
        for j in range(dim):
            a = 4
            x = a * x0 * (1 - x0)
            Positions[i, j] = lb[j] + (ub[j] - lb[j]) * x
            x0 = x
    return Positions


# Levy flight
def levy(dim):
    beta = 1.5
    sigma = (
        gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step


# Objective function
def classification_objective(params, X_train, y_train, X_test, y_test):
    C = 10 ** params[0]
    penalty_map = {0: "l1", 1: "l2", 2: "elasticnet"}
    solver_map = {0: "liblinear", 1: "saga", 2: "lbfgs"}
    penalty = penalty_map[int(params[1] % 3)]
    solver = solver_map[int(params[2] % 3)]
    max_iter = int(params[3])
    tol = 10 ** params[4]
    fit_intercept = bool(round(params[5]))

    try:
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=fit_intercept,
            l1_ratio=0.5 if penalty == "elasticnet" else None,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return 1 - acc
    except Exception:
        return 1  # Penalize invalid configurations


# HEOA main loop
def HEOA(N, Max_iter, lb, ub, dim, fobj):
    jump_factor = abs(lb[0] - ub[0]) / 1000
    A, LN, EN, FN = 0.6, 0.4, 0.4, 0.1
    LNNumber = int(N * LN)
    ENNumber = int(N * EN)
    FNNumber = int(N * FN)

    X = initialization_logistic(N, dim, lb, ub)
    fitness = np.array([fobj(ind) for ind in X])
    idx = np.argsort(fitness)
    X = X[idx]
    fitness = fitness[idx]
    GBestX = X[0].copy()
    GBestF = fitness[0]
    curve = []

    for i in range(Max_iter):
        X_new = np.copy(X)
        R = np.random.rand()
        for j in range(N):
            if i <= Max_iter / 4:
                X_new[j] = (
                    GBestX * (1 - i / Max_iter)
                    + (np.mean(X[j]) - GBestX)
                    * np.floor(np.random.rand() / jump_factor)
                    * jump_factor
                    + 0.2 * (1 - i / Max_iter) * (X[j] - GBestX) * levy(dim)
                )
            else:
                if j < LNNumber:
                    if R < A:
                        X_new[j] = (
                            0.2
                            * np.cos(np.pi / 2 * (1 - i / Max_iter))
                            * X[j]
                            * np.exp(
                                -i * np.random.randn() / (np.random.rand() * Max_iter)
                            )
                        )
                    else:
                        X_new[j] = 0.2 * np.cos(np.pi / 2 * (1 - i / Max_iter)) * X[
                            j
                        ] + np.random.randn(dim)
                elif j < LNNumber + ENNumber:
                    X_new[j] = np.random.randn(dim) * np.exp(
                        np.clip((X[-1] - X[j]) / (j + 1) ** 2, -100, 100)
                    )
                elif j < LNNumber + ENNumber + FNNumber:
                    X_new[j] = X[j] + 0.2 * np.cos(
                        np.pi / 2 * (1 - i / Max_iter)
                    ) * np.random.rand(dim) * (X[0] - X[j])
                else:
                    X_new[j] = GBestX + (GBestX - X[j]) * np.random.randn()

            X_new[j] = np.clip(X_new[j], lb, ub)

        fitness_new = np.array([fobj(ind) for ind in X_new])
        for j in range(N):
            if fitness_new[j] < GBestF:
                GBestF = fitness_new[j]
                GBestX = X_new[j].copy()

        X = X_new
        fitness = fitness_new
        curve.append(GBestF)
        print(f"Iteration {i+1}/{Max_iter} | Best Accuracy: {1 - GBestF:.4f}")

    return GBestX, 1 - GBestF, curve


# Run the optimizer
if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lb = [-6, 0, 0, 100, -6, 0]
    ub = [6, 2, 2, 1000, -2, 1]
    dim = 6
    N = 30
    Max_iter = 50

    def fobj_wrapper(params):
        return classification_objective(params, X_train, y_train, X_test, y_test)

    best_params, best_score, convergence = HEOA(N, Max_iter, lb, ub, dim, fobj_wrapper)

    print("\nBest Parameters:")
    print(f"log10(C): {best_params[0]:.4f} → C = {10 ** best_params[0]:.4f}")
    print(f"Penalty: {['l1','l2','elasticnet'][int(best_params[1] % 3)]}")
    print(f"Solver: {['liblinear','saga','lbfgs'][int(best_params[2] % 3)]}")
    print(f"Max Iter: {int(best_params[3])}")
    print(f"Tolerance: {10 ** best_params[4]:.6f}")
    print(f"Fit Intercept: {bool(round(best_params[5]))}")
    print(f"Best Accuracy: {best_score:.4f}")
