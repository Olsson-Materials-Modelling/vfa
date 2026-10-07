"""Held-out classification of cells from a few descriptors: a multinomial logistic model with every choice made inside the training cells."""
import numpy as np
from .intervals import quantile


def standardise(train, X):
    """Columns centred and scaled by the training cells only; a constant column stays centred."""
    mu, sd = train.mean(0), train.std(0, ddof=1); return (X - mu) / np.where(sd > 0, sd, 1.0)


def softmax_fit(X, y, classes, l2=1.0, iters=400, lr=0.3):
    """Weights (bias row first) of a multinomial logistic model by full-batch gradient descent with an L2 penalty."""
    n, d = X.shape; Xb = np.hstack([np.ones((n, 1)), X]); Y = np.stack([(y == c).astype(float) for c in classes], 1); W = np.zeros((d + 1, len(classes)))
    for _ in range(iters):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True); g = Xb.T @ (P - Y) / n; g[1:] += l2 * W[1:] / n; W -= lr * g
    return W


def softmax_predict(W, X):
    """Class probabilities of every row."""
    Z = np.hstack([np.ones((X.shape[0], 1)), X]) @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def held_out_shares(X_train, y_train, X_test, classes, seed=0, n_replicates=400, alpha=0.05, **fit):
    """Predicted class shares of the test cells from a model fitted on the training cells, with a percentile interval under resampling of the test cells."""
    W = softmax_fit(standardise(X_train, X_train), y_train, classes, **fit); Xt = standardise(X_train, X_test); rng = np.random.default_rng(seed); P = softmax_predict(W, Xt)
    reps = np.array([softmax_predict(W, Xt[rng.integers(0, len(Xt), len(Xt))]).mean(0) for _ in range(n_replicates)])
    return dict(classes=list(classes), predicted=P.mean(0), lower=np.array([quantile(reps[:, k], alpha / 2) for k in range(len(classes))]), upper=np.array([quantile(reps[:, k], 1 - alpha / 2) for k in range(len(classes))]), weights=W, log_loss=None)


def log_loss(P, y, classes):
    """Mean negative log probability of the observed class."""
    Y = np.stack([(y == c) for c in classes], 1); return float(-np.mean(np.log(np.clip((P * Y).sum(1), 1e-12, None))))


def learning_curve(X_train, y_train, X_test, classes, sizes, seed=0, n_replicates=200, alpha=0.05, **fit):
    """Predicted test shares against the number of training cells (the first n in the given order), with an interval under resampling of those n cells."""
    rng = np.random.default_rng(seed); rows = []
    for n in sizes:
        Xn, yn = X_train[:n], y_train[:n]; pred = softmax_predict(softmax_fit(standardise(Xn, Xn), yn, classes, **fit), standardise(Xn, X_test)).mean(0); reps = []
        for _ in range(n_replicates):
            j = rng.integers(0, n, n); Xb, yb = Xn[j], yn[j]; reps.append(softmax_predict(softmax_fit(standardise(Xb, Xb), yb, classes, **fit), standardise(Xb, X_test)).mean(0))
        reps = np.array(reps); rows += [dict(n_train=int(n), cls=c, predicted=float(pred[k]), lower=quantile(reps[:, k], alpha / 2), upper=quantile(reps[:, k], 1 - alpha / 2)) for k, c in enumerate(classes)]
    return rows
