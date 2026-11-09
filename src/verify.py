"""
Milestone 4 - Trust but verify: gradient descent vs scikit-learn.

Everything so far was a made-up valley L(w) = (w - 3)^2 + 1.
Now we fit a REAL linear model to REAL (synthetic) data:

    data:   points scattered around the line y = 3x
    model:  y_pred = w * x      (one weight, no bias)
    loss:   MSE = mean( (y - y_pred)^2 )    (mean squared error)

The derivative of MSE with respect to w (still by hand!):
    dL/dw = -2 * mean( x * (y - w*x) )

Then we fit the SAME data with scikit-learn's LinearRegression
and compare the two answers. If they agree, our from-scratch
gradient descent is correct.
"""

import numpy as np
from scipy.stats import mode
from sklearn.linear_model import LinearRegression


def mse(w, x, y):
    """Mean squared error: average of (prediction - truth)^2."""
    return np.mean(np.square(y - w * x))


def mse_slope(w, x, y):
    """d(MSE)/dw = -2 * mean( x * (y - w*x) )"""
    return -2 * np.mean(x * (y - w * x))


def fit_by_hand(x, y, learning_rate=0.001, max_steps=20000, tol=1e-8):
    """Our own gradient descent, fitting w to real data."""
    w = 0.0
    for step in range(max_steps):
        s = mse_slope(w, x, y)
        if abs(s) < tol:
            break
        w = w - learning_rate * s
    return w


def main():
    rng = np.random.default_rng(42)             # seed: same "random" data every run
    x = np.linspace(0, 10, 50)  # 50 x-values, 0 to 10
    y = 3 * x + rng.normal(0, 1.5, size=50)     # the true line (slope 3) + noise

    # 1. Our from-scratch answer
    w_hand = fit_by_hand(x, y)

    # 2. scikit-Learn's answer (professionals' tool)
    model = LinearRegression(fit_intercept=False)
    model.fit(x.reshape(-1, 1), y)          # sklearn wants 2D input
    w_sklearn = model.coef_[0]

    print("=== Who found the slope of the line better? ===")
    print(f"true slope (how we made the data) : 3.0000")
    print(f"our hand-made gradient descent    : {w_hand:.4f}")
    print(f"scikit-learn LinearRegression     : {w_sklearn:.4f}")
    print()
    print(f"difference between the two        : {abs(w_hand - w_sklearn):.6f}")


if __name__ == "__main__":
    main()
