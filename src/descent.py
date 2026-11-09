"""
Milestone 3 - Roll the ball: the FULL gradient descent loop.

Milestone 2 did ONE step. This file wraps that step in a loop
and answers the two questions from the understanding check:

    Q2: learning rate = 1.0  -> the ball OVERSHOOTS the valley
        and bounces further and further = DIVERGES (explodes)
    Q3: near the bottom the slope -> 0, so the step size -> 0
        and the ball STOPS BY ITSELF. We only need a loop
        with a "stop when the slope is tiny" condition.

This is LITERALLY the training loop inside PyTorch/TensorFlow:
    for each step:
        slope = gradient of loss
        weights = weights - learning_rate * slope
Only they do it with millions of weights at once.
"""

import numpy as np
import matplotlib.pyplot as plt


def loss(w):
    """L(w) = (w - 3)^2 + 1"""
    return np.square(w - 3) + 1


def slope(w):
    """dL/dw = 2(w - 3)  (the flashlight)"""
    return 2 * (w - 3)


def gradient_descent(start_w, learning_rate, max_steps=1000, tol=1e-6):
    """The full loop.

    Returns (final_w, path) where path is every w we visited,
    so we can draw the ball's trail afterwards.

    tol = tolerance: if the slope is tinier than this, we treat
    the ground as flat and stop (this is Q3 turned into code).
    """
    w = start_w
    path = [w]                          # remember where we've been

    for step in range(max_steps):
        current_slope = slope(w)

        if abs(current_slope) < tol:
            break                       # Q3: flat ground -> stop by ourselves

        w = w - learning_rate * current_slope
        path.append(w)

    return w, path


def show(label, final_w, path):
    print(f"{label:<28} final w = {final_w:>9.4f}   "
          f"loss = {loss(final_w):>10.4f}   steps taken = {len(path) - 1}")


def main():
    start_w = -1.0

    # ----------------------------------------------------------------
    # EXPERIMENT: same start point, three Learning rates.
    # This is the single most instructive table in ML.
    # ----------------------------------------------------------------
    print("=== Learning rate experiment (all start at w = -1) ===")

    # 1. Too small: crawls. Watch the step count explode.
    w_small, path_small = gradient_descent(start_w, learning_rate=0.01)
    show("lr = 0.01 (too small)", w_small, path_small)

    # 2. Just right: reaches the bottom quickly.
    w_good, path_good = gradient_descent(start_w, learning_rate=0.1)
    show("lr = 0.1  (just right)", w_good, path_good)

    # 3. Too big: OVERSHOOTS the valley every step and runs away.
    #    max_steps=20 so we can SEE the explosion without a
    #    gigantic useless path list.
    w_big, path_big = gradient_descent(start_w, learning_rate=1.0, max_steps=20)
    show("lr = 1.0  (too big)", w_big, path_big)

    print(f"\nTrue answer is w = 3. 'Just right' got within {abs(w_good - 3):.6f}.")

    # ----------------------------------------------------------------
    # Draw the good run's trail: landscape + red path dots
    # ----------------------------------------------------------------
    w_values = np.linspace(-2, 8, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(w_values, loss(w_values), color="steelblue", linewidth=2)

    path = np.array(path_good)
    plt.plot(path, loss(path), "o-", color="red", linewidth=1.5,
             markersize=5, label="start (w=-1)")

    plt.xlabel("w  (our guess)")
    plt.ylabel("loss")
    plt.title("Gradient descent rolling into the valley")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("charts/descent_path.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()