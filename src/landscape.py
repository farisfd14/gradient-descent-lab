"""
Milestone 1 - See the mountain.

Before we can roll a ball downhill, we need a hill.
In machine learning, the "hill" is called the LOSS LANDSCAPE:

    - We have one number w (imagine: the single weight of a model)
    - For every possible value of w, the model makes some error
    - That error is the LOSS
    - Good models have LOW loss -> we want to find the bottom of the valley

We use a simple made-up loss function on purpose:

        L(w) = (w - 3)^2 + 1

Why this one? Because we ALREADY know the answer: the smaleest possible
loss happens at w = 3, where (w - 3) becomes 0 and L = 1.
So this function is our training ground: gradient descent should find
w = 3 without us ever telling it answer.

This milestone only DRAWS the landscape. We don't optimize anything yet.
"""

import numpy as np
import matplotlib.pyplot as plt


def loss(w):
    """The loss function L(w) = (w - 3)^2 + 1.

    np.square works on a single number OR a whole array of numbers,
    which is why we can pass 200 w-values at once and get 200 losses back.
    """
    return np.square(w - 3) + 1


def main():
    # 1. Make 200 evenly spaced guesses for w, from -2 to 8
    w_values = np.linspace(-2, 8, 200)

    #2. Ask the loss function: "how wrong is each of these guesses?"
    loss_values = loss(w_values)

    #4. Draw the landscape
    plt.figure(figsize=(8, 5))
    plt.plot(w_values, loss_values, color="steelblue", linewidth=2)
    plt.xlabel("w  (our guess for the best value)")
    plt.ylabel("loss  (how wrong the guess is)")
    plt.title("The loss landscape - a valley with its floor at w = 3")
    plt.grid(True, alpha=0.3)

    # 4. Mark the known minimum, so we can check later whether
    #    gradient descent finds the same spot
    plt.scatter([3], [1], color="red", zorder=5, label="true minimum (w=3)")
    plt.legend()

    # Save to disk AND show on screen
    plt.savefig("charts/loss_landscape.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()