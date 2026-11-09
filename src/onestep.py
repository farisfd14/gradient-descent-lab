"""
Milestone 2 - Feel the slope (ONE gradient descent step).

The whole of gradient descent is a 3-line loop:
    1. FEEL:  compute the slope (derivative) where we stand.
    2. STEP:  move a little AGAINST the slope (downhill)
    3. REPEAT (that's Milestone 3 - here we do it ONCE)

We do NOT use any autograd library. The derivative is written
by hand, because understanding it is the entire point:

    L(w) = (w - 3)^2 + 1        <- our loss (from Milestone 1)
    dL/dw = 2(w - 3)            <- its derivative, by hand

Why "step AGAINST the slope"?
  - slope > 0 meanas the valley floor is to our LEFT -> step left
  - slope < 0 means the valley floor is to our RIGHT -> step right
So we SUBTRACT the slope (times a small learning rate).
"""

import numpy as np
import matplotlib.pyplot as plt


def loss(w):
    """L(w) = (w - 3)^2 + 1  (same landscape as Milestone 1)"""
    return np.square(w - 3) + 1


def slope(w):
    """dL/dw = 2(w - 3)  (the derivative, written out by hand).

    This function is the flashlight: pass it any position w
    and it returns the steepness AND direction of the hill there.
    """
    return 2 * (w - 3)


def main():
    # --------------------------------------------------------
    # Setup : start somewhere WRONG, on purpose.
    # w = -1 is far from the true answer (w = 3), so we can
    # clearly see whether our single step moves us closer.
    # ---------------------------------------------------------
    w = -1.0
    learning_rate = 0.1         # how BIG each step is (very important!)

    loss_before = loss(w)

    # ----------------------------------------------------------
    # STEP 1 - FEEL: how steep is the hill where we stand?
    # ----------------------------------------------------------
    current_slope = slope(w)

    # ----------------------------------------------------------
    # STEP 2 - STEP: move against the slope.
    # Subtract (Learning_rate * slope) from our position.
    # ----------------------------------------------------------
    w_new = w - learning_rate * current_slope

    loss_after = loss(w_new)

    # ----------------------------------------------------------
    # Report everything like a lab notebook
    # ----------------------------------------------------------
    print("=== ONE gradient descent step ===")
    print(f"start position w        = {w:.4f}")
    print(f"loss at start           = {loss_before:.4f}")
    print(f"slope at start          = {current_slope:.4f}")
    print(f"learning rate           = {learning_rate}")
    print(f"step taken              = -({learning_rate} * {current_slope:.4f}) = - {learning_rate * current_slope:.4f}")
    print(f"new position w          = {w_new:.4f}")
    print(f"loss at new position    = {loss_after:.4f}")
    print()

    # Sanity check: did we actually go DOWNHILL?
    if loss_after < loss_before:
        print("SUCCESS: the loss went DOWN. Our step moved toward the valley floor.")
    else:
        print("PROBLEM: the loss went UP. Something is wrong in the math.")

    # ----------------------------------------------------------
    # Draw it: the Landscape, our start point, our step (arrow)
    # ----------------------------------------------------------
    w_values = np.linspace(-2, 8, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(w_values, loss(w_values), color="steelblue", linewidth=2)

    plt.scatter([w], [loss_before], color="black", zorder=5, s=60,
                label=f"start (w={w})")
    plt.scatter([w_new], [loss_after], color="green", zorder=5, s=60,
                label=f"after 1 step (w={w_new:.2f})")
    plt.scatter([3], [1], color="red", zorder=5, label="true monimum (w=3)")

    # Arrow from start to after, showing the step direction
    plt.annotate("", xy=(w_new, loss_after), xytext=(w, loss_before),
                 arrowprops=dict(arrowstyle="->", color="green", lw=2))

    plt.xlabel("w  (our guess)")
    plt.ylabel("loss")
    plt.title("One step downhill")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("charts/one_step.png", dpi=120)


if __name__ == "__main__":
    main()
