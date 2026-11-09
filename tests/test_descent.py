# Tests for the toy-valley gradient descent (Milestone 3 code).
# Run: py -m pytest test/ -v

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from descent import loss, slope, gradient_descent


def test_coverages_to_the_minimum():
    w, path = gradient_descent(start_w=-1.0, learning_rate=0.1)
    assert abs(w - 3.0) < 1e-3          # must land essentially on w =3


def test_big_learning_rate_diverges():
    # Lr = 1.0 overshoots every step: the Loss EXPLODES instead of shrinking
    w, _ = gradient_descent(start_w=-1.0, learning_rate=1.0, max_steps=20)
    assert loss(w) >= loss(-1.0)        # was: > - oscillation ends EQUAL, not higher


def test_slope_is_zero_at_the_bottom():
    # Q3 as a permanent check: at the minimum the hill is flat
    assert abs(slope(3.0)) < 1e-9

