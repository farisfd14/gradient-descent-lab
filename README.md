# Gradient Descent Lab

Watching gradient descent work, from scratch - NumPy + matplotlib,
no autograd, no deep learning framework.

## What it does
- `src/landscape.py` - draws the loss landscape (Milestone 1)
- `src/onestep.py`   - one hand-computed step downhill (Milestone 2)
- `src/descent.py`   - the full loop + learning-rate experiment (Milestone 3)
- `src/verify.py`    - fits real data, compares against scikit-learn (Milestone 4)
- `tests/`           - 3 tests: convergence, divergence, slope-at-minimum

## The learning rate experiment (all starting at w = -1)
| Learning rate | Steps | Final w | What happened |
|-----------|-------|  | --- |
| 0.01      | 787   | 3.0000 | too small: crawls |
| 0.1 | 72   | 3.0000 | just right: converges to ~3 |
| 1.0 | 20   | -1.0000 | too big: diverges |

## What I learned
1. Gradient descent = feel the slope, step against it, repeat.
2. The step shrinks by itself near the bottom because slope -> 0.
3. Learning rate too big = bounce out of the valley; too small = forever.
4. My from-scratch loop matches scikit-learn on real data to within 0.000000.

## Run it
    py src/descent.py
    py src/verify.py
    py -m pytest tests/ -v