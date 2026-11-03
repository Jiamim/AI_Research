"""Small numerical tools for experimental design and domain adaptation."""
import json
import numpy as np


def reweight_domains(weights, excess_losses, eta=1.0, floor_mass=0.05):
    """One illustrative exponentiated-gradient step; not a DoReMi trainer."""
    w = np.asarray(weights, dtype=float)
    e = np.asarray(excess_losses, dtype=float)
    if w.ndim != 1 or w.size == 0 or w.shape != e.shape or np.any(w <= 0):
        raise ValueError("matching nonempty vectors and strictly positive weights required")
    if not np.isfinite(w).all() or not np.isfinite(e).all() or not np.isfinite(eta) or eta < 0 or not 0 <= floor_mass <= 1:
        raise ValueError("finite inputs, nonnegative eta and floor_mass in [0,1] required")
    # Normalization cancels inside softmax; never sum huge weights in ordinary space.
    logw = np.log(w)
    logw -= logw.max()
    if eta == 0:
        shifted = np.zeros_like(e)
    else:
        with np.errstate(over='ignore'):
            shifted = (e - e.max()) * eta
    logits = logw + shifted
    updated = np.exp(logits - logits.max())
    updated /= updated.sum()
    return (1 - floor_mass) * updated + floor_mass / len(w)


def pareto_front(rows, maximize):
    """Return indices not strictly dominated; one direction flag per metric."""
    x = np.asarray(rows, dtype=float)
    directions = np.asarray(maximize, dtype=bool)
    if x.ndim != 2 or len(x) == 0 or directions.shape != (x.shape[1],) or not np.isfinite(x).all():
        raise ValueError("finite nonempty matrix and one direction flag per metric required")
    oriented = x * np.where(directions, 1, -1)
    return [i for i, row in enumerate(oriented) if not any(
        np.all(other >= row) and np.any(other > row) for j, other in enumerate(oriented) if j != i)]


def factorial_interaction(s00, s10, s01, s11):
    return s11 - s10 - s01 + s00


def gradient_transfer_demo(step=0.1):
    """Two opposing quadratic tasks: local conflict, not an LM forgetting experiment."""
    theta = np.zeros(2)
    a, b = np.array([1., 0.]), np.array([-1., 0.])
    ga, gb = theta - a, theta - b
    updated = theta - step * gb
    return dict(gradient_inner_product=float(ga @ gb),
                predicted_old_loss_change=float(-step * (ga @ gb)),
                actual_old_loss_change=float(.5 * np.sum((updated - a) ** 2) - .5 * np.sum((theta - a) ** 2)),
                actual_new_loss_change=float(.5 * np.sum((updated - b) ** 2) - .5 * np.sum((theta - b) ** 2)))


if __name__ == "__main__":
    print(json.dumps(dict(domain_weights=reweight_domains([.5, .3, .2], [.1, .8, .2]).tolist(),
                          pareto_indices=pareto_front([[80, 200], [81, 300], [79, 250]], [True, False]),
                          interaction=factorial_interaction(70, 72, 73, 74),
                          gradient_transfer=gradient_transfer_demo()), indent=2))
