"""IsoFLOP scaling-law fitting on a synthetic loss surface, under a query budget.

The loss surface is a known Chinchilla-style function plus noise, so the
fitted allocation can be checked. Three designs are compared:
* dense: many model sizes per budget (what a textbook plot shows);
* budgeted: a few sizes per budget, chosen around the previous budget's
  optimum (what an A3-style fixed query budget allows);
* budgeted_extended: the same, but a budget whose lowest run lands on the grid
  edge gets extra sizes beyond that edge until the minimum is interior.
Each takes the lowest-loss run per budget (the A3 recommendation) and also
fit a parabola in log N (the Chinchilla variant), then fit N_opt = a C^b.
"""
import numpy as np

TRUE = dict(E=1.0, A=2.0, B=1.0, alpha=0.35, beta=0.3)


def _positive_vector(values, name, minimum=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must be a finite positive vector with at least {minimum} entries")
    return values


def loss(n, d, rng=None, noise=0.0):
    value = TRUE["E"] + TRUE["A"] * n ** -TRUE["alpha"] + TRUE["B"] * d ** -TRUE["beta"]
    return value + (rng.normal(0, noise, np.shape(n)) if rng is not None and noise else 0)


def optimal_allocation(compute, a=TRUE["A"], b=TRUE["B"], alpha=TRUE["alpha"], beta=TRUE["beta"]):
    # Loss uses normalized N,D units; C=6ND in the SAME units.
    n = ((alpha * a) / (beta * b)) ** (1 / (alpha + beta)) * (compute / 6) ** (beta / (alpha + beta))
    return n, compute / (6 * n)


def isoflop_minimum(sizes, losses, method="lowest"):
    """Lowest observed run, or the vertex of a parabola fitted in log N."""
    sizes, losses = _positive_vector(sizes, 'sizes'), np.asarray(losses, float)
    if losses.shape != sizes.shape or not np.isfinite(losses).all() or np.any(np.diff(sizes) <= 0) \
            or method not in ('lowest', 'parabola'):
        raise ValueError("strictly increasing sizes, matching finite losses and a supported fit method required")
    best = int(np.argmin(losses))
    if method == "lowest" or len(sizes) < 3:
        return float(sizes[best]), best in (0, len(sizes) - 1)
    c2, c1, _ = np.polyfit(np.log(sizes), losses, 2)
    if c2 <= 0:
        return float(sizes[best]), True
    vertex = -c1 / (2 * c2)
    edge = not np.log(sizes).min() <= vertex <= np.log(sizes).max()
    predicted = float(np.exp(vertex)) if vertex < np.log(np.finfo(float).max) else np.inf
    if not np.isfinite(predicted) or predicted <= 0:
        raise ValueError("parabola vertex is outside the representable model-size range")
    return predicted, edge


def fit_power_law(budgets, values):
    budgets, values = _positive_vector(budgets, 'budgets', 2), _positive_vector(values, 'values', 2)
    if budgets.shape != values.shape or np.unique(budgets).size < 2:
        raise ValueError("aligned samples with at least two distinct compute budgets required")
    slope, intercept = np.polyfit(np.log(budgets), np.log(values), 1)
    return float(np.exp(intercept)), float(slope)


def run_design(budgets, sizes_for, rng, noise, method, extend_edges=False, max_extensions=4):
    """Fit N_opt = a C^b over budgets. With extend_edges, a budget whose lowest run sits on the
    grid edge gets one more size beyond that edge (same log spacing) until the minimum is
    interior, or max_extensions is reached; every added run counts against the query budget.
    Record observed grid-edge minima separately from an extrapolated parabola vertex."""
    budgets = _positive_vector(budgets, 'budgets', 2)
    if np.any(np.diff(budgets) <= 0) or not np.isfinite(noise) or noise < 0 or type(max_extensions) is not int or max_extensions < 0:
        raise ValueError("increasing budgets, finite nonnegative noise and extension count required")
    optima, initial_edges, final_edges, fit_edges, queries, spent = [], [], [], [], 0, 0.0
    previous = None
    for c in budgets:
        sizes = _positive_vector(sizes_for(c, previous), 'sizes', 2 if extend_edges else 1)
        if np.any(np.diff(sizes) <= 0):
            raise ValueError("model sizes must be strictly increasing")
        losses = loss(sizes, c / (6 * sizes), rng, noise)
        first_edge = int(np.argmin(losses)) in (0, len(sizes) - 1)
        for _ in range(max_extensions if extend_edges else 0):
            best = int(np.argmin(losses))
            if best not in (0, len(sizes) - 1):
                break
            ratio = sizes[1] / sizes[0]
            new = sizes[0] / ratio if best == 0 else sizes[-1] * ratio
            sizes = np.append(sizes, new)
            losses = np.append(losses, loss(new, c / (6 * new), rng, noise))
            order = np.argsort(sizes)
            sizes, losses = sizes[order], losses[order]
        queries += len(sizes)
        spent += len(sizes) * c
        n_opt, edge = isoflop_minimum(sizes, losses, method)
        optima.append(n_opt)
        initial_edges.append(first_edge)
        final_edges.append(int(np.argmin(losses)) in (0, len(sizes) - 1))
        fit_edges.append(not sizes.min() <= n_opt <= sizes.max())
        previous = n_opt
    a, b = fit_power_law(budgets, optima)
    return dict(optima=optima, edge_minima=initial_edges, final_edge_minima=final_edges,
                fit_outside_grid=fit_edges, queries=queries, compute_spent=spent, coefficient=a, exponent=b)


def experiment(seed=11, noise=0.003, target=1e9):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("finite positive target compute required")
    rng = np.random.default_rng(seed)
    budgets = np.geomspace(1e3, 1e6, 7)
    dense_grid = np.geomspace(2, 20000, 41)
    # Budgeted design: 5 sizes spanning 4x either side of a guess that follows N ~ C^0.5.
    def budgeted(c, previous):
        guess = previous * np.sqrt(c / budgeted.last) if previous else np.sqrt(c / 6)
        budgeted.last = c
        return guess * np.geomspace(0.25, 4, 5)
    budgeted.last = budgets[0]
    true_n, true_d = optimal_allocation(target)
    designs = {}
    variants = [("dense", lambda c, _: dense_grid, False), ("budgeted", budgeted, False),
                ("budgeted_extended", budgeted, True)]
    for name, sizes_for, extend in variants:
        for method in ("lowest", "parabola"):
            budgeted.last = budgets[0]
            result = run_design(budgets, sizes_for, rng, noise, method, extend_edges=extend)
            predicted = result["coefficient"] * target ** result["exponent"]
            result.update(predicted_N=predicted, relative_error_N=predicted / true_n - 1,
                          loss_at_prediction=float(loss(predicted, target / (6 * predicted))),
                          loss_at_optimum=float(loss(true_n, true_d)))
            designs[f"{name}_{method}"] = result
    return dict(synthetic=True, seed=seed, noise=noise, budgets=budgets.tolist(), target_compute=target,
                true_exponent=TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"]), true_N_at_target=true_n,
                designs=designs)


def seed_study(seeds=range(20), **kwargs):
    """Repeat the whole design under fresh noise: one seed's error says little about a method."""
    runs = [experiment(seed=s, **kwargs) for s in seeds]
    if not runs:
        raise ValueError("at least one seed required")
    summary = {}
    for name in runs[0]["designs"]:
        errors = np.array([r["designs"][name]["relative_error_N"] for r in runs])
        exponents = np.array([r["designs"][name]["exponent"] for r in runs])
        queries = [r["designs"][name]["queries"] for r in runs]
        costs = [r["designs"][name]["compute_spent"] for r in runs]
        summary[name] = dict(queries=queries[0] if min(queries) == max(queries) else None,
                             queries_min=min(queries), queries_max=max(queries), queries_mean=float(np.mean(queries)),
                             compute_spent_min=min(costs), compute_spent_max=max(costs), compute_spent_mean=float(np.mean(costs)),
                             median_abs_error_N=float(np.median(np.abs(errors))),
                             worst_abs_error_N=float(np.abs(errors).max()),
                             exponent_mean=float(exponents.mean()), exponent_std=float(exponents.std()),
                             edge_minima_rate=float(np.mean([np.mean(r["designs"][name]["edge_minima"]) for r in runs])),
                             final_edge_minima_rate=float(np.mean([np.mean(r["designs"][name]["final_edge_minima"]) for r in runs])),
                             fit_outside_grid_rate=float(np.mean([np.mean(r["designs"][name]["fit_outside_grid"]) for r in runs])))
    return dict(synthetic=True, seeds=len(runs), true_exponent=runs[0]["true_exponent"], summary=summary)


if __name__ == "__main__":
    study = seed_study()
    print("true exponent", round(study["true_exponent"], 4), "over", study["seeds"], "noise seeds")
    for name, d in study["summary"].items():
        count = str(d['queries']) if d['queries'] is not None else f"{d['queries_min']}–{d['queries_max']}"
        print(f"{name:18s} queries={count:>5s} exponent={d['exponent_mean']:.3f}±{d['exponent_std']:.3f} "
              f"median |N error|={d['median_abs_error_N']:.1%} worst={d['worst_abs_error_N']:.1%}")
