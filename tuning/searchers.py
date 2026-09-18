import itertools
import random
import optuna

def grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def random_search(param_distributions, n_iter=10):
    for _ in range(n_iter):
        sample = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                sample[key] = random.choice(dist)
            elif callable(dist):  # distribuciones continuas (ej. lambda: 10**random.uniform(-4, -1))
                sample[key] = dist()
        yield sample

def bayesian_search(objective_fn, n_trials=20):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=n_trials)
    return study.best_params, study.best_value