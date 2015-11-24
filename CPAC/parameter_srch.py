from rlpy.Tools.hypersearch import find_hyperparameters
best, trials = find_hyperparameters(
    "./experiment.py",
    "./Results/param_srch/",
    max_evals=10, parallelization="joblib",
    trials_per_point=5)
print best
