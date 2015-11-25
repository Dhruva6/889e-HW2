from rlpy.Domains.HIVTreatment import HIVTreatment
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

from RMAX import RMAX
from RMAX_repr import RMAX_repr


#param_space = {'discretization': hp.quniform("discretization", 5, 50, 1),
#               'lambda_': hp.uniform("lambda_", 0., 1.),
#               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
#               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}
param_space = {'Rmax':hp.loguniform("Rmax", 5, 10),
               'lipschitz_constant':hp.uniform('lipschitz_constant', 100, 1000),
               'epsilon_d':hp.uniform('epsilon_d', 0, 1.0),
               'knn':hp.choice('knn', [1, 2, 3, 4, 5])}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        Rmax=10**10, lipschitz_constant=10**3, epsilon_d=0.01, knn = 1):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    #opt["max_steps"] = 150000
    opt["max_steps"] = 8000
    #opt["num_policy_checks"] = 30
    opt["num_policy_checks"] = 4
    opt["checks_per_policy"] = 2
    epsilon_d = 0.9
    knn = 1

    domain = HIVTreatment()
    opt["domain"] = domain
    representation = RMAX_repr(
        domain, Rmax, lipschitz_constant, epsilon_d=epsilon_d, k=knn)
    policy = eGreedy(representation, epsilon=0.0)

    opt["agent"] = RMAX(
        policy, representation,discount_factor=domain.discount_factor,
        lambda_=0, initial_learn_rate=0)

    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    #from rlpy.Tools.run import run_profiled
    #run_profiled(make_experiment)
    #experiment = make_experiment(2, "./Results/try")
    experiment = make_experiment(4)
    experiment.run(visualize_learning=False, visualize_steps=False, visualize_performance=1)
    experiment.plot()
    experiment.save()
