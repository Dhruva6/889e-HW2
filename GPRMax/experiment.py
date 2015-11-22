from rlpy.Domains.HIVTreatment import HIVTreatment
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from rlpy.Representations import IncrementalTabular

import numpy as np
from GPRMax import GPRMax

def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=238,
        lambda_=0.9,
        initial_learn_rate=.08,
        discretization=35):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 5000
    opt["num_policy_checks"] = 1
    opt["checks_per_policy"] = 1

    domain = HIVTreatment()
    opt["domain"] = domain
    representation = IncrementalTabular(domain)
    policy = eGreedy(representation)
    opt["agent"] = GPRMax(
        policy, representation, domain.actions_num, len(domain.state_names),
        discount_factor=domain.discount_factor,
        lambda_=0.9,initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
   # experiment.plot()
    # experiment.save()

