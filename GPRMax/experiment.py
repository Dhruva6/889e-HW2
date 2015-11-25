from rlpy.Domains.HIVTreatment import HIVTreatment
from rlpy.Policies import UniformRandom
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from rlpy.Representations import IncrementalTabular

import numpy as np
from GPRMax import GPRMax
from GPRMaxRepresentation import GPRMaxRepresentation

def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=238,
        lambda_=0.9,
        initial_learn_rate=.08,
        discretization=35):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 10000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 1

    # the Horizon length
    H = 50.0

    # the discount factor
    gamma = 1.0 - (1.0/H)

    print "Discount Factor for the experiment %.2f" % gamma

    domain = HIVTreatment()
    opt["domain"] = domain

    # update the discount factor for the domain
    #domain.discount_factor = gamma

    # create the representation
    representation = GPRMaxRepresentation(domain)

    # update the discount factor for the representation (directly)
    representation.discountFactor = gamma

    # instantiate the policy
    policy = eGreedy(representation, epsilon=0)
    
    opt["agent"] = GPRMax(
        policy, representation, domain.actions_num, len(domain.state_names),
        discount_factor=gamma,
        lambda_=0.9,initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
    # experiment.save()

