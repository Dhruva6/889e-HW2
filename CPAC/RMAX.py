"""Control Agents based on TD Learning, i.e., Q-Learning and SARSA"""
from rlpy.Agents.Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class RMAX(DescentAlgorithm, Agent):
    QMax = 0.0

    def __init__(self, policy, representation, discount_factor, lambda_=0, **kwargs):
        super(
            RMAX,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        # The previous state could never be terminal
        # (otherwise the episode would have already terminated)
        prevStateTerminal = False
        
        self.representation.pre_discover(s, prevStateTerminal, a, r, ns, terminal)

        if terminal:
            # If THIS state is terminal:
            self.episodeTerminated()
