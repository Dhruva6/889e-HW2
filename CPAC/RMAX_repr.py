"""Incrementally expanded Tabular Representation"""

from rlpy.Representations.Representation import Representation
import numpy as np
from copy import deepcopy
from sklearn.neighbors import LSHForest

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class RMAX_repr(Representation):
    """
    Identical to Tabular representation (ie assigns a binary feature function 
    f_{d}() to each possible discrete state *d* in the domain, with
    f_{d}(s) = 1 when d=s, 0 elsewhere.
    HOWEVER, unlike *Tabular*, feature functions are only created for *s* which
    have been encountered in the domain, not instantiated for every single 
    state at the outset.

    """
    def __init__(self, domain, Rmax, LQ, k = 1, epsilon_d = 0.01):
        # LQ is the lipschitz constant - 10**3 according to the paper (by Cross Validn)
        self.LQ = LQ
        self.gamma = domain.discount_factor

        self.rmax = Rmax
        self.qmax = Rmax / (1-self.gamma)
        self.qmax_tilda = Rmax + self.gamma * self.qmax
        self.epsilon = epsilon_d

        # Approximate k-NN is used when finding the Q value of a point
        self.k = k

        # We also keep track of the states sampled so far
        self.samples = {}

        # And we use an LSH to find the approximate k-Nearest neighbours
        # by training it on every s, a, r, s' tuple we see
        self.init_randomization()
        
        super(
            RMAX_repr,
            self).__init__(
            domain)

    def init_randomization(self):
        self.LSH = LSHForest(n_neighbors=self.k, random_state=self.random_state)

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return
    
    # def bestActions(self, s, terminal, p_actions, phi_s=None):
    #     return 1
    
    def Qs(self, s, terminal, phi_s=None):
        # Q -> Array of Q(s, a) values for this state
        # A -> Corresponding IDs
        num_a = self.actions_num
        Q = np.zeros(num_a) 
        return Q

    def featureType(self):
        return bool
