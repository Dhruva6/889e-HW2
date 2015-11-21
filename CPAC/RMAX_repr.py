"""Incrementally expanded Tabular Representation"""

from rlpy.Representations.Representation import Representation
import numpy as np
from copy import deepcopy
from sklearn.neighbors import LSHForest
from scipy.spatial.distance import chebyshev as distance

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
        self.sample_list = []
        # And a dictionary for quick lookups of already computed values
        self.sample_values = {}

        # And we use an LSH to find the approximate k-Nearest neighbours
        # by training it on every s, a, r, s' tuple we see
        self.init_randomization()
        
        super(
            RMAX_repr,
            self).__init__(
            domain)

    def init_randomization(self):
        self.LSH = LSHForest(n_neighbors=self.k, random_state=self.random_state)

    def is_known(self, s, a):
        # A s, a pair is 'known' if LQ * d(s, a, s', a') < epsilon_d
        try:
            indices = self.approx_nn(s, a)
        except ValueError:
            return False

        for idx in indices:
            s_p, a_p = self.sample_list[idx]
            if self.LQ * self.d(s, a, s_p, a_p) > self.epsilon:
                return False
        return True

    def pre_discover(self, s, p_terminal, a, r, ns, terminal):
        # In the learning stage, if sa is not 'known' add it to the sample list
        # and its value to sample value.
        if not self.is_known(s, a):
            self.sample_list.append((s, a))
            x = r + self.gamma * max(self.Q_tilda(ns, a_p) for a_p in range(self.actions_num))
            self.sample_values[self.sa_tuple(s, a)] = x
            self.LSH.partial_fit(np.append(s, a))
        super(RMAX_repr, self).pre_discover(s, p_terminal, a, ns, terminal)

    # Compute a distance metric between (s, a) and (ns, na).
    # Using max-norm as in the paper for now.
    def d(self, s, a, ns, na):
        # Create one big s,a array
        sa = np.append(s, a)
        nsa = np.append(ns, na)
        # Use scipy to compute the chebyshev distance => Max norm
        return distance(sa, nsa)

    def approx_nn(self, s, a):
        dist, indices = self.LSH.kneighbors(np.append(s, a))
        return indices

    def sa_tuple(self, s, a):
        return tuple(np.append(s, a))
    
    # The approximate Q function 
    def Q_tilda(self, s, a):
        k = self.k
        # First get the k-nearest sampled neighbours to this point using LSH
        try:
            indices = self.approx_nn(s, a)
        except ValueError:
            indices = []
        q = 0.0
        num_neighbors = 0

        for index in indices:
            sj, aj = self.sample_list[index]
            dij = self.d(s, a, sj, aj)
            if dij <= (self.qmax / self.LQ):
                xj = self.sample_values[self.sa_tuple(sj, aj)]
                q += dij * self.LQ + xj
                num_neighbors += 1

        # In case there were less than k neighbors - Use Qmax_tilda for the remaining
        for i in range(num_neighbors, k):
            q += self.qmax_tilda
        # Return the average Q
        return q/k
        

    def Qs(self, s, terminal, phi_s=None):
        # Q -> Array of Q(s, a) values for this state
        # A -> Corresponding IDs

        # Before any learning is done, the experiment calls the policy to
        # estimate prior performance. In that case, the LSHF would throw a 
        # Value Error. We pre-empt that here
        Q = np.zeros((self.actions_num))
        try :
            self.LSH.kneighbors(np.append(s, 0))
        except ValueError:
            return Q
    
        for a in range(self.actions_num):
            Q[a] = self.Q_tilda(s, a)
        return Q

    def featureType(self):
        return bool
