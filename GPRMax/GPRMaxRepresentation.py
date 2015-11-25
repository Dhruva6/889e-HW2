"""Incrementally expanded Tabular Representation"""

from rlpy.Representations import Representation, IncrementalTabular
import numpy as np
from copy import deepcopy
import math
from sklearn.neighbors import KDTree


__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class GPRMaxRepresentation(IncrementalTabular):
    """
    Identical to Tabular representation (ie assigns a binary feature function 
    f_{d}() to each possible discrete state *d* in the domain, with
    f_{d}(s) = 1 when d=s, 0 elsewhere.
    HOWEVER, unlike *Tabular*, feature functions are only created for *s* which
    have been encountered in the domain, not instantiated for every single 
    state at the outset.

    """
    hash = None

    # the upper bound on the Rewards
    RMax = 1e05

    #
    discountFactor = 0.9
    
    # the upper bound on the sum of discounted future rewards
    VMax = RMax/(1-discountFactor)

    # the minimum variance in the data below which exploitation is prefered over exploration
    minVarToExplore = 1;

    # the reward and transition learners
    transitionLearners = []
    rewardLearners = []

    # can the learners be used for computing Q values
    canUseLearners = False

    # just consider a single step when predicting Q values
    singleStep = False

    # is the cache valid
    cacheValid = False

       
    def __init__(self, domain, discretization=20):
        self.hash = {}
        self.features_num = 0
        self.isDynamic = True
        self.canUseLearners = False
        super(
            GPRMaxRepresentation,
            self).__init__(
            domain,
            discretization)


    def phi_nonTerminal(self, s):
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        F_s = np.zeros(self.features_num, bool)
        if hashVal is not None:
            F_s[hashVal] = 1
        return F_s

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return self._add_state(s) + self._add_state(sn)

    def _add_state(self, s):
        """
        :param s: the (possibly un-cached) state to hash.
        
        Accepts state ``s``; if it has been cached already, do nothing and 
        return 0; if not, add it to the hash table and return 1.
        
        """
        
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        if hashVal is None:
            # New State
            self.features_num += 1
            # New id = feature_num - 1
            hashVal = self.features_num - 1
            self.hash[hash_id] = hashVal
            # Add a new element to the feature weight vector, theta
            self.addNewWeight()
            return 1
        return 0

    def __deepcopy__(self, memo):
        new_copy = GPRMaxRepresentation(
            self.domain,
            self.discretization)
        new_copy.hash = deepcopy(self.hash)
        return new_copy

    def featureType(self):
        return bool

    def setCache(self, fullStateCache, qValCache):
        self.fullStatesCache = deepcopy(fullStateCache)
        self.qValCache = deepcopy(qValCache)
        self.cacheValid = True

    def Qs(self, s, terminal, phi_s=None):

        # start off with Q being 0
        Q = np.zeros(self.domain.actions_num)

        # catch
        if self.cacheValid == False:
            return Q

        # simple linear search
        minIdx = 0
        minDist = 1e9
        for i in range(len(self.fullStatesCache)):
            d = np.linalg.norm(np.subtract(self.fullStatesCache[i], np.array(s)))
            if d < minDist:
                minDist = d
                minIdx = i

        return self.qValCache[minIdx][:]
        
