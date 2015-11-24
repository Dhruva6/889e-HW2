"""Incrementally expanded Tabular Representation"""

from rlpy.Representations import Representation, IncrementalTabular
import numpy as np
from copy import deepcopy

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
    RMax = 1e09

    # the upper bound on the sum of discounted future rewards
    VMax = RMax

    # the minimum variance in the data below which exploitation is prefered over exploration
    minVarToExplore = 1;

    # the reward and transition learners
    transitionLearners = []
    rewardLearners = []

    # can the learners be used for computing Q values
    canUseLearners = False
       
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

        # update VMax
        self.VMax = self.RMax/(1-self.domain.discount_factor)


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

    def setLearners(self, transitionLearners, rewardLearners):
        self.transitionLearners = transitionLearners
        self.rewardLearners = rewardLearners

    def setCanUseLearners(self):
        self.canUseLearners = True
    
    def Qs(self, s, terminal, phi_s=None):

        # start off with Q being 0
        Q = np.zeros(self.domain.actions_num)

        # catch
        if self.canUseLearners == False:
            return Q

        # the reward 
        rewards = np.zeros(len(s))

        # iterate through all the actions
        for aIdx in range(self.domain.actions_num):

            actIdx = aIdx * len(s)
            
            # predict the rewards for the next state
            for i in range(len(s)):
                rewards[i], _, _, _, _ = self.rewardLearners[actIdx+i].predict(np.ones((1,1)) * s[i])

            # the mean reward of all state dimensions
            Q[aIdx] = np.mean(rewards)        

        # the predicted state transition delta
        ds = np.zeros(len(s))

        # the variance in the delta prediction 
        dsVar = np.zeros(len(s))

        # predicted variance score
        predVarScore = np.zeros(self.domain.actions_num)
        
        # the current state
        currState = s

        # the discount factor
        gamma = self.domain.discount_factor

        numSteps = 5
        for steps in range(numSteps):
            
            #
            # choose the best action, find the next state
            #
            bestAction = np.argmax(Q)
 
            # the index into the transition learners to predict the next action
            actIdx = bestAction * len(s)

            # predict the transition for the current action from the current state
            for i in range(len(s)):
                ds[i],dsVar[i], _, _, _ = self.transitionLearners[actIdx+i].predict(np.ones((1,1)) * currState[i])

                # scale dsVar
                dsVar[i] = dsVar[i]/self.minVarToExplore

                if dsVar[i] > 1.0:
                    dsVar[i] = 1.0
                    
            # the new state is the sum of the current state plus or transition
            currState = s + ds

            # update the discount factor
            gamma *= gamma
            
            # iterate through all the actions
            for aIdx in range(self.domain.actions_num):

                actIdx = aIdx * len(s)
                        
                # predict the rewards for the next state
                for i in range(len(s)):
                    rewards[i], _, _, _, _ = self.rewardLearners[actIdx+i].predict(np.ones((1,1)) * currState[i])

                # update single step
                Q[aIdx] = (1-dsVar[i]) * (Q[aIdx] + gamma * np.mean(rewards)) + (dsVar[i] * self.VMax)

        return Q


    # def Q_oneStepLookAhead(self, s, a, ns_samples, policy=None, numSteps=1):
    #     """
    #     Returns the state action value, Q(s,a), by performing n step
    #     look-ahead on the domain.

    #     .. note::
    #         For an example of how this function works, see
    #         `Line 8 of Figure 4.3 <http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html>`_
    #         in Sutton and Barto 1998.

    #     .. note::
    #         This function should not be called in any RL algorithms unless
    #         the underlying domain is an approximation of the true model.

    #     :param s: The given state
    #     :param a: The given action
    #     :param ns_samples: The number of samples used to estimate the one_step look-ahead.
    #     :param policy: (optional) Used to select the action in the next state
    #         (*after* taking action a) when estimating the one_step look-aghead.
    #         If ``policy == None``, the best action will be selected.

    #     :return: The one-step lookahead state-action value, Q(s,a).
    #     """
    #     discount_factor = self.domain.discount_factor

    #     #
    #     # compute the N-step lookahead state-action value
    #     #

    #     # the predicted state transition delta
    #     ds = np.zeros(len(s))

    #     # the variance in the delta prediction 
    #     dsVar = np.zeros(len(s))

    #     # the reward
    #     reward = np.zeros(len(s))

    #     # the current state and action
    #     currState = s;
    #     currAction = a;

    #     # compute the mean reward for the current state
    #     for i in range(len(s)):
    #         reward[i], _, _, _, _ = self.rewardLearners[a*len(s)+i].predict(np.ones((1,1)) * currState[i])

    #     # initialize the reward for the current step
    #     currReward = np.mean(reward)

    #     # cumulative reward
    #     cumulativeReward = currReward

    #     # the next reward
    #     nextReward = np.zeros(self.domain.actions_num)

    #     # the next state
    #     nextState = np.zeros((self.domain.actions_num, len(s)))

    #     # predicted variance score
    #     predVarScore = np.zeros(self.domain.actions_num)
        
    #     # iterate through and compute the approximate Q value for n-steps 
    #     for steps in range(numSteps):

    #         # iterate through all the actions
    #         for aIdx in range(self.domain.actions_num):

    #             # the action index
    #             actIdx = aIdx * len(s)
                
    #             # predict the transition for the current action from the current state
    #             for i in range(len(s)):
    #                 ds[i],dsVar[i], _, _, _ = self.transitionLearners[actIdx+i].predict(np.ones((1,1)) * currState[i])
                
    #             # score of the stacked predicted variance
    #             predVarScore[aIdx] =  np.max(dsVar)/self.minVarToExplore

    #             # upper bound the score
    #             if predVarScore[aIdx] > 1.0:
    #                 predVarScore[aIdx] = 1.0
                
    #             # the next state
    #             nextState[aIdx][:] = currState + ds
                
    #             # predict the rewards for the next state
    #             for i in range(len(s)):
    #                 reward[i], _, _, _, _ = self.rewardLearners[actIdx+i].predict(np.ones((1,1)) * nextState[aIdx][i])

    #             # the next state reward
    #             nextReward[aIdx] = discount_factor * np.mean(reward)
            
    #         # the predicted Q value for the next state
    #         currReward = (1-np.max(predVarScore)) * (currReward + np.max(nextReward)) #+ np.max(predVarScore) * self.VMax

    #         # print predVarScore
    #         # print nextReward
    #         # assert False
            
    #         # update cumulativeReward
    #         cumulativeReward += currReward
            
    #         # to the next state
    #         currState = currState + nextState[np.argmax(nextReward)][:]

    #         # update the discount factor
    #         discount_factor *= discount_factor

    #     return cumulativeReward
