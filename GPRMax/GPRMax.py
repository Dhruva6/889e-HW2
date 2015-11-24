"""Control Agents based on TD Learning, i.e., Q-Learning and SARSA"""
from rlpy.Agents.Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
from rlpy.MDPSolvers import TrajectoryBasedPolicyIteration
from random import sample
import pyGPs
import numpy as np
import collections

__copyright__ = ""
__credits__ = [""]
__license__ = "LGPL"


class GPRMax(DescentAlgorithm, Agent):

    #
    # A crude implementation of the paper "Gaussian process for sample efficient 
    # reinforcement learning with RMAX-like exploration" by
    # Tobias Jung and Peter Stone
    
    # the current iteration
    tick = 0

    # the number of iterations to skip before training the GP's
    trainEveryNSteps = 50;

    # the local copy of state, difference between next state and current state, action and rewards
    statesCache = collections.deque(maxlen=1e5)
    diffStatesCache = collections.deque(maxlen=1e5)
    actionsCache = collections.deque(maxlen=1e5)
    rewardsCache = collections.deque(maxlen=1e5)

    #
    # the number of GP's depend on the dimensionality of the state and action space
    #

    # the dimensionality of the actions
    actionsDim = 1;

    # the dimensionality of the state
    statesDim = 1;

    # the list of Gaussian process objects - a total of actionsDim * statesDim 
    transitionLearners = [];

    # the reward model learner
    rewardLearners = [];

    # the histogram of actions seen so far
    actionsHist = np.zeros(1);   

    def __init__(self, policy, representation, actionsDim, statesDim, discount_factor, lambda_=0, **kwargs):
        super(
            GPRMax,
            self).__init__(policy=policy,
            representation=representation, discount_factor=discount_factor, **kwargs)

        # update the dimensionality of the actions space
        self.actionsDim = actionsDim

        # update the dimensionality of the state space
        self.statesDim = statesDim

        # instantiate the histogram of actions seen so far
        self.actionHist = np.zeros(self.actionsDim)

        # instantiate transitionLearners
        self.transitionLearners = [pyGPs.GPR_FITC() for i in range(self.statesDim * self.actionsDim)]

        # instantiate rewardLearners
        self.rewardLearners = [pyGPs.GPR_FITC() for i in range(self.statesDim * self.actionsDim)]

        # pass along the learners 
        self.representation.setLearners(self.transitionLearners, self.rewardLearners)

        # trajectory based policy iteration
        self.jobId = 1;
        self.trajPI = TrajectoryBasedPolicyIteration(self.jobId, self.representation, self.representation.domain, planning_time=3) 

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        # set prevStateTerminal to false and do the necessary calls to pre_discover
        prevStateTerminal = False
        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

        # tick up the currentIteration
        self.tick += 1
      
        # print for debug
        if 0 == self.tick%100:
            self.logger.info("Current iteration %d" % self.tick);

        # update the actionsHist
        self.actionHist[a] += 1
        
        # add s, a, r to the statesCache, actionsCache and rewardsCache
        self.statesCache.append(s)
        self.actionsCache.append(a)
        self.diffStatesCache.append(ns-s)
        self.rewardsCache.append(r)

        # have the transistion and reward models been updated this iteration
        modelUpdated = False
        
        # if it is time to learn - currently not limiting the number of samples
        if self.tick % self.trainEveryNSteps == 0:
            
            # set the modelUpdated flag
            modelUpdated = True

            # dump the action distribution of samples seen so far
            self.logger.info("Training on %d samples.\nAction Histogram" % self.tick)
            self.logger.info(self.actionHist)
            
            # for every action, gather all the relevant states, diffStates and train
            for actIdx in range(self.actionsDim):

                # the start index for the transitionLearners
                gpStartIdx = actIdx * self.statesDim

                # find all the indexes in the actionsCache that match actIdx
                indexes = [i for i,x in enumerate(self.actionsCache) if x == actIdx]

                # if the indexes list is empty, the model has not been updated
                if indexes == []:
                    modelUpdated =  modelUpdated and False
                    continue

                # allocate memory for the input, output
                x = np.zeros(len(indexes))
                y = np.zeros(len(indexes))
                rwd = np.zeros(len(indexes))

                # setup the input and output
                for dim in range(self.statesDim):

                    # iterate through, populate the inputs and outputs
                    for idx,ii in zip(indexes, range(len(x))):
                        x[ii] = self.statesCache[idx][dim]
                        y[ii] = self.diffStatesCache[idx][dim]
                        rwd[ii] = self.rewardsCache[idx]

                    # pass the data along and learn 
                    self.transitionLearners[gpStartIdx+dim].setData(x,y)
                    self.transitionLearners[gpStartIdx+dim].optimize()

                    # pass the data along and learn
                    self.rewardLearners[gpStartIdx+dim].setData(x,rwd)
                    self.rewardLearners[gpStartIdx+dim].optimize()
 
        # If learning happened in this iteration, update the model that will be used
        if modelUpdated:
            self.representation.setCanUseLearners()
            #self.trajPI.solve()
        
        # expanded = self.representation.post_discover(
        #     s,
        #     prevStateTerminal,
        #     a,
        #     td_error,
        #     phi_s)

        if terminal:
            # If THIS state is terminal:
            self.episodeTerminated()
