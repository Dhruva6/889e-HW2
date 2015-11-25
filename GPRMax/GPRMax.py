"""Control Agents based on TD Learning, i.e., Q-Learning and SARSA"""
from rlpy.Agents.Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
import GPy
import numpy as np
import collections
import math
from sklearn.neighbors import NearestNeighbors

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
    trainEveryNSteps = 200

    # the maximum length of the queue
    maxQLen = 25000

    # the local copy of state, difference between next state and current state, action and rewards
    statesCache = collections.deque(maxlen=maxQLen)
    diffStatesCache = collections.deque(maxlen=maxQLen)
    actionsCache = collections.deque(maxlen=maxQLen)
    rewardsCache = collections.deque(maxlen=maxQLen)

    #
    # the number of GP's depend on the dimensionality of the state and action space
    #

    # the dimensionality of the actions
    actionsDim = 1;

    # the dimensionality of the state
    statesDim = 1;

    # the list of Gaussian process objects - a total of actionsDim * statesDim 
    transitionLearners = [];

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
        self.kernel = GPy.kern.RBF(1, 5, 1)

        self.discountFactor = discount_factor

        # the Q Dictionary
        
    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        # set prevStateTerminal to false and do the necessary calls to pre_discover
        prevStateTerminal = False
        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)

        # tick up the currentIteration
        self.tick += 1
      
        # # print for debug
        # if 0 == self.tick%100:
        #     self.logger.info("Current iteration %d" % self.tick);

        # update the actionsHist
        self.actionHist[a] += 1
        
        # add s, a, r to the statesCache, actionsCache and rewardsCache
        self.statesCache.append(s)
        self.actionsCache.append(a)
        self.diffStatesCache.append((ns-s))
        self.rewardsCache.append(r)

        qVal = np.zeros((1,self.actionsDim))
        qVal[0][a] = r

        #
        if self.tick == 1:
            self.fullStatesCache = np.atleast_2d(s)
            self.qValCache = np.atleast_2d(qVal)
        else:
            self.fullStatesCache = np.concatenate((self.fullStatesCache, np.atleast_2d(s)))
            self.qValCache = np.concatenate((self.qValCache, np.atleast_2d(qVal)))

        # have the transistion and reward models been updated this iteration
        modelUpdated = False

        # if it is time to learn - currently not limiting the number of samples
        if self.tick % self.trainEveryNSteps == 0:

            # set the modelUpdated flag
            modelUpdated = True

            # dump the action distribution of samples seen so far
            #self.logger.info("Training on %d samples." % self.tick)
            #self.logger.info(self.actionHist)

            # clear out the transition learners
            self.transitionLearners = []
            
            # for every action, gather all the relevant states, diffStates and train
            for actIdx in range(self.actionsDim):

                # the start index for the transitionLearners
                gpStartIdx = actIdx * self.statesDim

                # find all the indexes in the actionsCache that match actIdx
                indexes = [i for i,x in enumerate(self.actionsCache) if x == actIdx]

                # if the indexes list is empty, the model has not been updated
                if indexes == []:
                    modelUpdated =  modelUpdated and False
                    self.transitionLearners = []
                    break

                # allocate memory for the input, output
                X = np.zeros((len(indexes),1))
                y = np.zeros((len(indexes),1))

                # setup the input and output
                for dim in range(self.statesDim):

                    # iterate through, populate the inputs and outputs
                    for idx,ii in zip(indexes, range(len(indexes))):
                        X[ii][0] = self.statesCache[idx][dim]
                        y[ii][0] = self.diffStatesCache[idx][dim]

                    # pass the data along and learn
                    self.transitionLearners.append(GPy.models.SparseGPRegression(X,y))
                    self.transitionLearners[gpStartIdx+dim].optimize()

            # learn the reward GP
            X = np.atleast_2d(np.array(self.statesCache))
            rwd = np.atleast_2d(np.array(self.rewardsCache)).T

            self.rewardLearners = GPy.models.SparseGPRegression(X,rwd)
            self.rewardLearners.optimize()

            # now iterate through and update the Q values for all the entries that we have
            # seen so-far
 
        if modelUpdated == True:

            # iterate through all the stored state and update their respective Q values
            for ii in range(len(self.fullStatesCache)):

                # instantiate the current state
                currState = self.fullStatesCache[ii][:]

                # instantiate the gamma
                gamma = self.discountFactor

                # the delta state
                ds = np.zeros(len(currState))
                
                # iterate through all the actions
                for aIdx in range(self.actionsDim):

                    actIdx = aIdx * len(currState)

                    # predict the rewards for the next state
                    for idx in range(len(currState)):
                        yp = self.transitionLearners[actIdx+idx].predict(np.ones((1,1))*currState[idx])[0]
                        ds[idx] = yp.flatten()

                    # predict the reward for the delta state ds[i]+s[i]
                    ns = np.atleast_2d(np.array(currState + ds))
                    yp = self.rewardLearners.predict(ns)[0]

                    self.qValCache[ii][aIdx] +=  yp.flatten()
                    
            # pass along the cache, the learners
            self.representation.setCache(self.fullStatesCache, self.qValCache)

 
        if terminal:
            # If THIS state is terminal:
            self.episodeTerminated()
