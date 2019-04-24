import random
import sys
from sklearn.exceptions import NotFittedError
from scipy.special import softmax
import numpy as np
from numpy import asarray
from numpy import reshape
from numpy.random import choice

class SarsaEstimator:
    def __init__(self, estimator, possible_action_count, epsilon = 0.1, discount_factor = 0.9):
        self.estimator = estimator
        self.epsilon = epsilon
        self.discount_factor = discount_factor

        self.possible_action_count = possible_action_count
        #We will only need 2 lists allocated at each time
        #TODO: check to make sure that more than 1 is needed
        self.temp_list = [[],[]]
        self.offset = 0

    def populateTempList(self, state, action):
        self.offset = (self.offset + 1) % 2
        if len(self.temp_list[self.offset]) == 0:
            self.temp_list[self.offset] = state.to_list()
            action_list = [0]*self.possible_action_count
            action_list[action] = 1
            self.temp_list[self.offset] = self.temp_list[self.offset] + action_list
            self.temp_list[self.offset] = [self.temp_list[self.offset]]
        elif state is not None:
            for ind in range(state.len()):
                self.temp_list[self.offset][0][ind] = state.get(ind)
            for ind in range(state.len(), state.len()+self.possible_action_count):
                if ind - state.len() == action:
                    self.temp_list[self.offset][0][ind] = 1
                else:
                    self.temp_list[self.offset][0][ind] = 0
        return self.temp_list[self.offset]

    def value(self, state, action):
        try:
            lis = self.populateTempList(state, action)
            return self.estimator.predict(lis)
        except NotFittedError:
            return 0
        
    def observe(self, state, action, reward=None, state_prime=None, action_prime=None):
        lis = self.populateTempList(state, action)
        if reward is not None:
            #TODO: fix reward system
            self.estimator.partial_fit(lis, [self.discount_factor*(reward-50)])
        elif state_prime is not None and action_prime is not None:
            self.estimator.partial_fit(lis, [self.discount_factor*self.value(state_prime, action_prime)])
        else:
            raise ValueError("A reward or derived state + action must be given")

    def policy(self, state, legal_state):
        highest_value = float('-inf')
        highest_index = -sys.maxint - 1

        for ind in range(legal_state.get_count()):
            q_val = self.value(state, legal_state.get_legal(ind))
            if q_val > highest_value:
                highest_value = q_val
                highest_index = ind

        return legal_state.get_legal(highest_index)

    def nondet_policy(self, state, legal_state):
        distribution = self.policy_distribution(state, legal_state)
        distribution = softmax(distribution)
        selected = choice(legal_state.get_count(), 1, p=distribution)
        return legal_state.get_legal(selected)
    
    def policy_distribution(self, state, legal_state):
        distribution = []
        for ind in range(legal_state.get_count()):
            distribution.append(self.value(state,legal_state.get_legal(ind))[0])
        
        return distribution

    #During the training phase, we use epsilon-greedy selection.
    def policy_training(self, state, legal_state):
        if random.random() > self.epsilon:
            return self.policy(state, legal_state)
        else:
            return legal_state.get_legal(random.randrange(0, legal_state.get_count()))



class SarsaTabular:
    def __init__(self, epsilon = 0.1, discount_factor = 0.9, learning_rate = 0.1):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.Q = {}

    def stateActionToKey(self, state, action):
        temp_list = state.to_list()
        temp_list.append(action)
        keyStr = str(temp_list)
        return hash(keyStr)

    def value(self, state, action):
        key = self.stateActionToKey(state, action)
        if not key in self.Q:
            self.Q[key] = 0.0

        return self.Q[key]

    def observe(self, state, action, reward=None, state_prime=None, action_prime=None):
        key = self.stateActionToKey(state, action)
        
        if not key in self.Q:
            self.Q[key] = 0.0

        if reward is not None:
            self.Q[key] = self.Q[key] + self.learning_rate*((reward-50) - self.Q[key])
        elif state_prime is not None and action_prime is not None:
            self.Q[key] = self.Q[key] + self.learning_rate*(self.discount_factor * self.value(state_prime, action_prime) - self.Q[key])
        else:
            raise ValueError("A reward or derived state + action must be given")

    def policy(self, state, legal_state):
        highest_value = float('-inf')
        highest_index = -sys.maxint - 1

        for ind in range(legal_state.get_count()):
            q_val = self.value(state, legal_state.get_legal(ind))
            if q_val > highest_value:
                highest_value = q_val
                highest_index = ind
        return legal_state.get_legal(highest_index)

    def policy_training(self, state, legal_state):
        if random.random() > self.epsilon:
            return self.policy(state, legal_state)
        else:
            return legal_state.get_legal(random.randrange(0, legal_state.get_count()))

    