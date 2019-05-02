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
    
    def encode_state_action(self, state, action):
        state_list = state.to_list()
        processed_list = state_list + [0]*self.possible_action_count
        index = len(state_list) + action
        processed_list[index] = 1
        return [processed_list]
    
    def value(self, state, action):
        try:
            lis = self.encode_state_action(state, action)
            return self.estimator.predict(lis)
        except NotFittedError:
            return 50
        
    def observe(self, state, action, reward=None, state_prime=None, action_prime=None):
        lis = self.encode_state_action(state, action)

        error = -self.value(state, action)
        if reward is not None:
            error += reward
            self.estimator.partial_fit(lis, [self.discount_factor*reward])
        elif state_prime is not None and action_prime is not None:
            prime_value = self.discount_factor*self.value(state_prime, action_prime)
            error += prime_value
            self.estimator.partial_fit(lis, [prime_value])
        else:
            raise ValueError("A reward or derived state + action must be given")
        
        #return the error of the observation
        return abs(error)

        

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

    def state_action_to_key(self, state, action):
        temp_list = state.to_list()
        temp_list.append(action)
        keyStr = str(temp_list)
        return hash(keyStr)

    def value(self, state, action):
        key = self.state_action_to_key(state, action)
        if not key in self.Q:
            self.Q[key] = 50.0

        return self.Q[key]

    def observe(self, state, action, reward=None, state_prime=None, action_prime=None):
        key = self.state_action_to_key(state, action)
        
        if not key in self.Q:
            self.Q[key] = 50.0

        if reward is not None:
            self.Q[key] = self.Q[key] + self.learning_rate*(reward - self.Q[key])
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

    