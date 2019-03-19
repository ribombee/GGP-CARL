import random
import sys
from sklearn.exceptions import NotFittedError
import numpy as np
from numpy import asarray
from numpy import reshape

class Sarsa:
    def __init__(self, estimator, epsilon = 0.1, discount_rate = 0.9):
        self.estimator = estimator
        self.epsilon = epsilon
        self.discount_rate = discount_rate

    def value(self, state, action):
        try:
            data_tuple = state.to_list()
            data_tuple.append(action)
            data_tuple = [data_tuple]
            return self.estimator.predict(data_tuple)
        except NotFittedError:
            return 0
        

    #TODO: find a better way to fit data. As is, we append action as integer to state list (big nono).
    def observe(self, state, action, reward=None, state_prime=None):
        data_tuple = state.to_list()
        data_tuple.append(action)
        data_tuple = [data_tuple]
        if reward is not None:
            self.estimator.partial_fit(data_tuple, [self.discount_rate*reward])
        elif state_prime is not None:
            self.estimator.partial_fit(data_tuple, [self.discount_rate*self.value(state_prime, action)])
        else:
            raise ValueError("A reward or derived state must be given")

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
        if random.random > self.epsilon:
            return self.policy(state, legal_state)
        else:
            return legal_state.get_legal(random.randrange(0, legal_state.get_count()))