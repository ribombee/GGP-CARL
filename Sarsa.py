import random
import sys
from sklearn.exceptions import NotFittedError
import numpy as np
from numpy import asarray
from numpy import reshape

class SarsaEstimator:
    def __init__(self, estimator, epsilon = 0.1, discount_factor = 0.9):
        self.estimator = estimator
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.temp_list = None

    def populateTempList(self, state, action):
        if self.temp_list is None:
            self.temp_list = state.to_list()
            self.temp_list.append(action)
            self.temp_list = [self.temp_list]
            #self.temp_list = np.asarray(self.temp_list)
        elif state is not None:
            for ind in range(state.len()):
                self.temp_list[0][ind] = state.get(ind) if ind < state.len() - 1 else action

    def value(self, state, action):
        try:
            self.populateTempList(state, action)
            return self.estimator.predict(self.temp_list)
        except NotFittedError:
            return 0
        return 0
        

    #TODO: find a better way to fit data. As is, we append action as integer to state list (big nono).
    def observe(self, state, action, reward=None, state_prime=None, action_prime=None):
        self.populateTempList(state, action)
        if reward is not None:
            self.estimator.partial_fit(self.temp_list, [self.discount_factor*reward])
        elif state_prime is not None and action_prime is not None:
            self.estimator.partial_fit(self.temp_list, [self.discount_factor*self.value(state_prime, action_prime)])
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



class SarsaTabular:
    def __init__(self, epsilon = 0.1, discount_factor = 1, learning_rate = 0.1):
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
            #print "test, action: ", action, "value: ", self.value(state_prime, action_prime)
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