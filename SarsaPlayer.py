import random
from Sarsa import SarsaEstimator
from Sarsa import SarsaTabular
import time
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

from ggplib.player.base import MatchPlayer

class SarsaPlayer(MatchPlayer):
    def __init__(self, name=None):
        super(SarsaPlayer, self).__init__(name)
        self.sarsaAgents = {}
        self.role = 0

    def reset(self, match):
        #Keep the estimator from previous game if the next game is the same
        if not self.match or not self.match.game_info.game == match.game_info.game:
            self.role = match.our_role_index
            role_count = len(match.sm.get_roles())
            for role_index in range(role_count):
                self.sarsaAgents[role_index] = SarsaEstimator(SGDRegressor(loss='huber'), len(match.game_info.model.actions[role_index]))
                #self.sarsaAgents[role_index] = SarsaEstimator(MLPRegressor(hidden_layer_sizes=(100,40,100)), len(match.game_info.model.actions[role_index]))
                #self.sarsaAgents[role_index] = SarsaTabular()

        self.match = match

    def on_meta_gaming(self, finish_time):
        self.do_playouts(finish_time)


    #TODO: multithread?
    def do_playouts(self, finish_time):
        #Playout related variables
        temp_sm = self.match.sm.dupe()
        role_count = len(temp_sm.get_roles())
        playout_count = 0

        #State machine variables
        current_move = temp_sm.get_joint_move()
        current_state = temp_sm.new_base_state()
        last_move = temp_sm.get_joint_move()
        last_state = temp_sm.new_base_state()

        #To ensure that we don't overshoot our training time
        time_offset = 0.01

        while time.time() + time_offset < finish_time:
            #Rewind the state machine
            temp_sm.update_bases(self.match.sm.get_current_state())
            #temp_sm.reset()

            current_state = temp_sm.get_current_state(current_state)
            init = True
            while time.time() + time_offset < finish_time and not temp_sm.is_terminal():

                #Choose moves for all players.
                for role_index in range(role_count):
                    choice = self.sarsaAgents[role_index].policy_training(current_state, temp_sm.get_legal_state(role_index))
                    current_move.set(role_index, choice)

                if not init:
                    #update all agents with bootstrapping
                    for role_index in range(role_count):
                        self.sarsaAgents[role_index].observe(last_state, last_move.get(role_index), state_prime=current_state, action_prime=current_move.get(role_index))
                else:
                     init = False

                #Current move/action becomes last move/action
                if not temp_sm.is_terminal():
                    last_state.assign(current_state)
                    for role_index in range(role_count):
                        last_move.set(role_index, current_move.get(role_index))


                # Update current_state with joint move
                temp_sm.next_state(current_move, current_state)
                # update the state machine
                temp_sm.update_bases(current_state)
            
            #update all agents with end reward
            for role_index in range(role_count):
                self.sarsaAgents[role_index].observe(last_state, last_move.get(role_index), reward=temp_sm.get_goal_value(role_index))

            playout_count += 1
        return playout_count

    def on_next_move(self, finish_time):
        sm = self.match.sm
        
        print "Managed ",  self.do_playouts(finish_time), "playouts."
        
        print "Printing action state values of current state..."
        role_count = len(sm.get_roles())
        for role_index in range(role_count):
            ls = sm.get_legal_state(role_index)
            print "****************************************************"
            if role_index == self.role:
                print "Us"
            else:
                print "Other"
            print "****************************************************"
            for act in range(ls.get_count()):
                state = sm.get_current_state()
                print "action ", self.match.game_info.model.actions[role_index][ls.get_legal(act)], " value is ", self.sarsaAgents[role_index].value(state, act)
            print ""
        
        ls = sm.get_legal_state(self.role)
        return self.sarsaAgents[self.role].policy(sm.get_current_state(), ls)
