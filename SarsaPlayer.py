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
        self.sarsa_agents = {}
        self.role = 0
        self.error_over_time = 0
        self.error_lr = 0.0001

    def reset(self, match):
        self.sm = match.sm.dupe()
        self.role_count = len(match.sm.get_roles())
        self.role = match.our_role_index
        #Keep the estimator from previous game if the next game is the same
        if not self.match or not self.match.game_info.game == match.game_info.game:
            self.error_over_time = 0
            role_count = len(match.sm.get_roles())
            for role_index in range(role_count):
                self.sarsa_agents[role_index] = SarsaEstimator(SGDRegressor(loss='huber'), len(match.game_info.model.actions[role_index]))
                #self.sarsa_agents[role_index] = SarsaEstimator(MLPRegressor(hidden_layer_sizes=(50,50)), len(match.game_info.model.actions[role_index]))
                #self.sarsa_agents[role_index] = SarsaTabular()

        self.match = match

    def on_meta_gaming(self, finish_time):
        self.current_move = self.sm.get_joint_move()
        self.current_state = self.sm.new_base_state()
        self.last_move = self.sm.get_joint_move()
        self.last_state = self.sm.new_base_state()

        self.perform_sarsa(finish_time)


    #----SARSA
    
    def copy_state(self, copy_state, to_state = None):
        new_state = to_state
        if new_state is None: 
            new_state = self.sm.new_base_state()
        new_state.assign(copy_state)
        return new_state

    def copy_move(self, copy_move, to_move = None):
        new_move = to_move
        if new_move is None:
            new_move = self.sm.get_joint_move()

        for role_index in range(self.role_count):
            new_move.set(role_index, copy_move.get(role_index))
            
        return new_move
    
    #Choose moves for all players.
    def choose_moves(self):
        for role_index in range(self.role_count):
            choice = self.sarsa_agents[role_index].policy_training(self.current_state, self.sm.get_legal_state(role_index))
            self.current_move.set(role_index, choice)

    #Returns (state, move, reward) tuple.
    def create_history_tuple(self, current_state, current_move, terminal = False):
        hist_tuple = None
        hist_state = self.copy_state(current_state)
        hist_move = self.copy_move(current_move)

        reward = None
        if terminal:
            goal_list = []
            for role_index in range(self.role_count):
                goal_list.append(self.sm.get_goal_value(role_index))
            reward = goal_list

        hist_tuple = (hist_state, hist_move, reward)
        return hist_tuple
    
    #TODO: deallocate game_history after training
    def observe_history(self, game_history):
        history_size = len(game_history)
        for hist_ind in range(history_size - 1, -1, -1):
            current_tuple = game_history[hist_ind]
            for role_index in range(self.role_count):
                #If there's no reward, we aren't at terminal so we use s' a' 
                instance_error = 0
                if current_tuple[2] is None:
                    next_tuple = game_history[hist_ind + 1]
                    instance_error = self.sarsa_agents[role_index].observe(current_tuple[0], current_tuple[1].get(role_index), 
                                                        state_prime=next_tuple[0], action_prime=next_tuple[1].get(role_index))
                else:
                    instance_error = self.sarsa_agents[role_index].observe(current_tuple[0], current_tuple[1].get(role_index), 
                                                        reward=current_tuple[2][role_index])
                self.error_over_time += self.error_lr*(instance_error - self.error_over_time)
    def sarsa_playout(self, game_history):
        sm_terminal = self.sm.is_terminal()
        depth = 1
        while not sm_terminal:
            self.choose_moves()

            #Current move/action becomes last move/action
            self.last_state.assign(self.current_state)
            self.copy_move(self.current_move, self.last_move)

            #Update current_state with joint move
            self.sm.next_state(self.current_move, self.current_state)
            #Update the state machine
            self.sm.update_bases(self.current_state)

            sm_terminal = self.sm.is_terminal()
            #add last state + move to history. Also add goal value if terminal state 
            game_history.append(self.create_history_tuple(self.last_state, self.last_move, terminal = sm_terminal))

            depth += 1
        return depth

    #Playouts used by SARSA to learn policies before the game starts.
    def perform_sarsa(self, finish_time):
        #Playout related variables
        playout_count = 0

        #To ensure that we don't overshoot our training time
        time_offset = 0.05

        while time.time() + time_offset < finish_time:
            #Get game state & rewind the state machine
            self.match.sm.get_current_state(self.current_state)
            self.sm.update_bases(self.current_state)

            #Stores tuples of state, reward, action
            game_history = []

            depth = self.sarsa_playout(game_history)

            self.observe_history(game_history)

            print "Error over time: " + str(self.error_over_time)
            #Update average depth
            #self.average_depth = (playout_count*self.average_depth + depth) / float(playout_count + 1) 

            playout_count += 1
            
        return playout_count

    def on_next_move(self, finish_time):
        self.sm.update_bases(self.match.get_current_state())

        print "Managed ",  self.perform_sarsa(finish_time), "playouts."
        

        self.sm.update_bases(self.match.get_current_state())
        print "Printing action state values of current state..."

        for role_index in range(self.role_count):
            ls = self.sm.get_legal_state(role_index)
            print "****************************************************"
            if role_index == self.role:
                print "Us"
            else:
                print "Other"
            print "****************************************************"
            for act in range(ls.get_count()):
                state = self.sm.get_current_state()
                print "action ", self.match.game_info.model.actions[role_index][ls.get_legal(act)], " value is ", self.sarsa_agents[role_index].value(state, act)
            print ""
        
        ls = self.sm.get_legal_state(self.role)
        return self.sarsa_agents[self.role].policy(self.sm.get_current_state(), ls)
