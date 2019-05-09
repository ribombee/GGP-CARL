import random
import time
import datetime
import math
import csv

from Sarsa import SarsaEstimator
from sklearn.linear_model import SGDRegressor
from sklearn.base import clone

from ggplib.player.mcs import MoveStat
from ggplib.player.base import MatchPlayer
from ggplib import interface
import CarlUtils
from CarlUtils import Action, Node, tree_cleanup, hash_joint_move

class CarlPlayer(MatchPlayer):
    #General match variables
    role_count = 0
    role = 0
    sm = None
    
    #Current node information
    current_move = None
    current_state = None
    last_move = None
    last_state = None

    #Sarsa variables
    sarsa_agents = {}
    average_depth = 0
    average_branching_factor = 0
    sarsa_iterations = 0
    sarsa_expansions = 0
    sarsa_error = 0
    sarsa_error_lr = 0.0001
    
    #MCTS variables
    root = None
    selection_policy = None
    playout_policy = None
    ucb_constant = 1.414
    max_expansions = -1

    #Logging info
    csv_log_file = "PlayerLog.csv"
    iteration_count_list = []
    time_list = []
    
    
    #----Helper functions

    #Returns move name as string
    def get_move_name(self, role, move):
        return self.match.game_info.model.actions[role][move]

    #Print which move each player chose
    def print_joint_move(self,joint_move):
        for role_index in range(self.role_count):
            print "Player no.", role_index, " chose ", self.get_move_name(role_index, joint_move.get(role_index))


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
    
    #Choose moves in sarsa for all players. Returns average no. of moves per player.
    def choose_moves(self):
        move_count = 1
        for role_index in range(self.role_count):
            legal_state = self.sm.get_legal_state(role_index)
            move_count *= legal_state.get_count()

            choice = self.sarsa_agents[role_index].policy_training(self.current_state, legal_state)
            self.current_move.set(role_index, choice)
        return move_count

    #Returns (state, move, reward) tuple for sarsa.
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
                instance_error = abs(instance_error)
                self.sarsa_error += self.sarsa_error_lr*(instance_error - self.sarsa_error)

    def sarsa_playout(self, game_history):
        sm_terminal = self.sm.is_terminal()
        depth = 1
        game_branching_factor = 0
        while not sm_terminal:
            game_branching_factor += self.choose_moves()

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
            
        game_branching_factor = game_branching_factor/float(depth)
        return depth, game_branching_factor

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

            game_depth, game_branching_factor = self.sarsa_playout(game_history)

            self.observe_history(game_history)

            #Update average depth
            self.average_depth = (playout_count*self.average_depth + game_depth) / float(playout_count + 1) 
            self.average_branching_factor =(playout_count*self.average_branching_factor + game_branching_factor) / float(playout_count + 1) 
            self.sarsa_expansions += game_depth
            playout_count += 1
        
        return playout_count


    #----MCTS

    #Selection phase of the mcts. 
    #We move down the tree, selecting actions with the highest ucb values 
    #until we reach terminal state or an unexpanded node.
    def mcts_selection(self, root):
        last_node = root
        current_node = root
        next_move = self.sm.get_joint_move()

        while current_node is not None and len(current_node.actions) is not 0:
            self.selection_policy.choose(next_move, self.sm, current_node=current_node, current_state=current_node.state)
            
            last_node = current_node
            
            current_node = current_node.getChild(next_move, self.role_count)       
        
        return last_node, next_move

    #Expansion phase of MCTS. We expand a selected node with a selected action.
    def mcts_expansion(self, selected_node = None, next_move = None):
        if selected_node is not None:
            #Set statemachine state to leaf node state
            self.sm.update_bases(selected_node.state)
        
        next_state = self.sm.get_current_state()
        if next_move is not None:
            #Create state for expanded node
            self.sm.next_state(next_move, next_state)

            #Move statemachine to the expanded state
            self.sm.update_bases(next_state)

        new_node = Node(next_state, parent=selected_node, parent_move=next_move)

        if selected_node is not None:
            #Add our new node to selected node children
            selected_node.children[hash_joint_move(self.role_count, next_move)] = new_node

        if not self.sm.is_terminal():
            for role in range(self.role_count):
                new_node.actions.append({})
                legal_state = self.sm.get_legal_state(role)
                for action_index in range(legal_state.get_count()):
                    action = legal_state.get_legal(action_index)
                    new_node.actions[role][action] = Action(action)
                    
        return new_node
        
    #Simulation phase of MCTS.
    #We play randomly for each player until we reach a terminal state.
    def mcts_playout(self):
        current_move = self.sm.get_joint_move()
        current_state = self.sm.new_base_state()
        while True:
            self.mcts_expansions += 1
            if self.sm.is_terminal():
                break

            current_state = self.sm.get_current_state(current_state)

            self.playout_policy.choose(current_move, self.sm, current_state = current_state)
            #Update state + state machine
            self.sm.next_state(current_move, current_state)
            self.sm.update_bases(current_state)


    #Use the value of the terminal state from playout to update Q values for each visited node.
    def mcts_backpropagation(self, tree_node):
        while not (tree_node.parent_move == None or tree_node.parent == None):
            for role in range(self.role_count):
                index = tree_node.parent_move.get(role)
                action = tree_node.parent.actions[role][index]
                #Update the running Q value average
                action.Q = (action.Q*action.N + self.sm.get_goal_value(role))/(float(action.N+1.0))
                action.N += 1
            tree_node.N +=1
            tree_node = tree_node.parent
        self.root.N += 1

    #Performs each of the four phases of MCTS: Select, expand, simulate and backpropagate.
    def perform_mcts(self, finish_by):
        self.mcts_runs = 0
        root_state = self.sm.get_current_state()
        self.sm.update_bases(root_state)
        
        if self.root is None:
            self.root = self.mcts_expansion()
            self.master_root = self.root
        else:
            self.root = self.root.getChild(self.match.joint_move, self.role_count)
            if self.root is None:
                self.root = self.mcts_expansion()
            else:
                self.root.parent = None
                self.root.parent_move = None
        
        self.mcts_runs = 1
        self.mcts_expansions = 1
        while True:
            if time.time() > finish_by:
                break
                
            if self.max_expansions > 0 and self.mcts_expansions > self.max_expansions:
                break

            #Reset state machine
            self.sm.update_bases(root_state)

            node, move = self.mcts_selection(self.root)
            new_node = self.mcts_expansion(node, move)
            self.mcts_playout()
            self.mcts_backpropagation(new_node)
            self.mcts_runs += 1
            self.mcts_expansions += 1
        
        return self.mcts_expansions

    #Choose the next move to play at the end of search.
    def choose(self):
        highestQ = -float("inf")
        best_action = -1
        for action in self.root.actions[self.role]:
            actionQ = self.root.actions[self.role][action].Q
            print "Action: ", self.get_move_name(self.role, action), " value: ", actionQ, " visits: ", self.root.actions[self.role][action].N
            if actionQ > highestQ:
                highestQ = actionQ
                best_action = action
        return best_action


    #----GGPLIB

    def __init__(self, selection_policy_type, playout_policy_type, estimator, max_expansions=100000, name=None, keep_estimators=False):
        super(CarlPlayer, self).__init__(name)
        self.selection_policy_type = selection_policy_type
        self.playout_policy_type = playout_policy_type
        self.estimator = estimator
        self.max_expansions = max_expansions
        self.keep_estimators = keep_estimators

    def reset(self, match):
        self.role = match.our_role_index
        self.role_count = len(match.sm.get_roles())
        self.sarsa_error = 0
        self.average_branching_factor = 0
        self.average_depth = 0
        self.sarsa_expansions = 0
        for role_index in range(self.role_count):
            self.sarsa_agents[role_index] = SarsaEstimator(clone(self.estimator), len(match.game_info.model.actions[role_index]))
            #self.sarsaAgents[role_index] = SarsaTabular()
        
        self.playout_policy = CarlUtils.RandomPolicy(self.role_count)

        self.match = match

    def on_meta_gaming(self, finish_time):
        self.sm = self.match.sm.dupe()

        self.current_move = self.sm.get_joint_move()
        self.current_state = self.sm.new_base_state()
        self.last_move = self.sm.get_joint_move()
        self.last_state = self.sm.new_base_state()

        self.sarsa_iterations = self.perform_sarsa(finish_time)

        print "Sarsa finished."
        print "Average branching factor: " + str(self.average_branching_factor)
        print "Average depth: " + str(self.average_depth)
        estimated_explored = math.log(self.sarsa_expansions,self.average_branching_factor)/self.average_depth
        print "Estimated state space explored: " + str(estimated_explored)

        if self.selection_policy_type == "sucb":
            self.selection_policy = CarlUtils.SarsaSelectionPolicy(self.role_count, self.sarsa_agents)
        elif self.selection_policy_type == "ucb":
            self.selection_policy = CarlUtils.UCTSelectionPolicy(self.role_count)
        else:
            print("Invalid selection policy provided, defaulting to UCB.")
            self.selection_policy = CarlUtils.UCTSelectionPolicy(self.role_count)
            
        if self.playout_policy_type == "sarsa":
            self.playout_policy = CarlUtils.SarsaPlayoutPolicy(self.role_count, self.sarsa_agents)
        elif self.playout_policy_type == "random":
            self.playout_policy = CarlUtils.RandomPolicy(self.role_count)
        else:
            print("Invalid playout policy provided, defaulting to random.")
            self.playout_policy = CarlUtils.RandomPolicy(self.role_count)


    def on_next_move(self, finish_time):
        start_time = time.time()
        self.sm.update_bases(self.match.get_current_state())
        runs = self.perform_mcts(finish_time)
        print "Managed ",  runs, "expansions."
        
        self.iteration_count_list.append(runs)
        self.time_list.append(time.time() - start_time)
        
        return self.choose()

    #Search tree memory dealloc
    def cleanup(self):
        print "****************************************************"
        print "CLEANING"
        print "****************************************************"

        CarlUtils.log_to_csv(self)

        if self.master_root is not None:
            tree_cleanup(self.master_root)
            self.master_root = None
            self.root = None
        if self.current_state is not None:
            interface.dealloc_basestate(self.current_state)
            self.current_state = None
        if self.last_state is not None:
            interface.dealloc_basestate(self.last_state)
            self.last_state = None
        if self.sm is not None:
            interface.dealloc_statemachine(self.sm)
            self.sm = None
