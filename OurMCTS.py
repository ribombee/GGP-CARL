import random
import time
import math
import datetime
import csv

from Sarsa import SarsaEstimator
from sklearn.linear_model import SGDRegressor

from ggplib.player.mcs import MoveStat
from ggplib.util import log
from ggplib.player.base import MatchPlayer
from ggplib import interface

#TODO: Verify that memory cleanup makes sense
#TODO: Reset state machine correctly between new games

#Helper function for hashing joint moves to use as index for dictionary
def hash_joint_move(role_count, joint_move):
    joint_move_list = []
    for role_index in range(role_count):
        joint_move_list.append(joint_move.get(role_index))
    return hash(str(joint_move_list))

def tree_cleanup(node):
    if node is not None:
        if node.parent_move is not None:
            interface.dealloc_jointmove(node.parent_move)
        node.parent_move = None
        node.actions = None
        for child_index in node.children:
            tree_cleanup(node.children[child_index])
        node.children = None

#Node for MCTS search tree
class Node():
    #Get child node from joint move
    def getChild(self, joint_move, role_count):
        hash_key = hash_joint_move(role_count, joint_move)
        if hash_key in self.children:
            return self.children[hash_key]
        else:
            return None

    def __init__(self, parent = None, parent_move = None):
        #The parent of this node  
        #If the parent is null, this node is the root of the tree.
        self.parent = parent

        #The move taken from parent to reach this node
        self.parent_move = parent_move

        #The number of time this node has been visited.
        self.N = 0

        #List of possible actions per player.
        self.actions = [{}]

        #TODO: reconsider having children as dict?
        #Children nodes of the current node, indexed by hashed joint move.
        self.children = {}
            
#Action that a role can take from a node.
#There are multiple actions per role per node
class Action():

    def __init__(self, move, N = 0, Q = -1):
        #Integer that represents the action's move according
        self.move = move
        
        #Number of time this role has taken this action
        self.N = N
        
        #Q value for this action for this role
        self.Q = Q

class MCTSPlayer(MatchPlayer):
    role_count = 0
    role = 0
    sm = None

    #We want to put more emphasis on exploring barely visited nodes.
    ucb_constant = 1.414
    #Maximum iterations per game, -1 means no limit
    max_iterations = 10000

    csv_log_file = "PlayerLog.csv"
    iteration_count_list = []
    time_list = []
    sarsa_iterations = 0

    #TODO: use this instead of allocating new states/moves in phases
    current_move = None
    current_state = None
    last_move = None
    last_state = None
    
    root = None
    master_root = None

    mcts_runs = 1

    #----Helper functions

    def get_move_name(self, role, move):
        return self.match.game_info.model.actions[role][move]

    def print_joint_move(self,joint_move):
        for role_index in range(self.role_count):
            print "Player no.", role_index, " chose ", self.get_move_name(role_index, joint_move.get(role_index))

    #Helper function to calculate the ucb value of an action
    def ucb(self, node, action):
        if(action.N == 0 or node.N == 0):
            return float("inf")
        
        return action.Q/100.0 + self.ucb_constant * math.sqrt(math.log(node.N) / action.N) 

    #----MCTS

    #Here we select a move with the highest UCB value.
    def select_best_move(self, role, current_node):
        best_action = -1
        best_ucb = -float("inf")

        for action_index in current_node.actions[role]:
            temp = self.ucb(current_node, current_node.actions[role][action_index])
            if temp > best_ucb:
                best_ucb = temp
                best_action = action_index

        return best_action
        
    #Returns the joint move where each player has its highest valued move based on UCB.
    def select_joint_move(self, current_node):
        #TODO: optimize to not create new joint move every time
        #Maybe it is neccessary?
        joint_move = self.sm.get_joint_move()
        for role_index in range(self.role_count):
            joint_move.set(role_index, self.select_best_move(role_index, current_node))

        return joint_move

    #Selection phase of the mcts. 
    #We move down the tree, selecting actions with the highest ucb values 
    #until we reach terminal state or an unexpanded node.
    def do_selection(self, root):
        last_node = root
        current_node = root
        current_state = self.sm.get_current_state()
        next_move = None

        while (current_node is not None) and (not self.sm.is_terminal()):
            next_move = self.select_joint_move(current_node)
            last_node = current_node
            
            current_node = current_node.getChild(next_move, self.role_count)

            #Update state/state machine
            self.sm.next_state(next_move, current_state)
            self.sm.update_bases(current_state)       
        
        return last_node, next_move, current_state

    #TODO: make create_root and do_expansion be the same function?
    #Create the first node of our MCTS search tree.
    def create_root(self):
        self.root = Node(None, None)
        if not self.sm.is_terminal():
            for role in range(self.role_count):
                self.root.actions.append({}) 
                legal_state = self.sm.get_legal_state(role)
                for action_index in range(legal_state.get_count()):
                    action = legal_state.get_legal(action_index)
                    self.root.actions[role][action] = Action(action)


    #Expansion phase of MCTS. We expand a selected node with a selected action.
    def do_expansion(self, selected_node, next_move, selected_node_state):
        #We add one edge to the current node
        new_node = Node(parent=selected_node, parent_move=next_move)
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
    def do_playout(self):
        current_move = self.sm.get_joint_move()
        current_state = self.sm.new_base_state()
        while True:
            if self.sm.is_terminal():
                break

            current_state = self.sm.get_current_state(current_state)

            #Rdomly assign move for each player
            for role_index in range(self.role_count):
                ls = self.sm.get_legal_state(role_index)
                choice = ls.get_legal(random.randrange(0, ls.get_count()))
                current_move.set(role_index, choice)

            #Update state/state machine
            self.sm.next_state(current_move, current_state)
            self.sm.update_bases(current_state)

    #Use the value of the terminal state from playout to update Q values for each visited node.
    def do_backpropagation(self, tree_node):
        while not (tree_node.parent_move == None or tree_node.parent == None):
            for role in range(self.role_count):
                index = tree_node.parent_move.get(role)
                action = tree_node.parent.actions[role][index]
                #Update the running average
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
            self.create_root()
            #self.master_root = self.root
        else:
            old_root = self.root
            self.root = self.root.getChild(self.match.joint_move, self.role_count)
            del old_root.children[hash_joint_move(self.role_count, self.match.joint_move)]
            tree_cleanup(old_root)

            if self.root is None:
                self.create_root()
            else:
                self.root.parent = None
                self.root.parent_move = None
        
        self.mcts_runs = 1
        while True:
            if time.time() > finish_by:
                break
                
            if self.max_iterations > 0 and self.mcts_runs > self.max_iterations:
                break

            #Reset state machine
            self.sm.update_bases(root_state)

            node, move, state = self.do_selection(self.root)
            new_node = self.do_expansion(node, move, state)
            self.do_playout()
            self.do_backpropagation(new_node)
            self.mcts_runs += 1
        return self.mcts_runs

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

    def __init__(self, name=None):
        super(MCTSPlayer, self).__init__(name)

    def reset(self, match):
        if not self.match or not self.match.game_info.game == match.game_info.game:
            self.role = match.our_role_index
            self.role_count = len(match.sm.get_roles())

        self.match = match

    def on_meta_gaming(self, finish_time):
        self.sm = self.match.sm.dupe()

        self.current_move = self.sm.get_joint_move()
        self.current_state = self.sm.new_base_state()
        self.last_move = self.sm.get_joint_move()
        self.last_state = self.sm.new_base_state()

        self.role_count = len(self.sm.get_roles())

    #Searches the tree until finish_time (according to SARSA policy), then returns a move.
    def on_next_move(self, finish_time):
        self.sm.update_bases(self.match.get_current_state())
        runs = self.perform_mcts(finish_time)
        print "Managed ",  runs, "playouts."
        
        self.iteration_count_list.append(runs)
        self.time_list.append(time.time() - start_time)
        
        return self.choose()

    #Search tree memory dealloc
    def cleanup(self):
        print "****************************************************"
        print "CLEANING"
        print "****************************************************"
        self.log_to_csv()
        if self.root is not None:
            tree_cleanup(self.root)
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

    def log_to_csv(self):
    #This logs to the log file a single line. This line should be all the relevant data for one game in the following format.
    # <List with number of expansions per state> <List with time taken per state>

        with open(self.csv_log_file, 'w+') as log_file:
            log_file.write(str(self.sarsa_iterations))
            log_file.write(',')
            for i, item in enumerate(self.iteration_count_list):
                if i != 0:
                    log_file.write(';')
                log_file.write(str(item))
            log_file.write(',')
            for i, item in enumerate(self.time_list):
                if i != 0:
                    log_file.write(';')
                log_file.write(str(item))
