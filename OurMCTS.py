import random
import time
import math

from Sarsa import SarsaEstimator
from sklearn.linear_model import SGDRegressor

from ggplib.player.mcs import MoveStat
from ggplib.util import log
from ggplib.player.base import MatchPlayer

#Here we hash joint moves to use them as index
def hash_joint_move(role_count, joint_move):
    joint_move_list = []
    for role_index in range(role_count):
        joint_move_list.append(joint_move.get(role_index))
    return hash(str(joint_move_list))

class Node():
    
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
        self.children = {}
            

class Action():

    def __init__(self, move, N = 0, Q = -1):
        self.move = move
        self.N = N
        self.Q = Q

class MCTSPlayer(MatchPlayer):
    role_count = 0
    role = 0
    sm = None
    playout_base_state = None

    ucb_constant = 1.414
    max_iterations = -1

    current_move = None
    current_state = None
    last_move = None
    last_state = None
    
    root = None

    mcts_runs = 1
    #----GGPLIB

    def __init__(self, name=None):
        super(MCTSPlayer, self).__init__(name)

    def reset(self, match):
        if not self.match or not self.match.game_info.game == match.game_info.game:
            self.role = match.our_role_index
            self.role_count = len(match.sm.get_roles())

        self.match = match
    def get_move_name(self, role, move):
        return self.match.game_info.model.actions[role][move]

    def print_joint_move(self,joint_move):
        for role_index in range(self.role_count):
            print "Player no.", role_index, " chose ", self.get_move_name(role_index, joint_move.get(role_index))

    def on_meta_gaming(self, finish_time):
        self.sm = self.match.sm.dupe()

        self.current_move = self.sm.get_joint_move()
        self.current_state = self.sm.new_base_state()
        self.last_move = self.sm.get_joint_move()
        self.last_state = self.sm.new_base_state()

        self.role_count = len(self.sm.get_roles())

#----MCTS

    def ucb(self, node, action):
        if(action.N == 0 or node.N == 0):
            return float("inf")
        
        return action.Q + self.ucb_constant * math.sqrt(math.log(node.N) / action.N) 

    #Here we select a move with the highest UCB value.
    def select_best_move(self, role, current_node):
        best_action = -1
        best_ucb = -float("inf")

        for action_index in current_node.actions[role]:
            temp = self.ucb(current_node, current_node.actions[role][action_index])
            #print role, " player move: ", self.get_move_name(role, action_index), " ucb: ", temp
            if temp > best_ucb:
                best_ucb = temp
                best_action = action_index

        #print role, " player move ", self.get_move_name(role, best_action), " with ucb ", best_ucb, " chosen." 
        return best_action
        
    #Returns the joint move where each player has its highest valued move based on UCB.
    def select_joint_move(self, current_node):
        #TODO: optimize to not create new joint move every time
        joint_move = self.sm.get_joint_move()
        for role_index in range(self.role_count):
            joint_move.set(role_index, self.select_best_move(role_index, current_node))

        return joint_move

    #TODO: create an expansion policy?
    def do_selection(self, root):
        #print "----------------------------------------------------------------"
        #print "selection"
        #print "----------------------------------------------------------------"
        #Select, based on UCB, what path to traverse.
        last_node = root
        current_node = root
        current_state = self.sm.get_current_state()
        next_move = None

        while (current_node is not None) and (not self.sm.is_terminal()):
            next_move = self.select_joint_move(current_node)
            #self.print_joint_move(next_move)
            last_node = current_node
            
            current_node = current_node.getChild(next_move, self.role_count)

            self.sm.next_state(next_move, current_state)
            self.sm.update_bases(current_state)

            #print "selected, move ", self.mcts_runs, ": "
            #self.print_joint_move(next_move)
            #if self.mcts_runs < 50:
                #print "current node move hash: ", hash_joint_move(self.role_count, next_move)
                #self.mcts_runs += 1
            #    lol = "lol"
            #else:
            #    exit()
            
                
        
        return last_node, next_move, current_state

    def do_expansion(self, selected_node, next_move, selected_node_state):
        #We add one edge to the current node
        new_node = Node(parent=selected_node, parent_move=next_move)
        selected_node.children[hash_joint_move(self.role_count, next_move)] = new_node
        #print selected_node.parent == new_node
        if not self.sm.is_terminal():
            for role in range(self.role_count):
                new_node.actions.append({})
                legal_state = self.sm.get_legal_state(role)
                for action_index in range(legal_state.get_count()):
                    action = legal_state.get_legal(action_index)
                    new_node.actions[role][action] = Action(action)

        
        #if mcts_runs == 1:
        #    print "Test root: "
        #    for role in range(self.role_count):
        #        print "Role nr.", role, " actions: ", self.root.actions[role]
        return new_node
        
    
    def do_playout(self):
        # performs the simplest depth charge, returning our score
        current_move = self.sm.get_joint_move()
        current_state = self.sm.new_base_state()
        while True:
            if self.sm.is_terminal():
                break

            current_state = self.sm.get_current_state(current_state)

            # randomly assign move for each player
            for role_index in range(self.role_count):
                ls = self.sm.get_legal_state(role_index)
                choice = ls.get_legal(random.randrange(0, ls.get_count()))
                current_move.set(role_index, choice)

            # play move
            self.sm.next_state(current_move, current_state)
            self.sm.update_bases(current_state)

    def do_backpropagation(self, tree_node):
        #print "----------------------------------------------------------------"
        #print "backpropagation"
        #print "----------------------------------------------------------------"
        while not (tree_node.parent_move == None or tree_node.parent == None):
            #self.print_joint_move(tree_node.parent_move)
            for role in range(self.role_count):
                index = tree_node.parent_move.get(role)
                action = tree_node.parent.actions[role][index]
                action.Q = (action.Q*action.N + self.sm.get_goal_value(role))/(float(action.N+1.0))
                #action.Q += self.sm.get_goal_value(role)
                action.N += 1
            tree_node.N +=1
            tree_node = tree_node.parent
        self.root.N += 1

    def create_root(self):
        self.root = Node(None, None)
        if not self.sm.is_terminal():
            for role in range(self.role_count):
                self.root.actions.append({}) 
                legal_state = self.sm.get_legal_state(role)
                for action_index in range(legal_state.get_count()):
                    action = legal_state.get_legal(action_index)
                    #print "Root creation. Player no.", role, " move: ", self.get_move_name(role, action)
                    self.root.actions[role][action] = Action(action)
        #print "Test root: "
        #for role in range(self.role_count):
        #    print "Role nr.", role, " actions: ", self.root.actions[role]

    def perform_mcts(self, finish_by):
        self.mcts_runs = 0
        root_state = self.sm.get_current_state()
        self.sm.update_bases(root_state)
        
        if self.root is None:
            self.create_root()
        else:
            #self.print_joint_move(self.match.joint_move)
            self.root = self.root.getChild(self.match.joint_move, self.role_count)
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

            #reset state machine
            self.sm.update_bases(root_state)

            #print("Starting selection")
            node, move, state = self.do_selection(self.root)
            #print("Selection finished")
            new_node = self.do_expansion(node, move, state)
            #print("Expansion finished")
            self.do_playout()
            #print("playout finished")
            self.do_backpropagation(new_node)
            #print("backpropagation finished")
            self.mcts_runs += 1
        return self.mcts_runs

    def choose(self):
        highestQ = -float("inf")
        bestAction = -1
        for action in self.root.actions[self.role]:
            actionQ = self.root.actions[self.role][action].Q
            print "Action: ", self.get_move_name(self.role, action), " value: ", actionQ, " visits: ", self.root.actions[self.role][action].N
            if actionQ > highestQ:
                highestQ = actionQ
                bestAction = action
        return self.root.actions[self.role][bestAction].move

    # Searches the tree until finish_time (according to SARSA policy), then returns a move.
    def on_next_move(self, finish_time):
        self.sm.update_bases(self.match.get_current_state())
        runs = self.perform_mcts(finish_time)
        print "Managed ",  runs, "playouts."
        
        return self.choose()
