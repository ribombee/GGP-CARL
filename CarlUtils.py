import math, random
from ggplib import interface


#----Helper functions

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

#Helper function to calculate the ucb value of an action
def ucb(node, q_val, visits):
    UCB_CONST = 40
    if(visits == 0 or node.N == 0):
        return float("inf")
    
    return q_val + UCB_CONST * math.sqrt(math.log(node.N) / visits)


#----Search classes

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



#----Policies

class Policy:
    def __init__(self, role_count):
        self.role_count = role_count

    #choose a move from the policy
    def choose(self, current_move, sm, current_state = None, current_node = None):
        pass

class SarsaSelectionPolicy(Policy):

    def __init__(self, role_count, sarsa_agents, sucb_threshold):
        Policy.__init__(self, role_count)
        self.sarsa_agents = sarsa_agents
        self.sucb_threshold = sucb_threshold

    def find_action(self, current_node, current_state, sm, role_index):
        best_action = -1
        best_sucb = -float("inf")

        for action_index in current_node.actions[role_index]:
            action = current_node.actions[role_index][action_index]
            current_sucb = 0 
            if action.N < self.sucb_threshold:
                #What is the range of this? 0-100? yes?
                #TODO: find out how to use sarsa values in ucb (they are very low)
                #val = self.sarsa_agents[role_index].value(current_state, action_index)
                #current_sucb = ucb(current_node, val, action.N)


                best_action = self.sarsa_agents[role_index].nondet_policy(current_state, sm.get_legal_state(role_index))
                break
            else:
                current_sucb = ucb(current_node, action.Q, action.N)
                if current_sucb > best_sucb:
                    best_sucb = current_sucb
                    best_action = action_index

        return best_action

    def choose(self, current_move, sm, current_state = None, current_node = None):
        for role_index in range(self.role_count):
            current_move.set(role_index, self.find_action(current_node, current_state, sm, role_index))


class SarsaPlayoutPolicy(Policy):
    '''
        #Choose moves for all players.
            for role_index in range(self.role_count):
                choice = self.sarsaAgents[role_index].nondet_policy(current_state, self.sm.get_legal_state(role_index))
                current_move.set(role_index, choice)
    '''

    def __init__(self, role_count, sarsa_agents):
        Policy.__init__(self, role_count)
        self.sarsa_agents = sarsa_agents

    def choose(self, current_move, sm, current_state = None, current_node = None):
        for role_index in range(self.role_count):
            choice = self.sarsa_agents[role_index].nondet_policy(current_state, sm.get_legal_state(role_index))
            current_move.set(role_index, choice)

class UCTSelectionPolicy(Policy):
    def __init__(self, role_count):
        Policy.__init__(self, role_count)

    def find_action(self, current_node, sm, role_index):
        best_action = -1
        best_sucb = -float("inf")

        for action_index in current_node.actions[role_index]:
            action = current_node.actions[role_index][action_index]
            current_sucb = ucb(current_node, action.Q, action.N)
            if current_sucb > best_sucb:
                best_sucb = current_sucb
                best_action = action_index

        return best_action

    def choose(self, current_move, sm, current_state = None, current_node = None):
        for role_index in range(self.role_count):
            current_move.set(role_index, self.find_action(current_node, sm, role_index))


class RandomPolicy(Policy):
    def __init__(self, role_count):
        Policy.__init__(self, role_count)
        
    def choose(self, current_move, sm, current_state = None, current_node = None):
        #Randomly assign move for each player
        for role_index in range(self.role_count):
            ls = sm.get_legal_state(role_index)
            choice = ls.get_legal(random.randrange(0, ls.get_count()))
            current_move.set(role_index, choice)
        