import random
import time
import math

from Sarsa import SarsaEstimator
from sklearn.linear_model import SGDRegressor

from ggplib.player.mcs import MoveStat
from ggplib.util import log
from ggplib.player.base import MatchPlayer

class CarlPlayer(MatchPlayer):
    sarsaAgents = {}
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

    #----GGPLIB

    def __init__(self, name=None):
        super(CarlPlayer, self).__init__(name)

    def reset(self, match):
        #Keep the estimator from previous game if the next game is the same
        if not self.match or not self.match.game_info.game == match.game_info.game:
            self.role = match.our_role_index
            role_count = len(match.sm.get_roles())
            for role_index in range(role_count):
                self.sarsaAgents[role_index] = SarsaEstimator(SGDRegressor(loss='huber'), len(match.game_info.model.actions[role_index]))
                #self.sarsaAgents[role_index] = SarsaTabular()

        self.match = match

    def on_meta_gaming(self, finish_time):
        self.sm = self.match.sm.dupe()

        self.current_move = self.sm.get_joint_move()
        self.current_state = self.sm.new_base_state()
        self.last_move = self.sm.get_joint_move()
        self.last_state = self.sm.new_base_state()

        self.role_count = len(self.sm.get_roles())

        self.perform_sarsa(finish_time)

    

    #----SARSA

    #TODO: multithread?
    #Playouts used by SARSA to learn policies before the game starts.
    def perform_sarsa(self, finish_time):
        #Playout related variables
        playout_count = 0

        #To ensure that we don't overshoot our training time
        time_offset = 0.01

        while time.time() + time_offset < finish_time:
            #Rewind the state machine
            self.sm.update_bases(self.match.sm.get_current_state())

            init = True
            while time.time() + time_offset < finish_time and not self.sm.is_terminal():

                #Choose moves for all players.
                for role_index in range(self.role_count):
                    choice = self.sarsaAgents[role_index].policy_training(self.current_state, self.sm.get_legal_state(role_index))
                    self.current_move.set(role_index, choice)

                if not init:
                    #update all agents with bootstrapping
                    for role_index in range(self.role_count):
                        self.sarsaAgents[role_index].observe(self.last_state, self.last_move.get(role_index), state_prime=self.current_state, action_prime=self.current_move.get(role_index))
                else:
                     init = False

                #Current move/action becomes last move/action
                if not self.sm.is_terminal():
                    self.last_state.assign(self.current_state)
                    for role_index in range(self.role_count):
                        self.last_move.set(role_index, self.current_move.get(role_index))


                # Update current_state with joint move
                self.sm.next_state(self.current_move, self.current_state)
                # update the state machine
                self.sm.update_bases(self.current_state)
            
            #update all agents with end reward
            for role_index in range(self.role_count):
                self.sarsaAgents[role_index].observe(self.last_state, self.last_move.get(role_index), reward=self.sm.get_goal_value(role_index))

            playout_count += 1
        return playout_count


#----MCTS

    def do_playout(self):
        # performs the simplest depth charge, returning our score
        current_move = self.sm.get_joint_move()
        current_state = self.sm.new_base_state()
        while True:
            if self.sm.is_terminal():
                break

            current_state = self.sm.get_current_state(current_state)

            # assign move for each player based on policy learned by SARSA
            for role_index in range(self.role_count):
                legal_state = self.sm.get_legal_state(role_index)
                action_choice = self.sarsaAgents[role_index].nondet_policy(current_state, legal_state)
                current_move.set(role_index, action_choice)

            # play move
            self.sm.next_state(current_move, current_state)
            self.sm.update_bases(current_state)

        # we are only intereted in our score
        return [self.sm.get_goal_value(ii) for ii in range(self.role_count)]

    def select_move(self, choices, visits, all_scores):
        # here we build up a list of possible candidates, and then return one of them randomly.
        # Most of the time there will only be one candidate.

        candidates = []

        # add any choices, where it hasn't played at least 3 times
        for c in choices:
            stat = all_scores[c]
            if stat.visits < 3:
                candidates.append(c)

        if not candidates:
            best_score = -1
            log_visits = math.log(visits)
            for c in choices:
                stat = all_scores[c]

                # we can be assured that having no candidates means stat.visits >= 3
                
                #TODO: use value function to select instead of UCB when UCB is below a threshold
                score = stat.get(self.match.our_role_index) + self.ucb_constant * math.sqrt(log_visits / stat.visits)
                if score < best_score:
                    continue

                if score > best_score:
                    best_score = score
                    candidates = []

                candidates.append(c)

        assert candidates
        return random.choice(candidates)

    def perform_mcs(self, finish_by):
        self.current_state.assign(self.match.get_current_state())
        self.sm.update_bases(self.current_state)

        self.root = {}

        ls = self.sm.get_legal_state(self.match.our_role_index)
        our_choices = [ls.get_legal(ii) for ii in range(ls.get_count())]

        # now create some stats with depth charges
        for choice in our_choices:
            move = self.sm.legal_to_move(self.match.our_role_index, choice)
            self.root[choice] = MoveStat(choice, move, self.role_count)

        root_visits = 1
        while True:
            if time.time() > finish_by:
                break

            if self.max_iterations > 0 and root_visits > self.max_iterations:
                break

            if len(our_choices) == 1:
                if root_visits > 100:
                    break

            # return to current state
            self.current_state.assign(self.match.get_current_state())
            self.sm.update_bases(self.current_state)

            assert not self.sm.is_terminal()

            # select and set our move
            choice = self.select_move(our_choices, root_visits, self.root)
            self.current_move.set(self.match.our_role_index, choice)

            # and a random move from other players
            for idx, r in enumerate(self.sm.get_roles()):
                if idx != self.match.our_role_index:
                    ls = self.sm.get_legal_state(idx)
                    choices = [ls.get_legal(ii) for ii in range(ls.get_count())]

                    # only need to set this once :)
                    self.current_move.set(idx, choices[random.randrange(0, ls.get_count())])

            # create a new state
            self.sm.next_state(self.current_move, self.current_state)

            # do a depth charge, and update scores
            scores = self.do_playout()
            self.root[choice].add(scores)

            # and update the number of visits
            root_visits += 1

        log.debug("Total visits: %s" % root_visits)

    def choose(self):
        assert self.root is not None
        best_score = -1
        best_selection = None

        # ok - now we dump everything for debug, and return the best score
        for stat in sorted(self.root.values(),
                           key=lambda x: x.get(self.match.our_role_index),
                           reverse=True):
            score_str = " / ".join(("%.2f" % stat.get(ii)) for ii in range(self.role_count))
            log.info("Move %s, visits %d, scored %s" % (stat.move, stat.visits, score_str))

            s = stat.get(self.match.our_role_index)
            if s > best_score:
                best_score = s
                best_selection = stat

        assert best_selection is not None
        log.debug("choice move = %s" % best_selection.move)
        return best_selection.choice

    # Searches the tree until finish_time (according to SARSA policy), then returns a move.
    def on_next_move(self, finish_time):
        self.sm.update_bases(self.match.get_current_state())
        
        print "Managed ",  self.perform_mcs(finish_time), "playouts."
        
        print "Printing action state values of current state..."
        for role_index in range(self.role_count):
            legal_state = self.sm.get_legal_state(role_index)
            print "****************************************************"
            if role_index == self.role:
                print "Us"
            else:
                print "Other"
            print "****************************************************"
            for act in range(legal_state.get_count()):
                state = self.sm.get_current_state()
                print "action ", self.match.game_info.model.actions[role_index][legal_state.get_legal(act)], " value is ", self.sarsaAgents[role_index].value(state, act)
            print ""

        #TODO: use search tree
        self.sm.update_bases(self.match.get_current_state())
        legal_state = self.sm.get_legal_state(self.role)
        return self.choose()
