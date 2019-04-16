import random
from Sarsa import SarsaEstimator
import time
from sklearn.linear_model import SGDRegressor

from ggplib.player.base import MatchPlayer

class carlPlayer(MatchPlayer):

    #----GGPLIB

    def __init__(self, name=None):
        super(carlPlayer, self).__init__(name)
        self.carlAgents = {}
        self.role = 0

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
        self.do_playouts_sarsa(finish_time)

    # Returns a move according to SARSA policy
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

    #----SARSA

    #TODO: multithread?
    #Playouts used by SARSA to learn policies before the game starts.
    def do_playouts_sarsa(self, finish_time):
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


#----MCTS

    def do_playouts(self):
        # performs the simplest depth charge, returning our score
        self.sm.update_bases(self.playout_state)

        while True:
            if self.sm.is_terminal():
                break

            # randomly assign move for each player
            #for idx, r in enumerate(self.sm.get_roles()):
            #    ls = self.sm.get_legal_state(idx)
            #    choice = ls.get_legal(random.randrange(0, ls.get_count()))
            #    self.playout_joint_move.set(idx, choice)

            # assign move for each player based on policy learned by SARSA
            for idx, r, in enumerate(self.sm.get_roles()):
                ls = self.sm.get_legal_state(idx)
                action_choice = self.sarsaAgents[role_index].policy(current_state, temp_sm.get_legal_state(role_index))
                self.playout_joint_move.set(idx, action_choice)

            # play move
            self.sm.next_state(self.playout_joint_move, self.playout_state)
            self.sm.update_bases(self.playout_state)

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
        self.playout_state.assign(self.match.get_current_state())
        self.sm.update_bases(self.playout_state)

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
            self.playout_state.assign(self.match.get_current_state())
            self.sm.update_bases(self.playout_state)

            assert not self.sm.is_terminal()

            # select and set our move
            choice = self.select_move(our_choices, root_visits, self.root)
            self.joint_move.set(self.match.our_role_index, choice)

            # and a random move from other players
            for idx, r in enumerate(self.sm.get_roles()):
                if idx != self.match.our_role_index:
                    ls = self.sm.get_legal_state(idx)
                    choices = [ls.get_legal(ii) for ii in range(ls.get_count())]

                    # only need to set this once :)
                    self.joint_move.set(idx, choices[random.randrange(0, ls.get_count())])

            # create a new state
            self.sm.next_state(self.joint_move, self.playout_state)

            # do a depth charge, and update scores
            scores = self.do_playout()
            self.root[choice].add(scores)

            # and update the number of visits
            root_visits += 1

        log.debug("Total visits: %s" % root_visits)
