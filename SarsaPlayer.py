import random
from Sarsa import Sarsa
import time
from sklearn.linear_model import SGDRegressor

from ggplib.player.base import MatchPlayer

class SarsaPlayer(MatchPlayer):
    def __init__(self, name=None):
        super(SarsaPlayer, self).__init__(name)
        self.sarsa = None

    def do_playthroughs(self, finish_time):
        game_role_index = self.match.our_role_index

        temp_sm = self.match.sm.dupe()
        role_count = len(temp_sm.get_roles())

        joint_move = temp_sm.get_joint_move()
        current_state = temp_sm.new_base_state()
        last_state = None

        #To ensure that we don't overshoot our training time
        time_offset = 0.01

        player_choice = 0

        playthrough_count = 0
        while time.time() + time_offset < finish_time:
            #Start the temp state machine at the beginning
            #temp_sm.update_bases(self.match.sm.get_current_state())
            temp_sm.reset()
            while time.time() + time_offset < finish_time and not temp_sm.is_terminal():
                #Choose moves for all potential players.
                #Everyone other than our player plays randomly
                for role_index in range(role_count):
                    ls = temp_sm.get_legal_state(role_index)
                    if role_index == game_role_index:
                        player_choice = self.sarsa.policy_training(current_state, temp_sm.get_legal_state(game_role_index))
                        joint_move.set(role_index, player_choice)
                    else:  
                        choice = ls.get_legal(random.randrange(0, ls.get_count()))
                        joint_move.set(role_index, choice)

                # Update current_state with joint move
                temp_sm.next_state(joint_move, current_state)

                # update the state machine
                temp_sm.update_bases(current_state)

                if last_state is not None and not temp_sm.is_terminal():
                    #Update estimator with bootstrapping
                    self.sarsa.observe(last_state, player_choice, state_prime=current_state)
                else:
                    last_state = temp_sm.new_base_state()

                #Set last_state to current_state
                last_state.assign(current_state)
            
            #With the end of a game, we update estimator using the goal reward 
            self.sarsa.observe(current_state, player_choice, reward=temp_sm.get_goal_value(game_role_index))
            playthrough_count += 1
        return playthrough_count

    def on_next_move(self, finish_time):
        if self.sarsa is None:
            self.sarsa = Sarsa(SGDRegressor())

        game_role_index = self.match.our_role_index
        sm = self.match.sm
        
        print "Managed ",  self.do_playthroughs(finish_time), "playthroughs"

        ls = sm.get_legal_state(game_role_index)
        for ind in range(ls.get_count()):
            print "action ", ls.get_legal(ind), " value is ", self.sarsa.value(sm.get_current_state(), ind)

        return self.sarsa.policy(sm.get_current_state(), ls)
