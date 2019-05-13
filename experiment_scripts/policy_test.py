#Tests that we will run to analyse the quality of our suggested policies.

import sys, subprocess
from BatchGameRunner import BatchGameRunner
from BatchGameRunner import Player_info

class Policy_test:
        def __init__(self, p1_ip, p2_ip = None, server_results_folder="results"):
                self.games = ["connectFour", "breakthrough", "amazons", "checkers"]

                self.start_clock = 600
                self.play_clock = "400" #Should not be exceeded

                self.p1_ip = p1_ip
                
                self.p1_types = ["carl_playout", "mcts"]
                self.p1_port = 1337
                self.p1_regressor = "mlp"

                self.p2_ip = p1_ip if p2_ip is None else p2_ip
                self.p2_types = ["carl_playout", "mcts"]
                self.p2_port = 1337 if not p2_ip == p1_ip else 1338 
                self.p2_regressor = "mlp"

                self.expansions =  [1000, 50000]
                
                self.games_per_policy = 50 #try about 100 or so first, then see if the stderror calls for more.
                self.results_folder = server_results_folder

        def start_test(self):
            for game in self.games:
                for p1_type in self.p1_types:
                    if p1_type in self.p2_types:
                        self.p2_types.remove(p1_type)
                    for p2_type in self.p2_types:
                        relevant_test = p1_type != p2_type  #and not ( p1_type in ['mcts', 'random'] and p2_type in ['mcts', 'random'])
                        if relevant_test: #We do not need to test the winrate of one policy against itself.
                            for max_expansions in self.expansions:
                                start_clock_list = [self.start_clock]+[3]*(self.games_per_policy-1)
                                game_runner = BatchGameRunner()
                                game_runner.setup(game, self.play_clock, Player_info(self.p1_ip, p1_type, self.p1_port, self.p1_regressor), Player_info(self.p2_ip, p2_type, self.p2_port, self.p2_regressor), max_expansions, server_folder_id=self.results_folder)
                                game_runner.run_tests_from_list(start_clock_list)

if __name__ == "__main__":
    test = Policy_test(sys.argv[0])
    test.start_test()