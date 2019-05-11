#Our selected regressors will be tested against a baseline MCTS agent.

#Variables:
#--regressors, which will all be tested against baseline mcts
#--start clock, for which all experiments will be repeated for different values in a range.
import sys, subprocess
from BatchGameRunner import BatchGameRunner
from BatchGameRunner import Player_info

class Regressor_test:
        def __init__(self, p1_ip, p2_ip = None, server_results_folder="results"):
                self.game = "connectFour"
                self.start_clock_test_list = [300,600,1200]
                self.play_clock = "120" #this amount of time should be excessive

                self.p1_ip = p1_ip
                self.p1_type = "sarsa"
                self.p1_port = 1337
                self.p1_regressors = ["sgd", "mlp", "paggro"]


                self.p2_ip = p1_ip if p2_ip is None else p2_ip
                self.p2_type = "mcts"
                self.p2_port = 1337 if not p2_ip == p1_ip else 1338 
                self.p2_regressor = "sgd"

                self.expansions = 1000
                
                self.runs_per_permuation = 1
                self.games_per_model = 100
                self.results_folder = server_results_folder

        def start_test(self):
                for p1_regressor in self.p1_regressors:
                        for start_clock in self.start_clock_test_list:
                                start_clock_list = [start_clock]+[3]*(self.games_per_model-1)
                                game_runner = BatchGameRunner()
                                game_runner.setup(self.game, self.play_clock, Player_info(self.p1_ip, self.p1_type, self.p1_port, p1_regressor), Player_info(self.p2_ip, self.p2_type, self.p2_port, self.p2_regressor), self.expansions, server_folder_id=self.results_folder)
                                game_runner.run_tests_from_list(start_clock_list)

if __name__ == "__main__":
    test = Regressor_test(sys.argv[0])
    test.start_test()