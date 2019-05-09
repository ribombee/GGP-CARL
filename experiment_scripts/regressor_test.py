#Our selected regressors will be tested against a baseline MCTS agent.

#Variables:
#--regressors, which will all be tested against baseline mcts
#--start clock, for which all experiments will be repeated for different values in a range.
import sys, subprocess
from BatchGameRunner import BatchGameRunner
from BatchGameRunner import Player_info
game = "connectFour"
start_clock_min = 300
start_clock_max = 1800
start_clock_step = 300
play_clock = "30" #this amount of time should be excessive

p1_ip = sys.argv[1]
p1_type = "sarsa"
p1_port = 1337
p1_regressors = ["sgd", "mlp", "paggro"]


p2_ip = sys.argv[2]
p2_type = "mcts"
p2_port = 1337 if not p2_ip == p1_ip else 1338 
p2_regressor = "sgd"

expansions = 1000

runs_per_permuation = 30

for p1_regressor in p1_regressors:
    for start_clock in range(start_clock_min, start_clock_max, start_clock_step):
        game_runner = BatchGameRunner()
        game_runner.setup(game, str(start_clock), play_clock, Player_info(p1_ip, p1_type, p1_port, p1_regressor), Player_info(p2_ip, p2_type, p2_port, p2_regressor), expansions)
        game_runner.run_tests(runs_per_permuation)
        #base_call_string = "python BatchGameRunner.py " + runs_per_permuation + " " + game + " " + start_clock + " " + play_clock + " " +\
        #                    p1_ip + " " + p1_type + " " + p1_regressor + " " +\
        #                    p2_ip + " " + p2_type + " " + p2_regressor + " " + expansions
        #process = subprocess.Popen(base_call_string + str(expansions), shell=True)
        #process.wait()