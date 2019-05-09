import sys, subprocess
from BatchGameRunner import BatchGameRunner
from BatchGameRunner import Player_info

game = "connectFour"
start_clock = 600
play_clock = "99999"

p1_ip = sys.argv[1]
p1_type = "sarsa"
p1_port = 1337
p1_regressor = "sgd"


p2_ip = sys.argv[1] if len(sys.argv) == 2 else sys.argv[2]
p2_type = "mcts"
p2_port = 1337 if not p2_ip == p1_ip else 1338 
p2_regressor = "sgd"

expansions_start = 500
expansions_fin = 5000
expansions_step = 1000

runs_per_step = 20

for expansions in range(expansions_start, expansions_fin, expansions_step):
    game_runner = BatchGameRunner()
    game_runner.setup(game, str(start_clock), play_clock, Player_info(p1_ip, p1_type, p1_port, p1_regressor), Player_info(p2_ip, p2_type, p2_port, p2_regressor), expansions)
    game_runner.run_tests(runs_per_step)