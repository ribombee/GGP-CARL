import sys, subprocess
game = "connectFour"
start_clock = sys.argv[1]
play_clock = "99999"

p1_ip = sys.argv[2]
p1_type = "sarsa"
p1_regressor = "sgd"

p2_ip = sys.argv[3]
p2_type = "mcts"
p2_regressor = "sgd"

runs_per_expansion_val = "3"
expansions_start = 1000
expansions_fin = 100000

base_call_string = "python BatchGameRunner.py " + runs_per_expansion_val + " " + game + " " + start_clock + " " + play_clock + " " +\
                    p1_ip + " " + p1_type + " " + p1_regressor + " " +\
                    p2_ip + " " + p2_type + " " + p2_regressor + " "
for expansions in range(expansions_start, expansions_fin, 5000):
    process = subprocess.Popen(base_call_string + str(expansions), shell=True)
    process.wait()