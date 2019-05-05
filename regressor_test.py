#Our selected regressors will be tested against a baseline MCTS agent.

#Variables:
#--regressors, which will all be tested against baseline mcts
#--start clock, for which all experiments will be repeated for different values in a range.

#


import sys, subprocess
game = "connectFour"
start_clock_min = 300
start_clock_max = 1800
play_clock = "99999"

p1_ip = sys.argv[1]
p1_type = "sarsa"
p1_regressors = ["sgd", "mlp", "paggro"]

p2_ip = sys.argv[2]
    for start_clock in range(start_clock_min, start_clock_max, 300):

        base_call_string = "python BatchGameRunner.py " + runs_per_permuation + " " + game + " " + start_clock + " " + play_clock + " " +\
                            p1_ip + " " + p1_type + " " + p1_regressor + " " +\
                            p2_ip + " " + p2_type + " " + p2_regressor + " " + expansions
        process = subprocess.Popen(base_call_string + str(expansions), shell=True)
        process.wait()    for start_clock in range(start_clock_min, start_clock_max, 300):

        base_call_string = "python BatchGameRunner.py " + runs_per_permuation + " " + game + " " + start_clock + " " + play_clock + " " +\
                            p1_ip + " " + p1_type + " " + p1_regressor + " " +\
                            p2_ip + " " + p2_type + " " + p2_regressor + " " + expansions
        process = subprocess.Popen(base_call_string + str(expansions), shell=True)
        process.wait()    for start_clock in range(start_clock_min, start_clock_max, 300):

        base_call_string = "python BatchGameRunner.py " + runs_per_permuation + " " + game + " " + start_clock + " " + play_clock + " " +\
                            p1_ip + " " + p1_type + " " + p1_regressor + " " +\
                            p2_ip + " " + p2_type + " " + p2_regressor + " " + expansions
        process = subprocess.Popen(base_call_string + str(expansions), shell=True)
        process.wait()p2_type = "mcts"
p2_regressor = "sgd"

expansions = -1

runs_per_permuation = 30

for p1_regressor in p1_regressors:
    for start_clock in range(start_clock_min, start_clock_max, 300):

        base_call_string = "python BatchGameRunner.py " + runs_per_permuation + " " + game + " " + start_clock + " " + play_clock + " " +\
                            p1_ip + " " + p1_type + " " + p1_regressor + " " +\
                            p2_ip + " " + p2_type + " " + p2_regressor + " " + expansions
        process = subprocess.Popen(base_call_string + str(expansions), shell=True)
        process.wait()