    #0       1                   2       3                                  4                    5                        6                  7
    #Winner, P1 Error over time, P1 EEF, P1 List of errors per observation, P2 Sarsa iterations, P2 iterations per state, P2 time per state, moves

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import OrderedDict

VERBOSE = False

#Helper function to get list average
def list_avg(lst): 
    return sum(lst) / len(lst)

class LogAnalyzer:
    def __init__(self, file_name):
        self.file_name = file_name
    
        self.log_df = None

        self.p1_name = "p1"
        self.p1_wins = 0
        self.p1_regr = "NO REGR"

        self.p2_name = "p2"
        self.p2_wins = 0
        self.p1_regr = "NO REGR"

        self.EOT_avg = -1
        self.EEF_Avg = -1

        self.sarsa_iters_avg = -1
        self.state_iters_avg = -1
        self.state_times_avg = -1

        self.move_avg = -1

        self.mcts_win_rate = -1.0

        self.error_avgs = []
        self.error_count = []

    #Read metadata from header
    def read_metadata(self):
        with open(self.file_name, 'r') as log_file:
            self.timestamp = log_file.next().split(':', 1)[1].strip()
            self.game_name = log_file.next().split(':')[1].strip()
            p1_info = log_file.next().split(': ')[1].strip().split(' ')
            p2_info = log_file.next().split(': ')[1].strip().split(' ')
            self.p1_name = p1_info[0]
            self.p1_regr = p1_info[1]
            self.p2_name = p2_info[0]
            self.p2_regr = p2_info[1]
            self.runs = log_file.next().split(':')[1].strip()
            self.expansions = int(log_file.next().split(':')[1].strip())
            self.runtime = log_file.next().split(':')[1].strip()
            self.start_clock = int(log_file.next().split(':')[1].strip())
            self.play_clock = log_file.next().split(':')[1].strip()

    #Read csv file
    def read_data(self):
        self.log_df = pd.read_csv(self.file_name, skiprows=10)

    #Calculate averages
    def read_game_values(self):
        EOT = []
        EEF = []
        sarsa_iters = []
        state_iters = []
        state_times = []
        moves = []

        for index, row in self.log_df.iterrows():
            EOT.append(float(row[1]))
            EEF.append(float(row[2]))
            sarsa_iters.append(float(row[4]))
            state_iters.append(list_avg([float(x) for x in row[5].split(';')]))
            state_times.append(list_avg([float(x) for x in row[6].split(';')]))
            moves.append(float(row[7]))
        
        #Overall averages
        self.EOT_avg = list_avg(EOT)
        self.EEF_avg = list_avg(EEF)

        self.sarsa_iters_avg = list_avg(sarsa_iters)
        self.state_iters_avg = list_avg(state_iters)
        self.state_times_avg = list_avg(state_times)

        self.move_avg = list_avg(moves)

    #Read player win count
    def read_wins(self):
        for index, row in self.log_df.iterrows():
            if row[0] == "Player 1":
                self.p1_wins += 1
            else:
                self.p2_wins += 1

        self.mcts_win_rate = (self.p2_wins/float(self.p1_wins+self.p2_wins))*100.0

    def print_analysis(self):
        print "File name: " + self.file_name
        print "Expansions: " + str(self.expansions)
        print "Game: " + self.game_name
        print "Player 1: " + self.p1_name + "_" + self.p1_regr + "_"
        print "Player 2: " + self.p2_name + "_" + self.p2_regr + "_"
        print "Average moves made: " + str(round(self.move_avg,2))
        print "--------------------------------"
        print "Sarsa average EOT: " + str(round(self.EOT_avg,3))
        print "Sarsa average EEF: " + str(round(self.EEF_avg,3))
        print ""
        print "MCTS average SARSA iterations: " + str(round(self.sarsa_iters_avg,2))
        print "MCTS average state iterations: " + str(round(self.state_iters_avg,2))
        print "MCTS average time per state: " + str(round(self.state_times_avg,3))
        print ""
        print "                                 MCTS win rate: " + str(round(self.mcts_win_rate,2)) + "%"
        print "================================================================"

    def run_analysis(self):
        self.read_metadata()
        self.read_data()
        
        self.read_game_values()

        self.read_wins()

        if VERBOSE:
            self.print_analysis()

class BatchAnalyzer:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        
        self.sarsa_df = pd.DataFrame(columns=['regressor', 'start clock', 'winrate'])
        self.sarsa_df_dict = {'regressor':[],'start clock':[], 'winrate':[]}
        self.mcts_df = pd.DataFrame(columns=['start clock', 'winrate'])

    def plot_analysis(self):
        plt.figure(figsize=(15, 10))
        sns.lineplot(x="start clock", y="winrate",
             hue="regressor", data=self.sarsa_df)
        plt.show()

    #Analyze all csv files in given directory
    def batch_analyze(self):
        for filename in os.listdir(self.dir_name):
            file_analyzer = LogAnalyzer(self.dir_name + "\\" + filename)
            file_analyzer.run_analysis()

            self.sarsa_df_dict['regressor'].append(file_analyzer.p1_regr)
            self.sarsa_df_dict['start clock'].append(file_analyzer.start_clock)
            self.sarsa_df_dict['winrate'].append(100-file_analyzer.mcts_win_rate)

        self.sarsa_df = pd.DataFrame.from_dict(self.sarsa_df_dict)
        
        self.plot_analysis()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        VERBOSE = True
    analyzer = BatchAnalyzer("C:\\Hera_HR\\Lokaverkefni\\Logs\\logs_090519")
    analyzer.batch_analyze()
    