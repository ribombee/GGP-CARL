    #0       1                   2       3                                  4                    5                        6                  7
    #Winner, P1 Error over time, P1 EEF, P1 List of errors per observation, P2 Sarsa iterations, P2 iterations per state, P2 time per state, moves

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
from collections import OrderedDict

VERBOSE = False

#Helper function to get list average
def list_avg(lst):
    if len(lst) == 0:
        return 0
    else:    
        return sum(lst) / len(lst)

class MatchFile:
    def __init__(self, file_name):
        self.file_name = file_name
    
        self.log_df = None

        self.p1_name = "p1"
        self.p1_wins = 0
        self.p1_regr = "NO REGR"

        self.p2_name = "p2"
        self.p2_wins = 0
        self.p1_regr = "NO REGR"

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
        self.log_df = pd.read_csv(self.file_name, skiprows=9)

    def try_append(self, lst, val, func):
        try:
            item = func(val)
            lst.append(item)
        except:
            pass

    #Calculate averages
    def read_game_values(self):
        EOT = []
        EEF = []
        self.instance_errors = []
        sarsa_iters = []
        state_iters = []
        state_times = []
        moves = []

        get_list = lambda val : [float(x) for x in val.split(';')]
        average_list = lambda val : list_avg(get_list(val))

        conv_float = lambda val : float(val)
        for row_no, row in self.log_df.iterrows():
            try:
                if row_no == 0:
                    self.try_append(EOT, row[1], conv_float)
                    self.try_append(EEF, row[2], conv_float)
                    self.instance_errors.append(get_list(row[3]))
                    self.try_append(sarsa_iters, row[4], conv_float)

                self.try_append(state_iters, row[5], average_list)
                self.try_append(state_times, row[6], average_list)
                self.try_append(moves, row[7], conv_float)
            except:
                pass
        
        #Overall averages
        self.EOT_avg = list_avg(EOT)
        self.EEF_avg = list_avg(EEF)

        self.sarsa_iters_avg = list_avg(sarsa_iters)
        self.state_iters_avg = list_avg(state_iters)
        self.state_times_avg = list_avg(state_times)

        self.move_avg = list_avg(moves)

    #Read player win count
    def read_wins(self):
        for _, row in self.log_df.iterrows():
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

        self.matches = []

        self.regressor_df = pd.DataFrame(columns=['regressor', 'start clock', 'winrate', 'eot', 'eef'])
        self.regressor_dict = {'regressor':[],'start clock':[], 'winrate':[], 'eot':[], 'eef':[]}

        self.instance_error_dict = {'sgd': [], 'mlp': [], 'paggro': []}
        self.instance_error_df = pd.DataFrame(columns=['regressor', 'instance errors', 'steps'])

        self.mcts_df = pd.DataFrame(columns=['start clock', 'winrate'])

    def update_error_average(self, new_errors, error_avgs, error_count_list):
        list_range = 0
        if len(new_errors) > len(error_avgs):
            list_range = len(error_avgs)
            dif = len(new_errors) - len(error_avgs)
            error_avgs += new_errors[len(error_avgs):]
            error_count_list += [1]*dif
        else:
            list_range = len(new_errors)

        for i in range(list_range):
            error_avgs[i] = float(error_avgs[i]*error_count_list[i]+new_errors[i])/float(error_count_list[i]+1)
            error_count_list[i] += 1

    def get_error_avgs(self):
        error_avgs = {'sgd': [], 'mlp': [], 'paggro': []}
        
        for key in self.instance_error_dict:
            instance_error_count = []
            instance_error_avg = []
            for instance_errors in self.instance_error_dict[key]:
                self.update_error_average(instance_errors, instance_error_avg, instance_error_count)
            error_avgs[key] = instance_error_avg

        return error_avgs

    def adjust_graph_fontsize(self, ax):
        plt.setp(ax.get_legend().get_texts(), fontsize='22') 
        plt.setp(ax.get_legend().get_title(), fontsize='32')
        ax.set_xlabel(ax.get_xlabel(),fontsize=22)
        ax.set_ylabel(ax.get_ylabel(),fontsize=22)

    def visualize(self):
        plt.figure(figsize=(15, 10))
        ax = sns.lineplot(x="start clock", y="winrate",
            hue="regressor", data=self.regressor_df)
        self.adjust_graph_fontsize(ax)
        plt.show()

        plt.figure(figsize=(15,10))
        ax = sns.lineplot(x="start clock", y = "eot",
            hue="regressor", data = self.regressor_df)
        self.adjust_graph_fontsize(ax)
        plt.show()

        plt.figure(figsize=(15,10))
        ax = sns.lineplot(x="start clock", y = "eef",
            hue = "regressor", data = self.regressor_df)
        self.adjust_graph_fontsize(ax)
        plt.show()

        '''
        plt.figure(figsize=(15,10))
        ax = sns.lmplot(x="steps", y = "instance errors",
            hue = "regressor", data = self.instance_error_df)
        plt.show()
        '''

    #Analyze all csv files in given directory
    def analyze_batch(self):
        for filename in os.listdir(self.dir_name):
            match = MatchFile(self.dir_name + "/" + filename)
            match.run_analysis()
            
            self.matches.append(match)
            
            if match.start_clock == 1200:
                self.instance_error_dict[match.p1_regr] += match.instance_errors

            self.regressor_dict['regressor'].append(match.p1_regr)
            self.regressor_dict['start clock'].append(match.start_clock)
            self.regressor_dict['winrate'].append(100-match.mcts_win_rate)
            self.regressor_dict['eot'].append(match.EOT_avg)
            self.regressor_dict['eef'].append(match.EEF_avg)

        error_avg = self.get_error_avgs()
        error_avg_dict = {'regressor':[], 'instance errors':[], 'steps':[]}
        for key in error_avg:
            list_len = len(error_avg[key])
            for index in range(list_len):
                error_avg_dict['regressor'].append(key)
                error_avg_dict['instance errors'].append(error_avg[key][index])
                error_avg_dict['steps'].append(index*1000)
                
        self.instance_error_df = pd.DataFrame.from_dict(error_avg_dict)
        self.regressor_df = pd.DataFrame.from_dict(self.regressor_dict)
        
        self.visualize()

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print "args filepath [verbose]"
    elif len(sys.argv) >= 3:
        VERBOSE = True

    analyzer = BatchAnalyzer(sys.argv[1])
    analyzer.analyze_batch()
    