#This will run a batch of the following command:
#./gameServerRunner <results directory> <game key> <start clock> <play clock> <player host 1> <player port 1> <player name 1> <player host 2> <player port 2> <player name 2> etc.

import subprocess, os, sys, time, datetime, csv, json, fileinput
from PlayerClient import PlayerClient

class Player_info:
        def __init__(self, ip, p_type, port, regressor="sgd", client=None):
            self.ip = ip
            self.type = p_type
            self.port = port
            self.regressor = regressor
            self.client = client

class BatchGameRunner:
    game_name = ""
    start_clock = ""
    play_clock = ""
    player1 = None
    player2 = None
    command = ""
    ggp_base_path = ""
    ggp_base_results_file = "results"
    filedir = "logs/"
    filepath = ""
    max_expansions = -1

    #---- File IO
    def get_file_suffix(self, filename):
        stripped = filter(lambda x : x.isdigit(), filename)
        return stripped
        
    #Return incremental IDs for files with the same name.
    def choose_file_suffix(self, logname):
        our_path = os.path.dirname(os.path.realpath(__file__))
        filenames = os.listdir(our_path +  "/" +self.filedir)

        highest_count = -1
        for filename in filenames:
            if logname in filename:
                file_count = int(self.get_file_suffix(filename))
                if file_count > highest_count:
                    highest_count = file_count
        return highest_count + 1     

    #Write metadata header to csv file
    def write_metadata(self, runs):
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        with open(self.filepath, 'a') as log_file:

            log_file.write("#Timestamp: " + timestamp + '\n')
            log_file.write("#Game name: " + self.game_name + '\n')
            log_file.write("#Player 1: " + self.player1.type + ' ' + self.player1.regressor + '\n')
            log_file.write("#Player 2: " + self.player2.type + ' ' + self.player2.regressor + '\n')
            log_file.write("#No. runs: " + str(runs) + '\n')
            log_file.write("#Max expansions: " + self.max_expansions + '\n')
            log_file.write("#Total runtime: N/A" + '\n') 
            log_file.write("#Start clock: " + self.start_clock + '\n')
            log_file.write("#Play clock: " + self.play_clock + '\n')
            log_file.write("Winner, Player 1 sarsa iterations, Player 1 list of iterations per state, Player 1 list of time taken per state, Player 2 sarsa iterations, Player 2 list of iterations per state, Player 2 time taken per state, Number of moves made" + '\n')

    def update_total_runtime(self):
        time_now = str(time.time() - self.time_start)
        file = fileinput.FileInput(self.filepath, inplace=True)
        
        for line in file:
            if '#Total runtime: ' in line:
                sys.stdout.write('#Total runtime: ' + time_now + '\n')
            else:
                sys.stdout.write(line)

    def process_player_data(self, data_string):
        if isinstance(data_string, str):
            data_string.replace('\n', '')
            data_list = data_string.split(',')
            return data_list[0] + "," + data_list[1] + "," + data_list[2]
        else:
            return ",,"
        
             
    def write_game(self):
        goals, move_count = self.get_server_json(True)
        winner = ""

        if goals[0] > goals[1]:
            winner = "Player 1"
        else:
            winner = "Player 2"

        p1_data = self.player1.client.command("cat ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")
        p2_data = self.player2.client.command("cat ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")

        with open(self.filepath, 'a') as log_file:
            log_file.write(winner + ',')
            #log_file.write(self.process_player_data(p1_data) +  ',')
            #log_file.write(self.process_player_data(p2_data) +  ',')
            log_file.write(p1_data +  ',')
            log_file.write(p2_data +  ',')
            log_file.write(str(move_count) + '\n')

        self.player1.client.command("rm ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")
        self.player2.client.command("rm ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")

    #Get final rewards for each player and total move count from json string
    def get_server_json(self, remove_file = True):
        path = self.ggp_base_path + "/" + self.ggp_base_results_file
        filenames = os.listdir(path)
        filenames = filter(lambda x: '.json' in x, filenames)
        filename = filenames[0]

        goals, move_count = None, 0
        with open(path + "/" + filename, 'r') as results_file:
            info = json.loads(results_file.read())
            goals = info['goalValues']
            moves = info['moves']
            move_count = len(moves)
        if remove_file:
            remove_process = subprocess.Popen("rm " + filename, cwd=path, shell=True)
            remove_process.wait()
        return goals, move_count

    #---- Experiment execution

    def start_player(self, player):
        player.client = PlayerClient(player.ip)
        player.client.enter_project_dir()
        player.client.update_player_repo()
        player.client.start_player(player.type, player.regressor, player.port, self.max_expansions)

    def setup(self, game, start_clock, play_clock, player1_data, player2_data, max_expansions=-1):
        self.game_name = game
        self.start_clock = start_clock
        self.play_clock = play_clock
        
        self.player1 = player1_data
        self.player2 = player2_data
        
        self.max_expansions = max_expansions
        
        #Saving environment variables as a dict
        environment_vars = os.environ.copy()
        self.ggp_base_path = environment_vars["GGPLIB_PATH"] + "/ggp-base"
        server_command = "./gradlew gameServerRunner"

        self.command = server_command + " -Pmyargs=\"" + "results" + " " + self.game_name + " " + self.start_clock + " " + self.play_clock + " " \
                    + self.player1.ip + " " + str(self.player1.port) + " " + self.player1.type + " " \
                    + self.player2.ip + " " + str(self.player2.port) + " " + self.player2.type + "\""

        self.start_player(self.player1)
        self.start_player(self.player2)

        filename = self.game_name + "_" + self.player1.type + "_" + self.player2.type + "_"
        filename += str(self.choose_file_suffix(filename)) + ".csv"
        self.filepath = self.filedir + filename

    def run_tests(self, runs):
        self.write_metadata(runs)
        self.time_start = time.time()

        for iteration in range(runs):
            process = subprocess.Popen(self.command, cwd=self.ggp_base_path, shell=True)
            process.wait()

            self.write_game()

            print "Run #" + str(iteration+1) +" finished!"
            self.update_total_runtime()

        print "Batch run finished."

if __name__ == "__main__":
    '''
    if len(sys.argv) < 9:
        print("Error: Invalid arguments. Usage:")
        print("<nr. of runs> <game key> <start clock> <play clock> <player 1 IP> <player 1 type> <player 1 regressor>* <player 2 IP> <player 2 type> <player 2 regressor>* <max expansions>*")
        print("*optional, all must be set if used")
        print("Example: ticTacToe 20 10 127.0.0.1 random 127.0.0.1 random")
        exit()
    '''

    gameRunner = BatchGameRunner()
    gameRunner.setup("connectFour", 10, 10, ("p1_ip", "sarsa", 1337), ("p2_ip", "mcts", 1337))
    gameRunner.run_tests(10)
    
