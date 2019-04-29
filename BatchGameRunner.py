#This will run a batch of the following command:
#./gameServerRunner <results directory> <game key> <start clock> <play clock> <player host 1> <player port 1> <player name 1> <player host 2> <player port 2> <player name 2> etc.

import subprocess, os, sys, time, datetime, csv, json, fileinput
from PlayerClient import PlayerClient

class BatchGameRunner:
    runs = 0
    game_name = ""
    start_clock = ""
    play_clock = ""
    player1_ip, player1_type, player1_port, player1_client = "", "", 0, None
    player2_ip, player2_type, player2_port, player2_client = "", "", 0, None
    command = ""
    ggp_base_path = ""
    ggp_base_results_file = "results"
    filedir = "logs/"
    filepath = ""

    
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

    def update_player_repo(self, client):
        client.shell_send("git pull")
        time.sleep(1)

    def setup(self):
        #Saving environment variables as a dict
        environment_vars = os.environ.copy()

        if len(sys.argv) < 9:
            print("Error: Invalid arguments. Usage:")
            print("<nr. of runs> <game key> <start clock> <play clock> <player 1 IP> <player 1 type> <player 2 IP> <player 2 type>")
            print("Example: ticTacToe 20 10 127.0.0.1 random 127.0.0.1 random")
            exit()
        
        self.runs = int(sys.argv[1])
        self.game_name = sys.argv[2]
        self.start_clock = sys.argv[3]
        self.play_clock = sys.argv[4]
        self.player1_ip, self.player1_type, self.player1_port = sys.argv[5], sys.argv[6], 1337
        self.player2_ip, self.player2_type, self.player2_port = sys.argv[7], sys.argv[8], 1337

        #If players are on same address, make player 2 use a different port
        if self.player1_ip == self.player2_ip:
            self.player2_port = 1338
        
        self.ggp_base_path = environment_vars["GGPLIB_PATH"] + "/ggp-base"
        server_command = "./gradlew gameServerRunner"

        self.command = server_command + " -Pmyargs=\"" + "results" + " " + self.game_name + " " + self.start_clock + " " + self.play_clock + " " \
                    + self.player1_ip + " " + str(self.player1_port) + " " + self.player1_type + " " \
                    + self.player2_ip + " " + str(self.player2_port) + " " + self.player2_type + "\""
        
        self.player1_client = PlayerClient(self.player1_ip, "koder")
        self.player1_client.start_player(self.player1_type, self.player1_port)
        
        self.player2_client = PlayerClient(self.player2_ip, "koder")
        self.player2_client.start_player(self.player2_type, self.player2_port)

        self.update_player_repo(self.player1_client)
        self.update_player_repo(self.player2_client)

        filename = self.game_name + "_" + self.player1_type + "_" + self.player2_type + "_"
        filename += str(self.choose_file_suffix(filename)) + ".csv"
        self.filepath = self.filedir + filename

    #Write metadata header to csv file
    def write_metadata(self):
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        with open(self.filepath, 'a') as log_file:

            log_file.write("#Timestamp: " + timestamp + '\n')
            log_file.write("#Game name: " + self.game_name + '\n')
            log_file.write("#Player 1: " + self.player1_type + '\n')
            log_file.write("#Player 2: " + self.player2_type + '\n')
            log_file.write("#No. runs: " + str(self.runs) + '\n')
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

        p1_data = self.player1_client.command("cat ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")
        p2_data = self.player2_client.command("cat ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")

        with open(self.filepath, 'a') as log_file:
            log_file.write(winner + ',')
            log_file.write(self.process_player_data(p1_data) +  ',')
            log_file.write(self.process_player_data(p2_data) +  ',')
            log_file.write(str(move_count) + '\n')

        self.player1_client.command("rm ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")
        self.player2_client.command("rm ~/Documents/CarlAgent/GGP-CARL/PlayerLog.csv")

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

    def run_tests(self):
        self.write_metadata()
        self.time_start = time.time()

        for iteration in range(self.runs):
            process = subprocess.Popen(self.command, cwd=self.ggp_base_path, shell=True)
            process.wait()

            self.write_game()

            print "Run #" + str(iteration+1) +" finished!"
            self.update_total_runtime()

        print "Batch run finished."

if __name__ == "__main__":
    gameRunner = BatchGameRunner()
    gameRunner.setup()
    gameRunner.run_tests()
    
