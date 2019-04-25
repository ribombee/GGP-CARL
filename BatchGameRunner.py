#./gameServerRunner <results directory> <game key> <start clock> <play clock> <player host 1> <player port 1> <player name 1> <player host 2> <player port 2> <player name 2> etc.

import subprocess, os, sys

def get_command_vars():

environment_vars = os.environ.copy()
    if len(sys.argv) < 9:
        print("Error: Invalid arguments. Usage:")
        print("<nr. of runs> <game key> <start clock> <play clock> <player 1 IP> <player 1 type> <player 2 IP> <player 2 type>")
        print("Example: ticTacToe 20 10 127.0.0.1 random 127.0.0.1 random")
        exit()
    runs = int(sys.argv[1])
    game_name = sys.argv[2]
    start_clock = sys.argv[3]
    play_clock = sys.argv[4]
    player1_ip, player1_type, player1_port = sys.argv[5], sys.argv[6], 1337
    player2_ip, player2_type, player2_port = sys.argv[7], sys.argv[8], 1337

    #If players are on same address, make player 2 use a different port
    if player1_ip == player2_ip:
        player2_port = 1338
    
    game_result_filename = "results"
    path_to_ggp_base = environment_vars["GGPLIB_PATH"] + "/ggp-base"
    server_command = "./gradlew gameServerRunner"

    command = server_command + " -Pmyargs=\"" + game_result_filename + " " + game_name + " " + start_clock + " " + play_clock + " " \
            + player1_ip + " " + str(player1_port) + " " + player1_type + " " \
            + player2_ip + " " + str(player2_port) + " " + player2_type + "\""
    
    return command, path_to_ggp_base, runs

def start_player(ip):
    process = subprocess.Popen(["ssh", ip])
    pw = raw_input()
    stdout, stderr = process.communicate(pw)
    print stdout
    
def run_tests():
    cmd, ggp_base_path, runs = create_command()
    for iteration in range(runs):
        process = subprocess.Popen(command, cwd=ggp_base_path, shell=True)
        process.wait()
        print "Run #" + str(runs+1) +"finished!"
    subprocess.Popen.terminate()
    print "Batch run finished."
