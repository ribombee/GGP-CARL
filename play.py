
from twisted.internet import reactor
from twisted.web import server

from ggplib.util import log
from ggplib import interface
from ggplib.web.server import GGPServer

from SarsaPlayer import SarsaPlayer
from CarlPlayer import CarlPlayer
from OurMCTS import MCTSPlayer
from ggplib.player.random_player import RandomPlayer 

def play_runner(player, port):
    interface.initialise_k273(1, log_name_base=player.get_name())
    log.initialise()

    ggp = GGPServer()
    ggp.set_player(player)
    site = server.Site(ggp)

    log.info("Running player '%s' on port %d" % (player.get_name(), port))

    reactor.listenTCP(port, site)
    reactor.run()


def main():
    import sys

    player_name = sys.argv[1]
    port = int(sys.argv[2])

    player = None
    if player_name.lower() == "sarsa":
        player = SarsaPlayer("sarsa")
    elif player_name.lower() == "carl_playout":
        player = CarlPlayer("ucb", "sarsa", "Carl with SARSA in playout")
    elif player_name.lower() == "carl_selection":
        player = CarlPlayer("sucb", "random", "Carl with SARSA in selection")
    elif player_name.lower() == "carl_full":
        player = CarlPlayer("sucb", "sarsa", "Carl with SARSA in selection and playout")
    elif player_name.lower() == "mcts":
        player = CarlPlayer("ucb", "random", "MCTS")
    elif player_name.lower() == "random":
        player = RandomPlayer("Random")
    else:
        print("No valid player chosen")
        exit()
    
    play_runner(player, port)


###############################################################################

if __name__ == "__main__":
    main()
