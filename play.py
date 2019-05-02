
from twisted.internet import reactor
from twisted.web import server

from ggplib.util import log
from ggplib import interface
from ggplib.web.server import GGPServer


from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

from SarsaPlayer import SarsaPlayer
from CarlPlayer import CarlPlayer
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

    regressor = None
    expansions = -1
    if len(sys.argv) >= 4:
        regressor_name = sys.argv[3]

        if regressor_name.lower() == "sgd":
            regressor = SGDRegressor()
        elif regressor_name.lower() == "mlp":
            regressor = MLPRegressor((40, 20, 40))
        elif regressor_name.lower() == "paggro":
            regressor = PassiveAggressiveRegressor()

        if len(sys.argv) >= 5:
            expansions = int(sys.argv[4])

        if regressor is None:
            print("Invalid regressor given, please choose from the following: sgd, mlp, paggro, knn, hoeffding, forest, bagging")
    else:
        regressor = SGDRegressor()

    player = None
    if player_name.lower() == "sarsa":
        player = SarsaPlayer(regressor, "sarsa")
    elif player_name.lower() == "carl_playout":
        player = CarlPlayer("ucb","sarsa", regressor, expansions, "Carl with SARSA in playout")
    elif player_name.lower() == "carl_selection":
        player = CarlPlayer("sucb", "random", regressor, expansions, "Carl with SARSA in selection")
    elif player_name.lower() == "carl_full":
        player = CarlPlayer("sucb", "sarsa", regressor, expansions, "Carl with SARSA in selection and playout")
    elif player_name.lower() == "mcts":
        player = CarlPlayer("ucb", "random", regressor, expansions, "MCTS")
    elif player_name.lower() == "random":
        player = RandomPlayer("Random")
    else:
        print("No valid player chosen")
        exit()
    
    play_runner(player, port)


###############################################################################

if __name__ == "__main__":
    main()
