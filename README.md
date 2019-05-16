# GGP-CARL

This project was done as a thesis for a Bachelor of Science in Computer Science - Research Based at Reykjavík University. The results of our research are available in the form of a research report which is published on skemman.is. Our instructor for this project was Dr Stephan Schiffel, the project was done for Reykjavík University's Center for the Analysis and Design of Intelligent Agents (CADIA).

We propose an agent that learns generalizable information through reinforcement learning which it can then apply in search control for MCTS.

## Abstract
This paper proposes CARL, a pair of agents thatapply  reinforcement  learning  and  function  ap-proximation using regression to learn policies forgames where human heuristics cannot be applied.The purpose of these policies is to do search control in Monte Carlo Tree Search (MCTS), a heuristic search algorithm to see if the learned policiescan outperform upper confidence bound for trees(UCT)

# Requirements and setup

This project was written in Python 2.7 on Ubuntu 16.04.
We rely heavily on Richard Emslie's [ggplib](https://github.com/richemslie/ggplib).
We have written a shell script that can be found in /etc/SetupScript.sh, which will fetch required packages and repositories, as well as set up the environment required for the project.

This project has been written and tested on Ubuntu 16.04, and a vast majority of our experiments have been run on the google cloud computing service. The experiment scripts are provided under /experiment_scripts, but under no warranty of them working under other circumstances.

# Experiments

The experiments we ran can be seen in the research report on skemman.is. The code used to run the experiments is provided as part of this repository. 

## Running experiments
To run the experiments, you need to run ConsecutiveTestRunner. As an argument, pass in the name of a file containing pairs of ip addresses on which to run the agents. The ip addresses should be formatted so that the pairs are in the same line separated by a space. i.e.

IP1 IP2 \
IP3 IP4 \
IP5 IP6 


# Acknowledgements

This code is built in large part on Richard Emslie's work, in particular ggplib and the accompanying implementation of MCS. It is also built on Reykjavík University's CADIA player.
