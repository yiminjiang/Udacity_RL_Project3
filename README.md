# Udacity_RL_Project3

Project: Collaboration and Competition

Project Details:

For this project, two agents are trained to control rackets to bounce a ball over a net.

If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

Getting Started:
To run the code, you need to have Mac with Anaconda with Python 3.6 installed.

To download Anaconda, please click the link below:

https://www.anaconda.com/download/

Clone or download and unzip the DRLND_P3_collab-compet folder.

Download by clicking the link below and unzip the environment file under DRLND_P3_collab-compet folder

https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

Download by clicking the link below and unzip the ml-agents file under DRLND_P3_collab-compet folder

https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b

Dependencies :
To set up your Python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.
conda create --name drlnd python=3.6
activate drlnd
Install Pytorch by following the instructions in the link below.

https://pytorch.org/get-started/locally/

Then navigate to DRLND_P3_collab-compet/ml-agents-0.4.0b/python and install ml-agent.

pip install .
Install matplotlib for plotting graphs.

conda install -c conda-forge matplotlib
(Optional) Install latest prompt_toolkit may help in case Jupyter Kernel keeps dying

conda install -c anaconda prompt_toolkit 

Run the code

Open Tennis.ipynb in Jupyter and press Shift+Enter to run the first cell to import all the libraries.

(1) Fill the replay buffer by running the cell which contains "fill_memory function".

(2) Train an agent by running the cells which contains the "maddpg_train" function.

The environment is considered as solved when the average score of last 100 eposides is greater than 0.5

Saved model files: check_pt_actor.pth and check_pt_critic.pth.

Results are saved in the file "results.png".
