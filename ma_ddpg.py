from ddpg_agent import DDPG_Agent
from memory import ReplayBuffer
import torch
import numpy as np
from collections import namedtuple, deque
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001        # L2 weight decay
UPDATE_EVERY = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, num_agent, state_size, action_size):
        
        super(MADDPG, self).__init__()
        self.num_agent = num_agent
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 0.0
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.gamma = GAMMA
        
        self.iter = 0
        # when not trained an episode is about 13 steps (time for ball to fall)
        self.update_every_iter = 10 
        self.buffer_size = BUFFER_SIZE

        # Changed to only one agent and achieved more stable learning
        self.agents = []
        self.tmp_agent = DDPG_Agent(state_size=self.state_size, hidden_in_actor=400, hidden_out_actor=300, action_size=self.action_size, all_state_size=self.state_size*self.num_agent, hidden_in_critic=400, hidden_out_critic=300, all_action_size= self.action_size*self.num_agent, lr_actor=1.0e-4, lr_critic=1.0e-3, tau = self.gamma, random_seed =self.seed)
        
        for i in range (num_agent):
            self.agents.append(self.tmp_agent)
        
        # create central memory
        self.memory = ReplayBuffer(self.action_size,self.buffer_size,self.batch_size,self.seed)


    def act(self, all_states, add_noise=True):     
        actions=[]
        for i in range(len(self.agents)):
            agent = self.agents[i]
            action = agent.act((all_states[i]),add_noise)
            actions.append(action)
        return actions
        
    def step(self, state, action, reward, next_state, done):
        action= np.array(action).reshape(1, -1)
        state = state.reshape(1, -1)  
        next_state = next_state.reshape(1, -1)  
        
        self.memory.add(state, action, reward, next_state, done)


        self.iter = (self.iter + 1) % self.update_every_iter

        # If memory contains more experiences than required for a batch
        if self.iter == 0 and len(self.memory) > self.memory.batch_size:
            experiences = [] 
            # get different experiences for the different agent
            for i in range(self.num_agent):
                experiences.append(self.memory.sample())
            self.learn(experiences, self.gamma)
                   
    def learn(self, experiences, gamma):
        
        for i in range(len(self.agents)):
            agent = self.agents[i] 
            agent_id = torch.tensor([i]).to(device)
            experience = experiences[i]
            all_states, all_actions, all_rewards, all_next_states, all_dones = experience
            
            actions_next = []
            for j in range(len(self.agents)):
                target = self.agents[j]
                next_states = all_next_states.reshape(-1, self.num_agent, self.state_size).index_select(1, agent_id).squeeze(1)
                actions_next.append(target.actor_target(next_states))

            actions_pred = []
            for j in range(len(self.agents)):
                local = self.agents[j]
                states = all_states.reshape(-1, self.num_agent, self.state_size).index_select(1, agent_id).squeeze(1)
                actions_pred.append(local.actor_local(states))
                if j != i:
                    actions_pred[j] = actions_pred[j].detach()
                    
            agent.learn(experience,gamma,actions_next,actions_pred,i)

                          