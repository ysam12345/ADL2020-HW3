import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import json
from agent_dir.agent import Agent
from environment import Environment

DEVICE = 'cuda:1'

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64).to(DEVICE)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
    '''
    def make_action(self, state, test=False):
        # action = self.env.action_space.sample() # TODO: Replace this line!
        # 1. Use your model to output distribution over actions and sample from it.
        #    HINT: torch.distributions.Categorical 
        # 2. Save action probability in self.saved_action

        state_tensor = torch.Tensor([state]).to(DEVICE)
        self.model.eval()
        prediction = self.model(state_tensor)
        #print(prediction)
        probability = torch.distributions.Categorical(prediction)
        action = probability.sample().cpu().numpy()[0]
        #print(action)
        self.model.train()
        self.saved_actions.append((action, probability.probs[0].cpu().detach().numpy()))
       
        #print(probability.probs[0].detach().numpy())
        return action
    '''
    '''
    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        #print(len(self.rewards))
        #print(len(self.saved_actions))
        R = []
        
        for i in range(len(self.rewards)):
            R.append(sum([self.gamma**j * self.rewards[j] for j in range(len(self.rewards)-i)]))
       	
            
        #print(R)
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        
        loss = sum([-R[i] * log(self.saved_actions[i][1][self.saved_actions[i][0]]) for i in range(len(R))])
        loss = torch.tensor(loss, requires_grad=True).to(DEVICE)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    '''
    
    def make_action(self, state, test=False):
        #https://pytorch.apachecn.org/docs/0.3/distributions.html
        state_tensor = torch.Tensor([state]).to(DEVICE)
        prediction = self.model(state_tensor)
        probability = torch.distributions.Categorical(prediction)
        action = probability.sample()
        self.saved_actions.append(probability.log_prob(action))
        return action.item()
    
    def update(self):
        reward = []
        R = 0
        for i in reversed(self.rewards):
            R = i + self.gamma * R
            reward.append(R)
        reward = list(reversed(reward))

        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        loss = 0
        for i in range(len(reward)):
            loss += -reward[i] * self.saved_actions[i]
        #loss = torch.tensor(loss, requires_grad=True).to(DEVICE)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        learning_curve = dict()
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            learning_curve[epoch] = avg_reward
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                 

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                with open('plot/pg_learning_curve.json', 'w') as f:
                    json.dump(learning_curve, f)
                break
