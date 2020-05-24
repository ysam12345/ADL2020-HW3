import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, exp
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

        # saved rewards, states and actions
        self.rewards, self.states, self.actions, self.probs = [], [], [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.probs = [], []
    
    def get_probability(self, state, action):
        state_tensor = torch.Tensor([state]).to(DEVICE)
        action_tensor = torch.Tensor([action]).to(DEVICE)
        with torch.no_grad():
            prediction = self.model(state_tensor)
            probability = torch.distributions.Categorical(prediction)
        return probability.log_prob(action_tensor)

    def make_action(self, state, test=False):
        #https://pytorch.apachecn.org/docs/0.3/distributions.html
        state_tensor = torch.Tensor([state]).to(DEVICE)
        prediction = self.model(state_tensor)
        probability = torch.distributions.Categorical(prediction)
        action = probability.sample()
        self.probs.append(probability.log_prob(action))
        return action.item()
    
    def update(self):
        reward = []
        R = 0
        for i in reversed(self.rewards):
            R = i + self.gamma * R
            reward.append(R)
        reward = list(reversed(reward))

        Z = 0
        b = 0
        Z_s = []

        p = 0
        q = 0
        for i in range(len(reward)):
            p += self.get_probability(self.states[i], self.actions[i])
            q += self.probs[i]
        z = exp(p) / exp(q)
        b = z * sum(reward) / len(reward)



        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        loss = 0
        for i in range(len(reward)):
            loss += - (reward[i] - b) * self.probs[i]
        loss /= z

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
                self.states.append(state)
                self.actions.append(action)

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
