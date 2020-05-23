
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from time import time
import json

from agent_dir.agent import Agent
from environment import Environment
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.pool = []
        self.size = buffer_size

    def push(self, state, action, reward, next_state):
        if len(self.pool) >= self.size:
            self.pool.pop(0)
        self.pool.append({'state': state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state})

    def sample2batch(self, sample):
        batch_data = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': []
        }
        for s in sample:
            for k in s:
                batch_data[k].append(s[k])
        return batch_data

    def sample(self, batch_size):
        return self.sample2batch(random.sample(self.pool, batch_size))

    def print_pool(self):
        print(self.pool)


class DQN(nn.Module):

    # This architecture is the one from OpenAI Baseline, with small modification.

    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn_2239')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4  # frequency to train the online network
        # before we start to update our network, we wait a few steps first to fill the replay.
        self.learning_start = 10000
        self.batch_size = 32
        self.num_timesteps = 3000000  # total training steps
        self.display_freq = 10  # frequency to display training progress
        self.save_freq = 200000  # frequency to save the model
        self.target_update_freq = 1000  # frequency to update target network
        self.buffer_size = 10000  # max size of replay buffer

        self.epsilon = 1
        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0  # num. of passed steps

        # TODO: initialize your replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def actionNum(self):
        return self.num_actions

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(
                torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(
                torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(
                load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(
                load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps

        #action = self.env.action_space.sample()
        if test:
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)
        if not test and random.random() < self.epsilon:
            action = self.env.action_space.sample()
            self.epsilon = max(0.05, 1 - (self.steps+1) / (self.num_timesteps/3))
            #self.epsilon = max(0, 1 - 10 * (self.steps+1) / (self.num_timesteps/3))
        else:
            state = state.cuda() if use_cuda else state
            prediction = self.online_net(state)
            action = prediction.argmax().item()
        return action

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        update_start = time()
        experience = self.replay_buffer.sample(self.batch_size)
        loss = torch.Tensor([0])
        loss = loss.cuda() if use_cuda else loss
        # print(experience['state'])
        state = torch.cat(experience['state'])
        #action = torch.cat(experience['action'])
        action = experience['action']
        reward = torch.Tensor(experience['reward'])
        #reward = experience['reward']
        next_state = torch.cat(experience['next_state'])

        state = state.cuda() if use_cuda else state
        reward = reward.cuda() if use_cuda else reward
        next_state = next_state.cuda() if use_cuda else next_state

        # step 2: Compute Q(s_t, a) with your model.
        o_q = self.online_net(state)
        # step 3: Compute Q(s_{t+1}, a) with target model.
        with torch.no_grad():
            t_q = self.target_net(next_state)
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        expected_q = reward + self.GAMMA * t_q.max(1)[0]
        # step 5: Compute temporal difference loss
        loss = ((torch.stack([o_q[i][action[i]]for i in range(self.batch_size)])-expected_q)**2).mean()
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        '''
        for experience in experiences:
            # step 2: Compute Q(s_t, a) with your model.
            experience['state'] = experience['state'].cuda() if use_cuda else experience['state']
            experience['next_state'] = experience['next_state'].cuda() if use_cuda else experience['next_state']
            
            o_q = self.online_net(experience['state'])[0][experience['action']]
            # step 3: Compute Q(s_{t+1}, a) with target model.
            with torch.no_grad():
                t_q = self.target_net(experience['next_state'])[0]
            # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
            expected_q = experience['reward'] + self.GAMMA * max(t_q)
            # step 5: Compute temporal difference loss
            diff = (o_q - expected_q)**2
            loss += diff
            # HINT:
            # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
            # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
            #    is the terminal state.
        '''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_end = time()
        return loss.item()

    def train(self):
        start_time = time()
        episodes_done_num = 0  # passed episodes
        total_reward = 0  # compute average reward
        loss = 0
        learning_curve = dict()
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # process new state
                next_state = torch.from_numpy(
                    next_state).permute(2, 0, 1).unsqueeze(0)

                # TODO: store the transition in memory
                self.replay_buffer.push(state, action, reward, next_state)
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(
                        self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn_{}'.format(episodes_done_num))
                    with open('plot/dqn_learning_curve_{}.json'.format(episodes_done_num), 'w') as f:
                        json.dump(learning_curve, f)

                self.steps += 1

            learning_curve[episodes_done_num] = total_reward / \
                self.display_freq

            if episodes_done_num % self.display_freq == 0:
                print('Time: %d | Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f | epsilon %f' %
                      (time()-start_time, episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss, self.epsilon ))
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
