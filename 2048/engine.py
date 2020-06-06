import gym_2048
import gym
import os
import torch
import random
import math
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from itertools import count
from torch.utils.tensorboard import SummaryWriter

import model
import custom_utils
from replay_memory import Transition, ReplayMemory

class Engine:
    def __init__(self,
                 logdir='./logdir',
                 phase='train',
                 batch_size=128,
                 gamma=0.999,
                 eps_start=0.9,
                 eps_end=0.1,
                 eps_decay=500,
                 num_episodes=10000):
        self.logdir = logdir
        self.phase = phase
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_episodes = num_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make('2048-v0')

        # if self.phase == 'train':
        #     self.env.seed(42)

        self.policy_net = model.DQN()
        self.policy_net.to(self.device)

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)

        self.n_actions = self.env.action_space.n
        self.steps_done = 0

        self.writer = SummaryWriter(os.path.join(self.logdir, 'summaries'))

    def get_state(self, board):
        board_flatten = []

        for row in board:
            for num in row:
                board_flatten.extend(custom_utils.num_to_vector(num))

        board_flatten = np.array(board_flatten, dtype=np.float32)
        return torch.from_numpy(board_flatten).to(self.device).unsqueeze(0)

    def select_action(self, state, exploration=True):
        action = -1
        if not exploration:
            with torch.no_grad():
                action = self.policy_net(state).max(dim=1)[1].view(1, 1)
        else:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1

            if sample > eps_threshold:
                with torch.no_grad():
                    action = self.policy_net(state).max(dim=1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = self.policy_net(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print('Loss: {}'.format(loss.item()))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        for i_episode in range(self.num_episodes):
            self.env.reset()
            state = self.get_state(self.env.board)

            for t in count():
                if t % 10 == 0:
                    print('Episode {}/{}, iter {}'.format(i_episode + 1, self.num_episodes, t + 1))

                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.get_state(next_state)

                reward = (reward - 1.) / 8.
                reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                self.memory.push(state, action, reward, next_state)

                state = next_state

                self.optimize_model()
                if done:
                    break

            # if (i_episode + 1) % 100 == 0:
            #     self.env.reset()
            #     state = self.get_state(self.env.board)
            #     self.env.render()
            #     for t in count():
            #         action = self.select_action(state, exploration=False)
            #         next_state, _, done, _ = self.env.step(action.item())
            #         next_state = self.get_state(next_state)

            #         if torch.all(torch.eq(state, next_state)).item() == True:
            #             break

            #         print('Action: {}'.format(gym_2048.Base2048Env.ACTION_STRING[action.item()]))
            #         self.env.render()
            #         print()
            #         if done:
            #             break

            #         state = next_state
