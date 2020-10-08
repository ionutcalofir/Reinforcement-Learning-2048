import gym_2048
import gym
import os
import torch
import random
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from itertools import count
from torch.utils.tensorboard import SummaryWriter

import model
import custom_utils
from replay_memory import Transition, ReplayMemory

class Engine:
    def __init__(self,
                 logdir='./logdir',
                 phase='train',
                 batch_size=256,
                 gamma=0.995,
                 eps_start=1.,
                 eps_end=0.05,
                 eps_decay=0.999,
                 num_episodes=1000000,
                 learning_rate=5e-4,
                 memory_size=10000,
                 update_every=16,
                 tau=1e-3):
        self.logdir = logdir
        self.phase = phase
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.update_every = update_every
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make('2048-v0')

        self.policy_net = model.DQN().to(self.device)

        self.target_net = model.DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(self.memory_size)

        self.n_actions = self.env.action_space.n
        self.eps_threshold = self.eps_start
        self.steps_done = 0

        self.writer = SummaryWriter(os.path.join(self.logdir, 'summaries'))
        os.makedirs(os.path.join(self.logdir, 'models'))

    def get_state(self, board):
        board_flatten = []

        for row in board:
            for num in row:
                board_flatten.extend(custom_utils.num_to_vector(num))

        board_flatten = np.array(board_flatten, dtype=np.float32)
        return torch.from_numpy(board_flatten).to(self.device).unsqueeze(0)

    def select_action(self, state, eps_greedy=False, inference=False):
        self.policy_net.eval()
        action = -1
        if inference:
            with torch.no_grad():
                action = self.policy_net(state).max(dim=1)[1].view(1, 1)
        elif eps_greedy:
            self.steps_done += 1
            sample = random.random()

            if sample > self.eps_threshold:
                with torch.no_grad():
                    action = self.policy_net(state).max(dim=1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else: # probabilistic
            self.steps_done += 1
            softmax = nn.Softmax(dim=1)
            with torch.no_grad():
                action = softmax(self.policy_net(state))
                act = np.random.choice([0, 1, 2, 3], p=action.detach().cpu().numpy()[0])

                action = torch.tensor([[act]], device=self.device, dtype=torch.long)

        self.policy_net.train()
        self.eps_threshold = max(self.eps_threshold * self.eps_decay, self.eps_end)
        return action

    def soft_optimize_model(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1. - self.tau) * target_param.data)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0

        self.policy_net.train()
        self.target_net.eval()

        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Double DQN
        # actions_q_policy = self.policy_net(non_final_next_states).detach().max(1)[1].unsqueeze(1).long()
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, actions_q_policy)[:, 0]

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_optimize_model()

        return loss.item()

    def train(self):
        running_loss = 0.
        running_loss_cnt = 1
        logging.info('Episode {}/{}'.format(0, self.num_episodes))
        for i_episode in range(self.num_episodes):
            self.env.reset()
            state = self.get_state(self.env.board)

            for t in count():
                action = self.select_action(state, eps_greedy=True)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.get_state(next_state)

                reward = reward / 8.
                if torch.all(state == next_state).item() == True:
                    reward = -1.

                if done:
                    next_state = None

                reward = torch.tensor([reward], device=self.device, dtype=torch.float)

                self.memory.push(state, action, reward, next_state)

                if (self.steps_done + 1) % self.update_every == 0:
                    loss = self.optimize_model()
                    running_loss += loss
                    running_loss_cnt += 1

                if done:
                    if (i_episode + 1) % 20 == 0:
                        self.writer.add_scalar('train loss',
                                               running_loss / running_loss_cnt,
                                               i_episode + 1)
                        running_loss = 0.
                        running_loss_cnt = 1

                        self.writer.add_scalar('train_episode_t',
                                               t,
                                               i_episode + 1)
                    break

                state = next_state

            if (i_episode + 1) % 2 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (i_episode + 1) % 20 == 0:
                score = 0
                max_tile = 0

                self.env.reset()
                state = self.get_state(self.env.board)
                for t in count():
                    action = self.select_action(state, inference=True)
                    next_state, _, done, _ = self.env.step(action.item())
                    next_state = self.get_state(next_state)

                    if done or (torch.all(torch.eq(state, next_state)).item() == True):
                        score, max_tile = custom_utils.get_board_score(self.env.board)
                        self.writer.add_scalar('episode_score',
                                               score,
                                               i_episode + 1)
                        self.writer.add_scalar('episode_max_tile',
                                               max_tile,
                                               i_episode + 1)
                        break

                    state = next_state

                logging.info('Episode {}/{}, step {}\nScore {}\nMax tile {}\nEps threshold {}'.format(i_episode + 1, self.num_episodes, self.steps_done,
                                                                                                   score,
                                                                                                   max_tile,
                                                                                                   self.eps_threshold))
                self.env.render()

            if i_episode % 999 == 0:
                torch.save(self.policy_net.state_dict(), os.path.join(self.logdir, 'models', 'model_{}.pt'.format(i_episode)))

    def test(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()

        scores = []
        max_tiles = []

        num_episodes = 1000
        for i in range(num_episodes):
            # self.eps_threshold = 1. # for random
            self.eps_threshold = 0.05

            logging.info('Episode {}/{}'.format(i, num_episodes))
            self.env.reset()
            state = self.get_state(self.env.board)

            for t in count():
                action = self.select_action(state, eps_greedy=True)
                next_state, _, done, _ = self.env.step(action.item())
                next_state = self.get_state(next_state)

                if done:
                    scores.append(np.sum(self.env.board))
                    max_tiles.append(np.amax(self.env.board))
                    break

                state = next_state

        d = {0: 0,
             1: 0,
             2: 0,
             4: 0,
             8: 0,
             16: 0,
             32: 0,
             64: 0,
             128: 0,
             256: 0,
             512: 0,
             1024: 0,
             2048: 0}
        for tile in max_tiles:
            d[tile] += 1

        plt.figure()
        plt.hist(scores)
        plt.suptitle('Scores')
        plt.xlabel('Scores')
        plt.ylabel('Games')

        plt.figure()
        plt.suptitle('Max Tiles')
        plt.bar(range(len(d)), list(d.values()), align='center')
        plt.xticks(range(len(d)), list(d.keys()))
        plt.xlabel('Tiles')
        plt.ylabel('Games')

        plt.show()
