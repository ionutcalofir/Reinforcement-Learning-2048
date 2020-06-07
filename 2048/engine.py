import gym_2048
import gym
import os
import torch
import random
import math
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
                 gamma=0.999,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=2000,
                 num_episodes=1000000):
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

        self.policy_net = model.DQN().to(self.device)

        self.target_net = model.DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001, weight_decay=0.000001)
        self.memory = ReplayMemory(10000)

        self.n_actions = self.env.action_space.n
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
        action = -1
        if inference:
            with torch.no_grad():
                action = self.policy_net(state).max(dim=1)[1].view(1, 1)
        elif eps_greedy:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1

            if sample > eps_threshold:
                with torch.no_grad():
                    action = self.policy_net(state).max(dim=1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            softmax = nn.Softmax(dim=1)
            with torch.no_grad():
                action = softmax(self.policy_net(state))
                act = np.random.choice([0, 1, 2, 3], p=action.detach().cpu().numpy()[0])

                return torch.tensor([[act]], device=self.device, dtype=torch.long)

        return action

    def optimize_model(self):
        if len(self.memory) < 5000:
            return 0

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
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        tries = 0
        running_loss = 0.
        running_loss_cnt = 1
        logging.info('Episode {}/{}'.format(0, self.num_episodes))
        for i_episode in range(self.num_episodes):
            self.env.reset()
            state = self.get_state(self.env.board)

            for t in count():
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.get_state(next_state)

                reward = (reward - 1.) / 8.
                if torch.all(torch.eq(state, next_state)).item() == True:
                    reward = -1

                if done:
                    next_state = None

                reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                self.memory.push(state, action, reward, next_state)

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

                if torch.all(torch.eq(state, next_state)).item() == True:
                    tries += 1
                    if tries == 30:
                        break
                else:
                    tries = 0

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

                logging.info('Episode {}/{}\nScore {}\nMax tile {}'.format(i_episode + 1, self.num_episodes,
                                                                           score,
                                                                           max_tile))
                self.env.render()

            if i_episode % 999 == 0:
                torch.save(self.policy_net.state_dict(), os.path.join(self.logdir, 'models', 'model_{}.pt'.format(i_episode)))
