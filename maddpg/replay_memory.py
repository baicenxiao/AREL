import random
from collections import namedtuple
import numpy as np
import pickle

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

Transition_e = namedtuple(
    'Transition_episode', ('states', 'actions', 'masks', 'next_states', 'rewards'))

globals()["Transition"] = Transition
globals()["Transition_episode"] = Transition_e

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)


class ReplayMemory_episode(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_e(*args)
        self.position = (self.position + 1) % self.capacity
        

    def sample(self, batch_size, n_steps_per_episode=8, Rmax=None, Rmin=None):
        n_trajectories = int(batch_size/n_steps_per_episode)
        trajectories = random.sample(self.memory, n_trajectories)

        samples = Transition_e(*zip(*trajectories))

        ind1 = np.arange(n_trajectories)
        ind2 = np.random.randint(25, size=(n_trajectories, n_steps_per_episode))

        states = np.array(samples.states)[ind1[:,None], ind2].reshape(batch_size,-1)
        actions = np.array(samples.actions)[ind1[:,None], ind2].reshape(batch_size,-1)
        masks = np.array(samples.masks)[ind1[:,None], ind2].reshape(batch_size,-1)
        next_states = np.array(samples.next_states)[ind1[:,None], ind2].reshape(batch_size,-1)
        if Rmax:
            rewards = (np.array(samples.rewards)[ind1[:,None], ind2].reshape(batch_size,-1) - Rmin)/(Rmax-Rmin)
        else:
            rewards = np.array(samples.rewards)[ind1[:,None], ind2].reshape(batch_size,-1)

        return Transition(states, actions, masks, next_states, rewards)

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)
