#Neural networks for policy and value

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Policy_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Dropout(p = 0.8)(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.Softmax(dim = 0)(x)


class Value_net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Dropout(p = 0.8)(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

def discount_rewards(rewards, gamma = 0.99):
    """
    Calculate the discounted rewards
    """
    discounted_rewards = np.zeros_like(rewards)
    running = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            running = 0
        running = gamma * running + rewards[i]
        discounted_rewards[i] = running

    return discounted_rewards
