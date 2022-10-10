import torch
import torch.nn as nn
import numpy as np


class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)) -> None:
        super().__init__()
        # self.bn = nn.BatchNorm1d(obs_dim)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(obs_dim, hidden_sizes[0])] + \
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)] + \
            [nn.Linear(hidden_sizes[-1], act_dim)]
        )
        self.act_fn = nn.Tanh()
    
    def forward(self, x):
        # out = self.bn(x)
        out = x
        for i in range(len(self.linear_layers) - 1):
            out = self.act_fn(self.linear_layers[i](out))
        out = self.linear_layers[-1](out)
        return out


class GaussianMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)) -> None:
        super().__init__()
        self.action_dim = act_dim
        self.fc_network = FCNetwork(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32), requires_grad=True)
    
    def forward(self, x):
        action_mean = self.fc_network.forward(x)
        action_std = torch.exp(self.log_std)
        return (action_mean, action_std)
    
    def take_action(self, x, deterministic: bool = False):
        action_mean, action_std = self.forward(x)
        if deterministic:
            return action_mean
        else:
            return action_mean + torch.randn_like(action_std) * action_std
    
    def log_likelihood(self, x, actions):
        action_mean, action_std = self.forward(x)
        zs = (actions - action_mean) / action_std
        return - 0.5 * torch.sum(zs ** 2, dim=1) - torch.sum(self.log_std) - 0.5 * self.action_dim * np.log(2 * np.pi)
