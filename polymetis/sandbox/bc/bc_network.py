import torch
import torch.nn as nn
from torch.distributions import Categorical
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
        self.criterion = nn.MSELoss()
    
    def forward(self, x, deterministic=True):
        # out = self.bn(x)
        out = x
        for i in range(len(self.linear_layers) - 1):
            out = self.act_fn(self.linear_layers[i](out))
        out = self.linear_layers[-1](out)
        return out
    
    def get_loss(self, x: torch.Tensor, action: torch.Tensor):
        return self.criterion(self.forward(x), action.detach())

class DiscreteNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim=(3, 3, 3, 3), hidden_sizes=(64, 64)) -> None:
        super().__init__()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(obs_dim, hidden_sizes[0])] + \
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]
        )
        self.action_head = nn.ModuleList(
            [nn.Linear(hidden_sizes[-1], act_dim[i]) for i in range(len(act_dim))]
        )
        self.act_fn = nn.Tanh()
        self.action_dim = act_dim
    
    def _get_logits(self, x):
        out = x
        for i in range(len(self.linear_layers)):
            out = self.act_fn(self.linear_layers[i](out))
        action_logits = []
        for i in range(len(self.action_head)):
            _logit = self.action_head[i](out)
            action_logits.append(_logit)
        return action_logits # a list of length num_action, each of size (batch_size, action_dim[i]) 

    def forward(self, x, deterministic=True):
        action_logits = self._get_logits(x)
        action_dists = [Categorical(logits=logit) for logit in action_logits]
        if not deterministic:
            action = torch.stack([action_dists[i].sample().float() / (self.action_dim[i] - 1) for i in range(len(action_dists))], dim=-1)
            action = 2 * action - 1
        else:
            action = torch.stack([
                2 * torch.argmax(action_logits[i], dim=-1).float() / (self.action_dim[i] - 1) - 1 for i in range(len(action_logits))
            ], dim=-1)
        return action
    
    def get_loss(self, x, action):
        action_logits = self._get_logits(x)
        action_dists = [Categorical(logits=logit) for logit in action_logits]
        action_probs = [nn.functional.softmax(logit, dim=-1) for logit in action_logits]
        log_probs = []
        for i in range(len(action_probs)):
            action_idx = torch.round((action[:, i] + 1) / 2 * (self.action_dim[i] - 1)).long()
            log_probs.append(action_dists[i].log_prob(action_idx))
            # log_probs.append(torch.log(action_probs[i][action_idx]))
        log_probs = torch.stack(log_probs, dim=0).sum(dim=0)
        return -log_probs.mean()


class EncoderFCNetwork(nn.Module):
    def __init__(self, encode_fn: nn.Module, control_net: nn.Module, transform: lambda x: x) -> None:
        super().__init__()
        self.encode_fn = encode_fn
        self.control_net = control_net
        self.transform = transform
    
    def forward(self, image, x):
        processed_image = self.transform(image)
        embed = self.encode_fn.forward(processed_image)
        input = torch.cat([embed, x], dim=-1)
        action = self.control_net.forward(input, deterministic=False)
        return action, input, processed_image


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
