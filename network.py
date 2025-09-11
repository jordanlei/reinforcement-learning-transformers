import torch
import torch.nn as nn

class Feedforward(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Feedforward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
    
    def forward(self, state):
        return self.net(state)

        
        
        