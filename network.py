import torch
import torch.nn as nn

class Feedforward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[50]):
        super(Feedforward, self).__init__()
        
        # Build network dynamically based on hidden_dims
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Add dropout for regularization
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        return self.net(state)


class ConvNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(ConvNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        return self.net(state)

        
        
        