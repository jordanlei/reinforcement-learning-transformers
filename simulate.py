import gymnasium as gym
from network import Feedforward
from runner import DQNRunner, PolicyGradientRunner, ActorCriticRunner
import torch
import numpy as np
import random
import os
import shutil

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def main():
    print("Starting simulation")
    if os.path.exists("saved_runners"):
        shutil.rmtree("saved_runners")
    os.makedirs("saved_runners", exist_ok=True)

    # Lunar Lander specific parameters
    max_episodes = 2e5  # Reduced for faster testing
    max_steps = 2e5  # Lunar Lander episodes are typically shorter

    for runner_id in range(20): 
        print(f"Training runner {runner_id}")
        set_seed(runner_id)
        os.makedirs("saved_runners", exist_ok=True)
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)

        
        print("Training DQN")
        dqn_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, 
                             hidden_dims=[256, 256, 128])
        dqn_optimizer = torch.optim.Adam(dqn_net.parameters(), lr=1e-3)
        dqn_runner = DQNRunner(env, dqn_net, dqn_optimizer, device, batch_size=64, 
                              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
        dqn_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        dqn_runner.save(f"saved_runners/DQNRunner_{runner_id}.pkl")
        
        # Create separate networks for DQN and Policy Gradient with Lunar Lander optimized architecture
        print("Training Policy Gradient")
        policy_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, 
                                hidden_dims=[256, 256, 128])
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)  # Slightly higher LR for Lunar Lander
        policy_runner = PolicyGradientRunner(env, policy_net, policy_optimizer, device)
        policy_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        policy_runner.save(f"saved_runners/PolicyGradientRunner_{runner_id}.pkl")

        print("Training Actor Critic")
        actor_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, 
                               hidden_dims=[256, 256, 128])
        critic_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=1, 
                                hidden_dims=[256, 256, 128])
        actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=1e-4)
        ac_runner = ActorCriticRunner(env, actor_net, critic_net, actor_optimizer, critic_optimizer, device)
        ac_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        ac_runner.save(f"saved_runners/ActorCriticRunner_{runner_id}.pkl")

        

        

        

        

        

if __name__ == "__main__":
    main()