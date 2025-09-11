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
    if os.path.exists("saved_runners"):
        shutil.rmtree("saved_runners")
    os.makedirs("saved_runners", exist_ok=True)

    max_episodes = 5e4
    max_steps = 5e4

    for runner_id in range(20): 
        print(f"Training runner {runner_id}")
        set_seed(runner_id)
        os.makedirs("saved_runners", exist_ok=True)
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        print(env.action_space)
        print(env.observation_space)

        # Create separate networks for DQN and Policy Gradient
        print("Training Policy Gradient")
        policy_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-3)
        policy_runner = PolicyGradientRunner(env, policy_net, policy_optimizer, device)
        policy_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        policy_runner.save(f"saved_runners/PolicyGradientRunner_{runner_id}.pkl")

        print("Training Actor Critic")
        actor_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        critic_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=1)
        actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=5e-3)
        ac_runner = ActorCriticRunner(env, actor_net, critic_net, actor_optimizer, critic_optimizer, device)
        ac_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        ac_runner.save(f"saved_runners/ActorCriticRunner_{runner_id}.pkl")

        

        print("Training DQN")
        dqn_net = Feedforward(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        dqn_optimizer = torch.optim.Adam(dqn_net.parameters(), lr=5e-3)
        dqn_runner = DQNRunner(env, dqn_net, dqn_optimizer, device)
        dqn_runner.run(n_episodes = max_episodes, max_steps = max_steps)
        dqn_runner.save(f"saved_runners/DQNRunner_{runner_id}.pkl")

        

        

        

        

if __name__ == "__main__":
    main()