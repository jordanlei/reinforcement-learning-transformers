from jax._src.traceback_util import C
from numpy.testing import break_cycles
import torch 
import random
from collections import deque, defaultdict
import torch.nn.functional as F
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        return state, action, reward, next_state, done


class Runner:
    def __init__(self, env, device, gamma=0.99):
        self.device = device
        self.env = env
        self.gamma = gamma
        self.metrics = defaultdict(list)
        self.steps = 0
    
    def select_action(self, state):
        #returns action and (optional info, e.g. log_prob for policy gradient)
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError
    
    def run(self, n_episodes=1000, max_steps=1e4):
        raise NotImplementedError
    
    def play_and_save_gif(self, title="CartPole Demo", filename="play.gif", show_step=True, show_reward=True):
        """Run a single episode using greedy policy and save as a GIF with matplotlib titles."""
        frames = []
        state, info = self.env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            frame = self.env.render()
            
            # Create matplotlib figure with title
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(frame)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add step and reward info as text
            info_text = ""
            if show_step:
                info_text += f"Step: {step_count}"
            if show_reward:
                if info_text:
                    info_text += f" | Reward: {total_reward:.1f}"
                else:
                    info_text += f"Reward: {total_reward:.1f}"
            
            if info_text:
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.axis('off')
            
            # Convert matplotlib figure to numpy array
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame_array = np.asarray(buf)
            frame_array = frame_array[:, :, :3]
            frames.append(frame_array)
            
            plt.close(fig)

            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(self, 'model'):
                    # For DQN
                    action = self.model(state_t).argmax(dim=1).item()
                else:
                    # For ActorCritic and PolicyGradient
                    action_probs = F.softmax(self.actor(state_t), dim=1)
                    action = action_probs.argmax(dim=1).item()

            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1
            total_reward += reward

        self.env.close()
        imageio.mimsave(filename, frames, fps=30)
        print(f"Saved GIF to {filename} (Total reward: {total_reward:.1f}, Steps: {step_count})")
    
    def plot_metrics(self, filename=None):
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        
        # Plot rewards
        rewards = np.array(self.metrics['rewards'])
        smoothed_rewards = gaussian_filter1d(rewards, sigma=2)
        ax1.plot(self.metrics['episodes'], rewards, alpha=0.3, label='Raw Reward', color="blue")
        ax1.plot(self.metrics['episodes'], smoothed_rewards, label='Smoothed Reward', color="blue")
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()

        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename, "rb"))


class DQNRunner(Runner):
    def __init__(self, env, model, optimizer, device, gamma=0.99, batch_size=32, epsilon=0.1):
        super().__init__(env, device, gamma)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        # Initialize target network
        self.target_model = type(model)(model.net[0].in_features, model.net[-1].out_features).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample(), None
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(state_t).argmax(dim=1).item(), None

    def train_step(self, target_update_freq=100):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        # Q(s,a)
        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # target: r + γ max_a' Q_target(s',a')
        with torch.no_grad():
            next_q = self.target_model(next_state).max(dim=1)[0]
            target_q = reward + self.gamma * next_q * (1 - done)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps % target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def run(self, n_episodes=1000, max_steps=1e4):
        episode = 0
        progress_bar = tqdm(desc="Training", total=max_steps)
        
        while episode < n_episodes and self.steps < max_steps:
            state, info = self.env.reset()
            episode_reward = 0
            done = False

            while self.steps < max_steps:
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.steps += 1
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                loss = self.train_step()
                progress_bar.update(1)

                if loss is not None:
                    self.metrics['losses'].append(loss)
                    self.metrics['steps'].append(self.steps)
                state = next_state
                episode_reward += reward

                if done: 
                    self.metrics['rewards'].append(episode_reward)
                    self.metrics['reward_steps'].append(self.steps)
                    self.metrics['episodes'].append(episode)
                    progress_bar.set_description(
                    f"Training: Episode {episode}, "
                    f"Reward: {self.metrics['rewards'][-1]}, "
                    f"Steps: {self.steps}, "
                    f"Loss: {np.mean(self.metrics['losses'][-100:]): .2f}"
                    )
                    episode += 1
                    break

        progress_bar.close()


class PolicyGradientRunner(Runner):
    def __init__(self, env, model, optimizer, device, gamma=0.99):
        super().__init__(env, device, gamma)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.rewards, self.log_probs = [], []
    
    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.model(state_t)
        action_probs = F.softmax(out, dim=1)
        C = Categorical(action_probs)
        action = C.sample()
        log_prob = C.log_prob(action)
        return action.item(), log_prob

    def train_step(self):
        returns = []
        R = 0
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(self.log_probs).squeeze()


        # baselining for stability (optional)
        returns = returns - returns.mean()

        loss = -(returns * log_probs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards, self.log_probs = [], []
        return loss.item()
    
    def run(self, n_episodes=1000, max_steps=1e4):
        episode = 0
        progress_bar = tqdm(desc="Training", total=max_steps)
        
        while episode < n_episodes and self.steps < max_steps:
            state, info = self.env.reset()
            done = False
            episode_reward = 0

            while self.steps < max_steps:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.steps += 1
                progress_bar.update(1)
                done = terminated or truncated
                episode_reward += reward
                self.rewards.append(reward)
                self.log_probs.append(log_prob)
                state = next_state

                if done:
                    loss = self.train_step()
                    self.metrics['losses'].append(loss)
                    self.metrics['steps'].append(self.steps)
                    self.metrics['rewards'].append(episode_reward)
                    self.metrics['reward_steps'].append(self.steps)
                    self.metrics['episodes'].append(episode)

                    progress_bar.set_description(
                        f"Training: Episode {episode}, "
                        f"Reward: {self.metrics['rewards'][-1]}, "
                        f"Steps: {self.steps}, "
                        f"Loss: {np.mean(self.metrics['losses'][-100:]): .2f}"
                    )
                    
                    episode += 1
                    break

        progress_bar.close()


class ActorCriticRunner(Runner):
    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer, device, gamma=0.99):
        super().__init__(env, device, gamma)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = F.softmax(self.actor(state_t), dim=1)
        C = Categorical(action_probs)
        action = C.sample()
        log_prob = C.log_prob(action)
        entropy = C.entropy()
        return action.item(), (log_prob, entropy)
    
    def train_step(self, state, action, reward, next_state, done, log_prob, entropy):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        log_prob = log_prob.squeeze()
        entropy = entropy.squeeze()
        state_value = self.critic(state).squeeze()
        next_state_value = self.critic(next_state).squeeze().detach() if not done else 0
        advantage = reward + self.gamma * next_state_value - state_value

        actor_loss = -(advantage.detach() * log_prob - 0.01 * entropy).mean()
        critic_loss = 0.01 * advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item()

    def run(self, n_episodes=1000, max_steps=1e4):
        episode = 0
        progress_bar = tqdm(desc="Training", total=max_steps)
        
        while episode < n_episodes and self.steps < max_steps:
            state, info = self.env.reset()
            done = False
            episode_reward = 0

            while self.steps < max_steps:
                action, (log_prob, entropy) = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.steps += 1
                progress_bar.update(1)
                done = terminated or truncated
                episode_reward += reward
                actor_loss, critic_loss = self.train_step(state, action, reward, next_state, done, log_prob, entropy)
                self.metrics['actor_losses'].append(actor_loss)
                self.metrics['critic_losses'].append(critic_loss)
                self.metrics['steps'].append(self.steps)

                state = next_state

                if done: 
                    self.metrics['rewards'].append(episode_reward)
                    self.metrics['reward_steps'].append(self.steps)
                    self.metrics['episodes'].append(episode)

                    progress_bar.set_description(
                        f"Training: Episode {episode}, "
                        f"Reward: {self.metrics['rewards'][-1]}, "
                        f"Steps: {self.steps}, "
                        f"Actor Loss: {np.mean(self.metrics['actor_losses'][-100:]): .2f}, "
                        f"Critic Loss: {np.mean(self.metrics['critic_losses'][-100:]): .2f}"
                    )
                    episode += 1
                    break
            

                    
