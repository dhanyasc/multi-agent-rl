"""
MADDPG Agent Implementation
Multi-Agent Deep Deterministic Policy Gradient
Author: Dhanya Sri Cherukuri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from collections import deque
import random


class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    """Centralized critic network for MADDPG."""
    
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for multi-agent training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs: List[np.ndarray], actions: List[int], 
             rewards: List[float], next_obs: List[np.ndarray], done: bool):
        self.buffer.append((obs, actions, rewards, next_obs, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient Agent.
    
    Implements centralized training with decentralized execution (CTDE).
    Each agent has its own actor but shares a centralized critic.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        hidden_dim: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize MADDPG agent.
        
        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            hidden_dim: Hidden layer dimension
            device: Device to use for training
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Create actors for each agent
        self.actors = [Actor(obs_dim, action_dim, hidden_dim).to(device) 
                      for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim, action_dim, hidden_dim).to(device) 
                             for _ in range(n_agents)]
        
        # Create centralized critics for each agent
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        self.critics = [Critic(total_obs_dim, total_action_dim, hidden_dim).to(device) 
                       for _ in range(n_agents)]
        self.target_critics = [Critic(total_obs_dim, total_action_dim, hidden_dim).to(device) 
                              for _ in range(n_agents)]
        
        # Initialize target networks
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) 
                                for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr_critic) 
                                 for critic in self.critics]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
    def select_actions(self, observations: List[np.ndarray], 
                       explore: bool = True) -> List[int]:
        """
        Select actions for all agents.
        
        Args:
            observations: List of observations for each agent
            explore: Whether to add exploration noise
            
        Returns:
            List of actions for each agent
        """
        actions = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs = self.actors[i](obs_tensor)
            
            if explore:
                # Sample from probability distribution
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Take greedy action
                action = action_probs.argmax().item()
            
            actions.append(action)
        
        return actions
    
    def store_transition(self, obs: List[np.ndarray], actions: List[int],
                        rewards: List[float], next_obs: List[np.ndarray], done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(obs, actions, rewards, next_obs, done)
    
    def update(self, batch_size: int = 256) -> dict:
        """
        Update all agents using a batch from replay buffer.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        obs_batch = [torch.FloatTensor(np.array([o[i] for o in obs_batch])).to(self.device) 
                    for i in range(self.n_agents)]
        next_obs_batch = [torch.FloatTensor(np.array([o[i] for o in next_obs_batch])).to(self.device) 
                        for i in range(self.n_agents)]
        action_batch = [torch.LongTensor([a[i] for a in action_batch]).to(self.device) 
                       for i in range(self.n_agents)]
        reward_batch = [torch.FloatTensor([r[i] for r in reward_batch]).to(self.device) 
                       for i in range(self.n_agents)]
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Concatenate all observations and actions for centralized critic
        all_obs = torch.cat(obs_batch, dim=-1)
        all_next_obs = torch.cat(next_obs_batch, dim=-1)
        
        # One-hot encode actions
        all_actions = torch.cat([F.one_hot(a, self.action_dim).float() 
                                for a in action_batch], dim=-1)
        
        metrics = {'actor_loss': [], 'critic_loss': []}
        
        # Update each agent
        for i in range(self.n_agents):
            # Get target actions for next state
            target_actions = []
            for j in range(self.n_agents):
                with torch.no_grad():
                    target_action = self.target_actors[j](next_obs_batch[j])
                target_actions.append(target_action)
            all_target_actions = torch.cat(target_actions, dim=-1)
            
            # Compute target Q-value
            with torch.no_grad():
                target_q = self.target_critics[i](all_next_obs, all_target_actions).squeeze()
                target_value = reward_batch[i] + self.gamma * (1 - done_batch) * target_q
            
            # Update critic
            current_q = self.critics[i](all_obs, all_actions).squeeze()
            critic_loss = F.mse_loss(current_q, target_value)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            
            # Update actor
            current_actions = []
            for j in range(self.n_agents):
                if j == i:
                    current_actions.append(self.actors[j](obs_batch[j]))
                else:
                    current_actions.append(F.one_hot(action_batch[j], self.action_dim).float())
            all_current_actions = torch.cat(current_actions, dim=-1)
            
            actor_loss = -self.critics[i](all_obs, all_current_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            
            metrics['actor_loss'].append(actor_loss.item())
            metrics['critic_loss'].append(critic_loss.item())
        
        # Soft update target networks
        self._soft_update()
        
        return metrics
    
    def _soft_update(self):
        """Soft update target networks."""
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                          self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critics[i].parameters(), 
                                          self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, state_dict in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(state_dict)
            self.target_actors[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['critics']):
            self.critics[i].load_state_dict(state_dict)
            self.target_critics[i].load_state_dict(state_dict)


if __name__ == '__main__':
    # Test MADDPG agent
    n_agents = 4
    obs_dim = 10
    action_dim = 5
    
    agent = MADDPGAgent(n_agents, obs_dim, action_dim)
    
    # Simulate some transitions
    for _ in range(1000):
        obs = [np.random.randn(obs_dim) for _ in range(n_agents)]
        actions = agent.select_actions(obs)
        rewards = [np.random.randn() for _ in range(n_agents)]
        next_obs = [np.random.randn(obs_dim) for _ in range(n_agents)]
        done = np.random.rand() < 0.01
        
        agent.store_transition(obs, actions, rewards, next_obs, done)
    
    # Update agent
    for _ in range(10):
        metrics = agent.update()
        if metrics:
            print(f"Actor loss: {np.mean(metrics['actor_loss']):.4f}, "
                  f"Critic loss: {np.mean(metrics['critic_loss']):.4f}")
    
    print("MADDPG agent test passed!")
