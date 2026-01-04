"""
Multi-Agent Reinforcement Learning Environment
Author: Dhanya Sri Cherukuri
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Dict, Optional

class MultiAgentEnv(gym.Env):
    """
    A flexible multi-agent environment supporting cooperative and competitive scenarios.
    
    Features:
    - Configurable number of agents (2-8)
    - Grid-based world with continuous positions
    - Customizable observation and action spaces
    - Support for communication between agents
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        n_agents: int = 4,
        grid_size: int = 20,
        max_steps: int = 100,
        reward_type: str = 'cooperative',
        observation_radius: float = 5.0,
        communication: bool = False,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the multi-agent environment.
        
        Args:
            n_agents: Number of agents in the environment
            grid_size: Size of the grid world
            max_steps: Maximum steps per episode
            reward_type: 'cooperative', 'competitive', or 'mixed'
            observation_radius: How far each agent can observe
            communication: Whether agents can communicate
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.observation_radius = observation_radius
        self.communication = communication
        self.render_mode = render_mode
        
        # Define observation space for each agent
        # [own_pos(2), own_vel(2), relative_positions(n_agents-1 * 2), goal_pos(2)]
        obs_dim = 2 + 2 + (n_agents - 1) * 2 + 2
        if communication:
            obs_dim += (n_agents - 1) * 4  # Message from other agents
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space: 5 discrete actions (4 directions + stay)
        self.action_space = spaces.Discrete(5)
        
        # Action mappings
        self.action_to_direction = {
            0: np.array([0, 0]),   # Stay
            1: np.array([0, 1]),   # Up
            2: np.array([0, -1]),  # Down
            3: np.array([-1, 0]),  # Left
            4: np.array([1, 0])    # Right
        }
        
        # Initialize state
        self.agents_pos = None
        self.agents_vel = None
        self.goal_pos = None
        self.current_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[List[np.ndarray], dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Random initial positions for agents
        self.agents_pos = np.random.uniform(0, self.grid_size, (self.n_agents, 2))
        self.agents_vel = np.zeros((self.n_agents, 2))
        
        # Random goal position
        self.goal_pos = np.random.uniform(0, self.grid_size, 2)
        
        self.current_step = 0
        
        observations = self._get_observations()
        info = {'step': 0}
        
        return observations, info
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        assert len(actions) == self.n_agents, f"Expected {self.n_agents} actions, got {len(actions)}"
        
        # Update agent positions based on actions
        for i, action in enumerate(actions):
            direction = self.action_to_direction[action]
            self.agents_vel[i] = direction * 0.5
            self.agents_pos[i] += self.agents_vel[i]
            
            # Clip to grid boundaries
            self.agents_pos[i] = np.clip(self.agents_pos[i], 0, self.grid_size)
        
        self.current_step += 1
        
        # Calculate rewards based on reward type
        rewards = self._calculate_rewards()
        
        # Check termination conditions
        terminated = self._check_goal_reached()
        truncated = self.current_step >= self.max_steps
        
        observations = self._get_observations()
        info = {
            'step': self.current_step,
            'goal_reached': terminated,
            'agents_pos': self.agents_pos.copy()
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _get_observations(self) -> List[np.ndarray]:
        """Generate observations for each agent."""
        observations = []
        
        for i in range(self.n_agents):
            obs = []
            
            # Own position (normalized)
            obs.extend(self.agents_pos[i] / self.grid_size)
            
            # Own velocity
            obs.extend(self.agents_vel[i])
            
            # Relative positions of other agents
            for j in range(self.n_agents):
                if i != j:
                    rel_pos = self.agents_pos[j] - self.agents_pos[i]
                    # Mask if outside observation radius
                    dist = np.linalg.norm(rel_pos)
                    if dist > self.observation_radius:
                        rel_pos = np.zeros(2)
                    obs.extend(rel_pos / self.grid_size)
            
            # Goal position (relative)
            goal_rel = self.goal_pos - self.agents_pos[i]
            obs.extend(goal_rel / self.grid_size)
            
            # Communication messages (placeholder)
            if self.communication:
                for j in range(self.n_agents):
                    if i != j:
                        obs.extend([0, 0, 0, 0])  # Empty message
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def _calculate_rewards(self) -> List[float]:
        """Calculate rewards based on the reward type."""
        rewards = []
        
        # Distance to goal for each agent
        distances = [np.linalg.norm(self.agents_pos[i] - self.goal_pos) 
                    for i in range(self.n_agents)]
        
        if self.reward_type == 'cooperative':
            # Team reward based on average distance to goal
            avg_dist = np.mean(distances)
            team_reward = -avg_dist / self.grid_size
            rewards = [team_reward] * self.n_agents
            
        elif self.reward_type == 'competitive':
            # Individual rewards, closer to goal = higher reward
            max_dist = np.sqrt(2) * self.grid_size
            for dist in distances:
                rewards.append((max_dist - dist) / max_dist)
                
        else:  # mixed
            avg_dist = np.mean(distances)
            team_reward = -avg_dist / self.grid_size
            for i, dist in enumerate(distances):
                individual = -dist / self.grid_size
                rewards.append(0.5 * team_reward + 0.5 * individual)
        
        return rewards
    
    def _check_goal_reached(self) -> bool:
        """Check if any agent reached the goal."""
        for pos in self.agents_pos:
            if np.linalg.norm(pos - self.goal_pos) < 1.0:
                return True
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render to screen using matplotlib."""
        import matplotlib.pyplot as plt
        
        plt.clf()
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        
        # Plot agents
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_agents))
        for i, pos in enumerate(self.agents_pos):
            plt.scatter(pos[0], pos[1], c=[colors[i]], s=100, label=f'Agent {i}')
        
        # Plot goal
        plt.scatter(self.goal_pos[0], self.goal_pos[1], c='gold', s=200, marker='*', label='Goal')
        
        plt.legend()
        plt.title(f'Step: {self.current_step}')
        plt.pause(0.01)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array."""
        img_size = 256
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        scale = img_size / self.grid_size
        
        # Draw goal
        goal_px = (self.goal_pos * scale).astype(int)
        cv2_available = False
        try:
            import cv2
            cv2_available = True
            cv2.circle(img, tuple(goal_px), 10, (255, 215, 0), -1)
        except ImportError:
            pass
        
        # Draw agents
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
        
        if cv2_available:
            for i, pos in enumerate(self.agents_pos):
                agent_px = (pos * scale).astype(int)
                color = colors[i % len(colors)]
                cv2.circle(img, tuple(agent_px), 8, color, -1)
        
        return img
    
    def close(self):
        """Clean up resources."""
        pass


class CooperativeEnv(MultiAgentEnv):
    """Cooperative variant of the multi-agent environment."""
    
    def __init__(self, **kwargs):
        kwargs['reward_type'] = 'cooperative'
        super().__init__(**kwargs)


class CompetitiveEnv(MultiAgentEnv):
    """Competitive variant of the multi-agent environment."""
    
    def __init__(self, **kwargs):
        kwargs['reward_type'] = 'competitive'
        super().__init__(**kwargs)


# Register environments with Gymnasium
def register_envs():
    """Register custom environments."""
    gym.register(
        id='MultiAgentEnv-v0',
        entry_point='envs.multi_agent_env:MultiAgentEnv',
    )
    gym.register(
        id='CooperativeEnv-v0',
        entry_point='envs.multi_agent_env:CooperativeEnv',
    )
    gym.register(
        id='CompetitiveEnv-v0',
        entry_point='envs.multi_agent_env:CompetitiveEnv',
    )


if __name__ == '__main__':
    # Test the environment
    env = MultiAgentEnv(n_agents=4, render_mode='human')
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observations shape: {[o.shape for o in obs]}")
    
    for step in range(50):
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break
    
    env.close()
