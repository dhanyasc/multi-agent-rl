# Multi-Agent Reinforcement Learning Environment

A custom multi-agent environment for training cooperative and competitive reinforcement learning agents using OpenAI Gym and PyTorch.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gym](https://img.shields.io/badge/OpenAI_Gym-0.26+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project implements a flexible multi-agent reinforcement learning (MARL) environment that supports both cooperative and competitive scenarios. The environment is designed to study agent coordination, strategy learning, and emergent behaviors in multi-agent systems.

## ✨ Features

- **Custom Multi-Agent Environment**: Gym-compatible environment supporting 2-8 agents
- **Flexible Agent Types**: Support for cooperative, competitive, and mixed scenarios
- **Observation Space Design**: Configurable observation spaces for agent coordination
- **Action Space Design**: Discrete and continuous action space support
- **Training Algorithms**: Implemented MADDPG, QMIX, and Independent PPO
- **Visualization**: Real-time rendering and training metrics dashboard

## 🏗️ Architecture

```
multi-agent-rl/
├── envs/
│   ├── __init__.py
│   ├── multi_agent_env.py      # Core environment
│   ├── cooperative_env.py       # Cooperative scenarios
│   └── competitive_env.py       # Competitive scenarios
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Base agent class
│   ├── maddpg_agent.py         # MADDPG implementation
│   ├── qmix_agent.py           # QMIX implementation
│   └── ppo_agent.py            # Independent PPO
├── utils/
│   ├── replay_buffer.py        # Experience replay
│   ├── networks.py             # Neural network architectures
│   └── logger.py               # Training logger
├── configs/
│   └── default_config.yaml     # Configuration file
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
└── requirements.txt
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/dhanyasri/multi-agent-rl.git
cd multi-agent-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

```
torch>=2.0.0
gymnasium>=0.26.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.12.0
pyyaml>=6.0
```

## 💻 Usage

### Training Agents

```python
from envs import MultiAgentEnv
from agents import MADDPGAgent

# Create environment
env = MultiAgentEnv(n_agents=4, scenario='cooperative')

# Initialize agents
agents = [MADDPGAgent(env.observation_space, env.action_space) for _ in range(4)]

# Training loop
for episode in range(10000):
    obs = env.reset()
    done = False
    
    while not done:
        actions = [agent.select_action(o) for agent, o in zip(agents, obs)]
        next_obs, rewards, done, info = env.step(actions)
        
        # Store experience and update
        for i, agent in enumerate(agents):
            agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], done)
            agent.update()
        
        obs = next_obs
```

### Custom Environment Configuration

```python
env_config = {
    'n_agents': 4,
    'grid_size': 20,
    'max_steps': 100,
    'reward_type': 'cooperative',  # 'cooperative', 'competitive', 'mixed'
    'observation_radius': 5,
    'communication': True
}

env = MultiAgentEnv(**env_config)
```

## 📊 Results

| Algorithm | Cooperative Score | Competitive Win Rate | Training Time |
|-----------|------------------|---------------------|---------------|
| MADDPG    | 85.3 ± 4.2       | 62.1%               | 4.5 hrs       |
| QMIX      | 91.7 ± 3.1       | 58.4%               | 3.8 hrs       |
| Ind. PPO  | 72.4 ± 6.8       | 51.2%               | 2.1 hrs       |

## 🔬 Key Components

### Observation Space
- Agent position and velocity
- Relative positions of other agents
- Goal/target information
- Communication messages (optional)

### Action Space
- Movement: 4 discrete directions + stay
- Communication: Optional message passing
- Special actions: Scenario-specific

### Reward Structure
- **Cooperative**: Team-based rewards for goal completion
- **Competitive**: Individual rewards, zero-sum scenarios
- **Mixed**: Combination with configurable weights

## 📈 Training Visualization

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- [QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485)
- [OpenAI Gym Documentation](https://gymnasium.farama.org/)

## 👤 Author

**Dhanya Sri Cherukuri**
- GitHub: [@dhanyasri](https://github.com/dhanyasri)
- LinkedIn: [dhanyasri](https://linkedin.com/in/dhanyasri)
- Email: dhanyasricherukuri.03@gmail.com
