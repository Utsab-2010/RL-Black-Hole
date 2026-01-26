# Black Hole RL Environment & Training

This repository contains the Black Hole game implemented as a custom Gymnasium environment, along with a Self-Play DQN training script.

## Installation

1. **Activate Environment**:
   ```powershell
   conda activate IRL_env
   ```

2. **Install Dependencies**:
   ```powershell
   pip install gymnasium numpy torch matplotlib
   ```

3. **Install the Game Package**:
   ```powershell
   pip install -e .
   ```

## Running the Training

To start the self-play training loop:

```powershell
python train_dqn.py
```

### What the Script Does
- **Self-Play**: The agent plays against a copy of itself ("Opponent").
- **Promotions**: Every 100 episodes, the agent is evaluated against the Opponent. If the agent's win rate exceeds 55%, the Opponent is updated to the current agent's weights. This creates a curriculum of increasing difficulty.
- **Model**: Uses a simple 3-layer MLP (DQN) with PyTorch.
- **Logging**:
    - Print output shows episode progress and win rates.
    - `training_log.png` is saved at the end, showing Loss and Win Rate curves.
    - `black_hole_dqn.pth` is saved as the final model checkpoint.

## Usage in Custom Code

```python
import gymnasium as gym
import black_hole
from black_hole.wrappers import SelfPlayWrapper

# Standard Env
env = gym.make("BlackHole-v0")

# Self-Play Wrappper
# opponent_policy can be any callable that takes an observation and returns an action
env = SelfPlayWrapper(env, opponent_policy=my_policy_function)
```
