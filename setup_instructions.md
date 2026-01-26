# Black Hole RL Environment Setup

I have created the `black_hole` python package which implements the Black Hole game as a Gymnasium environment.

## File Structure
- `black_hole/`
    - `__init__.py`: Registers `BlackHole-v0`.
    - `game.py`: Core logic.
    - `env.py`: Gym wrapper.
- `verify_env.py`: Script to test the environment.

## Setup Instructions

1. **Activate your environment**:
   ```powershell
   conda activate IRL_env
   ```

2. **Install requirements** (if not already installed):
   You need `gymnasium` and `numpy`.
   ```powershell
   pip install gymnasium numpy
   ```

3. **Install the package**:
   Run this from the `d:/RL-games-kgts` directory (where `black_hole` folder is located):
   ```powershell
   pip install -e .
   ```
   *Note: Since I haven't created a setup.py, you might just want to append python path or run locally. The easiest way without setup.py is:*
   
   Just run the verify script directly, as it imports `black_hole` locally:
   ```powershell
   python verify_env.py
   ```

## Usage in Code
```python
import gymnasium as gym
import black_hole # Must import to register

env = gym.make("BlackHole-v0")
obs, info = env.reset()
```

## Game Rules Implemented
- **Board**: 21 spaces in a triangle.
- **Pieces**: Players 1 and 2 take turns placing 1, 1, 2, 2... 10, 10.
- **Scoring**: Rings around the last empty spot (Black Hole).
