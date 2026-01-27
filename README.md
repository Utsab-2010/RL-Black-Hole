# Black Hole RL Project

This project implements the **Black Hole** board game as a Reinforcement Learning environment using **Gymnasium**, calculates strategies using **Self-Play DQN**, and includes a **Pygame** interface to play against the trained AI.

## 📂 Project Structure

- **`black_hole/`**: The Python package for the game environment.
  - `game.py`: Core game logic (Board string representation, Move validation, Scoring).
  - `env.py`: Gymnasium wrapper implementing the standard RL API.
  - `wrappers.py`: `SelfPlayWrapper` that handles the opponent's moves automatically, allowing single-agent training.
- **`train_dqn.py`**: The main training script using Deep Q-Networks (DQN) with Embedding Analysis.
- **`blackhole_test.py`**: Pygame script to play against the AI graphically.
- **`verify_env.py`**: Simple script to verify the Gym environment standard compliance.
- **`setup_instructions.md`**: Guide for initial environment setup.

## 🚀 Setup

1.  **Environment**:
    ```powershell
    conda activate IRL_env
    ```
2.  **Install Dependencies**:
    ```powershell
    pip install gymnasium numpy torch matplotlib pygame
    ```
3.  **Install the Game Package** (Run from project root):
    ```powershell
    pip install -e .
    ```

## 🧠 Training the AI

The agent learns via **Self-Play**. It plays against a frozen previous version of itself ("Opponent"). When the agent's win rate exceeds `75%`, effectively beating the opponent, the opponent is updated to the agent's current level.

To start training:
```powershell
python train_dqn.py
```

### Training Output
Artifacts are saved in `trained_models/BlackHole_DQN_v{X}/`:
- `model.pth`: The trained model weights.
- `training_log.png`: A plot showing **Average Reward** and **Win Rate** over time.
- `training_log.txt`: Hyperparameters and final stats.

**Key Hyperparameters (in `train_dqn.py`)**:
- `NUM_EPISODES`: Total episodes (Default: 30,000).
- `OPPONENT_UPDATE_MIN_EPISODES`: Minimum wait (1000 eps) before updating the opponent, ensuring stability.

## 🎮 Playing Against the AI

Once you have a trained model, test your skills against it!

**Run with latest model**:
```powershell
python blackhole_test.py
```

**Run with specific model**:
```powershell
python blackhole_test.py --model "trained_models/BlackHole_DQN_v1/model.pth"
```

### Game Controls
- **You are Player 1 (Red)**.
- **AI is Player 2 (Green)**.
- **Click** on any valid circle to place your tile.
- The game automatically handles the AI's turn.

## 🤖 Model Architecture

The model uses an **Embedding-based Q-Network**:
- **Inputs**: Integers representing the board state.
- **Embeddings**:
  - `Position`: Maps the 21 board spots to 4D vectors.
  - `Value`: Maps tile values (0-10) to 4D vectors.
  - `Player`: Maps player IDs (0-2) to 3D vectors.
- **Architecture**:
  - Embeddings -> Concatenate -> Flatten -> Linear(128) -> Linear(64) -> Linear(21 Actions).

This architecture allows the model to learn the topological relationships of the triangular board better than a simple flattened array.
