# Black Hole RL Agent

This project implements the **Black Hole** board game (refer to `RULES.md`) as a Reinforcement Learning environment using **Gymnasium**, calculates strategies using **Self-Play DQN**, and includes a **Pygame** interface to play against the trained AI.

## 📂 Project Structure

- **`black_hole/`**: The Python package for the game environment.
  - `game.py`: Core game logic (Board representation, Move validation, Scoring).
  - `env.py`: Gymnasium wrapper implementing the standard RL API.
  - `wrappers.py`: `SelfPlayWrapper` that handles the opponent's moves automatically.
  - `model.py`: **Shared Model Architecture**. Contains the `QNetwork` (ResNet-18 + Sinusoidal Embeddings) and helper functions (`preprocess_obs`, `get_action_mask`) used by all scripts.
- **`train_dqn_vector.py`**: **(Recommended if GPU)** The high-performance vectorized training script using `gym.vector.SyncVectorEnv` to run 128+ games in parallel.
- **`train_dqn.py`**: The original single-threaded training script (reference/debugging).
- **`blackhole_test.py`**: Pygame script to play against the AI graphically.
- **`verify_env.py`**: Simple script to verify the Gym environment standard compliance.

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

The agent learns via **Self-Play**. It plays against a previous version of itself ("Opponent"). When the agent's win rate against the opponent exceeds `75%` **for 3 consecutive evaluations**, the opponent is promoted to the current agent's level. This ensures the new agent is consistently better, not just lucky.

### Option 1: Vectorized Training (Fastest) -> **Recommended**
Runs multiple environments in parallel to maximize GPU utilization.
```powershell
python train_dqn_vector.py
```
**Options**:
- `--resume "path/to/checkpoint.pth"`: Resume training from a saved state (restore episode & weights).
- `--load "path/to/model.pth"`: Start fresh training (Experiment 0) but initialized with trained weights (Fine-Tuning).
- `--stochastic`: Use **Top-21 Softmax Sampling** (Full Distribution) for exploration instead of Epsilon-Greedy. Evaluation uses **Top-5**. Good for robust training.

**Key Hyperparameters**:
- `NUM_ENVS`: Number of parallel games (Default: 512).
- `BATCH_SIZE`: Training batch size (Default: 2048).
- `NUM_CYCLIC_DECAY_CYCLES`: Number of times Epsilon resets (Cyclic Decay).

### Option 2: Single-Threaded Training
Good for debugging or understanding the flow.(use if no gpu)
```powershell
python train_dqn.py
```

### Option 3: Vectorized AlphaZero Training (Experimental)
Trains a dual-headed model (Policy + Value) using highly optimized, batched **MCTS Self-Play**.
```powershell
python train_alphazero.py
```
- **Batched Inference**: Extensively parallelized self-play. Passes multiple active MCTS tree queries through the network at once for extreme speedups.
- **Checkpoint Resuming**: Full support for continuing long training sessions without losing optimizer state (`--resume "/path/to/model.pth"`).
- **Fine-Tuning**: Start fresh training runs with pretrained weights (`--load "/path/to/model.pth"`).
- **Live Evaluating**: Halts training every `EVAL_INTERVAL` iterations to test greedy network policy performance (P1 & P2 pairs) against pure random moves.
- **Auto-plotting**: Saves and seamlessly appends to `training_plot.png` showing Loss, Self-Play MCTS Reward, and Eval vs Random Reward.

**Key Hyperparameters**:
- `TRAINING_ITERATIONS`: Total training loops (Self-Play -> Train).
- `SELF_PLAY_EPISODES`: Parallel games played per iteration to generate data.
- `MCTS_SIMS`: Simulations per move (Teacher strength, e.g., 50-200).
- `BATCH_SIZE`: Minibatch size (e.g., 256).
- `TRAINING_EPOCHS_PER_ITER`: Passes through the replay buffer per iteration.
- `EVAL_INTERVAL`: How often to test the raw network against a random baseline.

### Training Output
Artifacts are saved in `trained_models/BlackHole_DQN_v{X}/`:
- `model.pth`: The trained model weights + config.
- `checkpoint.pth`: Periodic checkpoint (resumable).
- `training_plot.png`: A plot showing **Average Reward** and **Win Rate** over time.
- `training_log.txt`: Hyperparameters and final stats.

## 🎮 Playing Against the AI

Once you have a trained model, you can play against it, watch two models fight, or run simulations.

### 1. You vs AI (Interactive)
The script automatically loads the latest model. By default, you are Player 1.
```powershell
python blackhole_test.py
```
**Options**:
- `--model "path/to/model.pth"`: Play against a specific model.
- `--player 2`: Play as Player 2 (Green) instead of Player 1 (Red).
- `--stochastic-p1`: Player 1 uses Top-3 Sampling instead of Argmax.
- `--stochastic-p2`: Player 2 uses Top-3 Sampling instead of Argmax.

### 2. AI vs AI (Watch Mode)
Watch two models battle it out.
```powershell
python blackhole_test.py --p1 "model_A.pth" --p2 "model_B.pth"
```

### 3. Headless Simulation (Benchmarking)
Quickly simulate 100 games between two models without graphics to get win rates.
```powershell
python blackhole_test.py --sim --p1 "model_A.pth" --p2 "model_B.pth" --num-games 100
```

### Game Controls
- **You are Player 1 (Red)** or **Player 2 (Green)**.
- **Click** on any valid circle to place your tile.
- The game automatically handles the AI's turn (with 0.3s delay).
- The game automatically handles the AI's turn.

## 🤖 Model Architecture (ResNet-18)

The model logic is centralized in `black_hole/model.py`.
- **Input**: 6x6x2 Grid (Plane 1 = Player 1, Plane 2 = Player 2).
- **Backbone**: **ResNet-18** style CNN blocks (32 -> 64 -> 128 -> 256 channels).
- **Time/Turn Encoding**: **Sinusoidal Embeddings** (Transformer-style) injected into the dense layer to tell the agent "how late" in the game it is.
- **Output**: 21 logits corresponding to board positions, masked for validity.

This architecture captures spatial relationships on the board much better than standard MLPs.
