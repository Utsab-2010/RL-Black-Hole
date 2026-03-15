# Black Hole RL Agent

This project implements the **Black Hole** board game (refer to `RULES.md`) as a Reinforcement Learning environment using **Gymnasium**. It supports both **Self-Play DQN** and **AlphaZero MCTS** training, with a **Pygame** interface for interactive play.

## 📂 Project Structure

- **`black_hole/`**: The Python package for the game environment.
  - `game.py`: Core game logic. Parametrized by `layers` (default: **9 layers, 45 hexes**).
  - `env.py`: Gymnasium wrapper. Action/observation spaces scale dynamically with the board.
  - `wrappers.py`: `SelfPlayWrapper` that handles the opponent's moves automatically.
  - `model.py`: **Shared Model Architecture**. Contains `QNetwork` (DQN) and `AlphaBH` (AlphaZero dual-head). Both take an `LxL` grid input and scale output heads to `num_hexes` automatically.
  - `mcts.py`: AlphaMCTS implementation with batched tree search.
- **`train_dqn_vector.py`**: **(DQN)** High-performance vectorized training using `gym.vector.SyncVectorEnv`.
- **`train_dqn.py`**: **(DQN)** Single-threaded reference training script.
- **`train_alphazero.py`**: **(AlphaZero)** MCTS self-play training with batched inference.
- **`blackhole_test.py`**: Pygame interface for the DQN model.
- **`blackhole_test_alphazero.py`**: Pygame interface for the AlphaZero model (uses MCTS during play).

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

### Option 1: AlphaZero Training → **Recommended**
Trains a dual-headed Policy + Value network via MCTS self-play.
```powershell
python train_alphazero.py
python train_alphazero.py --resume "trained_models/AlphaZero_DQN_run1/model.pth"
python train_alphazero.py --load "trained_models/AlphaZero_DQN_run1/model.pth"
```
**Arguments**:
- `--resume`: Resume training fully (restores optimizer, scheduler, iteration count).
- `--load`: Start fresh training initialized with pretrained weights (fine-tuning).

**Key Hyperparameters** (edit at top of file):
| Parameter | Default | Description |
|---|---|---|
| `TRAINING_ITERATIONS` | 5000 | Total self-play → train loops |
| `SELF_PLAY_EPISODES` | 20 | Parallel games per iteration |
| `MCTS_SIMS` | 10 | MCTS simulations per move (self-play) |
| `MCTS_SIMS_EVAL` | 8 | MCTS simulations per move (evaluation) |
| `BATCH_SIZE` | 512 | Training minibatch size |
| `TRAINING_EPOCHS_PER_ITER` | 8 | Buffer passes per training step |
| `EVAL_INTERVAL` | 10 | Iterations between evaluations |
| `EVAL_GAMES` | 5 | Games per role vs random |
| `EVAL_UPGRADE_THRESHOLD` | 0.6 | If vs-random score ≥ this, also eval vs last checkpoint |
| `EVAL_CHECKPOINT_GAMES` | 5 | Games per role vs checkpoint |
| `SCHEDULER_STEP` | 50 | Iterations between LR decay steps |

**Evaluation logic**: Every `EVAL_INTERVAL` iterations, the agent plays vs a random opponent using MCTS. If that score ≥ `EVAL_UPGRADE_THRESHOLD`, a second harder eval runs against the last saved `model.pth`. The harder score is reported in the training plot.

**Output** saved to `trained_models/AlphaZero_DQN_run{N}/`:
- `model.pth`: Weights + optimizer + scheduler state (resumable).
- `training_plot.png`: Loss, self-play reward, eval reward curves.
- `training.log`: Per-iteration CSV log.

---

### Option 2: Vectorized DQN Training
```powershell
python train_dqn_vector.py
python train_dqn_vector.py --resume "path/to/checkpoint.pth"
python train_dqn_vector.py --load "path/to/model.pth"
python train_dqn_vector.py --stochastic
```
**Arguments**:
- `--resume / --load`: Same as AlphaZero script.
- `--stochastic`: Use Top-N Softmax Sampling for exploration instead of Epsilon-Greedy.

**Key Hyperparameters**:
- `NUM_ENVS`: Parallel environments (Default: 512).
- `EPSILON_START / EPSILON_END`: Exploration schedule.
- `SELF_PLAY_UPDATE_THRESHOLD`: Win rate needed to promote the opponent (Default: 55%).

---

### Option 3: Single-Threaded DQN (Debug / No GPU)
```powershell
python train_dqn.py
```

---

## 🎮 Playing Against the AI

### AlphaZero Test Script (MCTS-powered)
Always uses MCTS for decision-making. Launches **fullscreen** Pygame.
```powershell
# Human vs AI (auto-loads latest model)
python blackhole_test_alphazero.py

# Human vs specific model as Player 2
python blackhole_test_alphazero.py --model "path/to/model.pth" --player 2

# AI vs AI (watch mode)
python blackhole_test_alphazero.py --p1 "model_A.pth" --p2 "model_B.pth"

# Headless simulation (no graphics)
python blackhole_test_alphazero.py --sim --p1 "model_A.pth" --p2 "model_B.pth" --num-games 50

# Control MCTS search depth
python blackhole_test_alphazero.py --eval-sims 30
```
**All arguments**:
| Argument | Default | Description |
|---|---|---|
| `--model` | auto | Path to model for Human vs AI mode |
| `--p1` | — | Path to Player 1 model (AI vs AI) |
| `--p2` | — | Path to Player 2 model (AI vs AI) |
| `--player` | 1 | Your player ID in Human vs AI (1 or 2) |
| `--eval-sims` | 10 | MCTS simulation count per move |
| `--sim` | off | Run headless simulation (no window) |
| `--num-games` | 100 | Number of games in `--sim` mode |
| `--cpu` | off | Force CPU inference |

---

### DQN Test Script
```powershell
python blackhole_test.py
python blackhole_test.py --model "path/to/model.pth" --player 2
python blackhole_test.py --p1 "model_A.pth" --p2 "model_B.pth"
python blackhole_test.py --sim --p1 "model_A.pth" --p2 "model_B.pth" --num-games 100
python blackhole_test.py --stochastic-p1 --stochastic-p2
```
**Arguments**: Same as AlphaZero script except no `--eval-sims`; also adds `--stochastic-p1` / `--stochastic-p2` for top-K sampling.

---

### Game Controls
- **Click** on any valid hex to place your tile.
- The AI moves automatically after a short delay.
- Press **Escape** or close the window to exit.

---

## 🤖 Model Architecture

The model logic is centralized in `black_hole/model.py`. Both models (`QNetwork` for DQN, `AlphaBH` for AlphaZero) share the same input encoding and backbone.

**Input**: `L×L×1` Grid (Single Channel), where `L = layers` (default 9 → 9×9 grid).
- **Tile Weights**: Each occupied hex stores a signed, normalized tile value.  
  Player 1's tile `v` → `+v / tiles_per_player`. Player 2's tile `v` → `-v / tiles_per_player`.
- **Empty**: `0.0`
- **Void Cells** (outside the triangle): `-1000.0` (strongly penalized, effectively masked out).

**Backbone**: ResNet-18–style CNN (32 → 64 → 128 → 256 channels, stride-2 downscaling).

**Heads**:
- `QNetwork`: Single Q-value head → `num_hexes` logits (masked softmax).
- `AlphaBH`: **Policy head** → `num_hexes` logits + **Value head** → scalar `[-1, 1]` (Tanh).

**Turn inference**: Because exact tile values (1–22) are encoded directly in the board, the network naturally infers the turn number without any external time embeddings.
