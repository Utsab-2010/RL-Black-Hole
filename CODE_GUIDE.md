# Black Hole Codebase Guide

This document explains the core components, classes, and functions used in the project.

## 1. Game Core (`black_hole/game.py`)
The raw Python implementation of the game logic.

*   **`BlackHoleGame` Class**:
    *   `reset()`: Clears the board and resets tiles (1-10 per player).
    *   `_build_adjacency()`: Creates the graph of the triangular grid (who is next to whom).
    *   `make_move(position, value)`: Validates and executes a move. Returns `True` if game over.
    *   `calculate_score()`: **Critical**. Performs Breadth-First Search (BFS) from the empty spot (Black Hole) to define "Rings". Sums values in each ring to determine the winner based on "Lower Sum Wins".

## 2. Environment (`black_hole/env.py`)
Wraps the game in a Gymnasium interface for RL.

*   **`BlackHoleEnv` Class**:
    *   `step(action)`: Takes an integer (0-20), plays the move, checks for win/loss, and gives rewards.
        *   **Reward Scheme**: +1 for Win, -1 for Loss, 0 for Draw/Step.
    *   `observation_space`: Dictionary containing:
        *   `board`: (21, 2) array. Column 0 = Player ID, Column 1 = Tile Value.
        *   `current_player`: 1 or 2.

## 3. The Brain (`black_hole/model.py`)
The neural network and data processing helpers.

*   **`QNetwork` Class**:
    *   **Input**: 6x6x2 Grid (transformed from triangle).
    *   **Architecture**: ResNet-18 extraction -> Flatten -> **Concatenate Sinusoidal Embeddings** -> MLP -> Output.
    *   **Sinusoidal Embeddings**: Injects a concept of "Time" (Turn Number) into the network so it knows if it's early or late game.
*   **`get_action_mask(obs)`**: Returns a Boolean mask (True=Valid, False=Occupied) to prevent illegal moves.
*   **`preprocess_obs(obs)`**: Converts the raw dictionary observation into a Tensor for the GPU.

## 4. Vectorized Training (`train_dqn_vector.py`)
The main engine for training using 512+ parallel games.

*   **`train()`**: The main loop.
*   **`canonicalize_batch(obs, mask)` (Crucial)**:
    *   Logic: "If I am Player 2 (Green), flip the board so 1s become 2s and 2s become 1s."
    *   Purpose: Allows the Neural Network to always think "I am Player 1 (Red)". One model learns both sides.
*   **`evaluate_winrate()`**:
    *   Runs 20 games: **Agent (Pure Exploitation)** vs **Opponent (Pure Exploitation)**.
    *   Used to decide if the Agent is "smart enough" to replace the Opponent.

## 5. Play Script (`blackhole_test.py`)
The interface for Human vs AI.

*   **`get_ai_move(agent, obs)`**:
    *   Handles the entire inference pipeline: Canonicalization (Flipping) -> Preprocessing -> Inference -> Masking -> Argmax.
    *   Includes a `0.3s` artificial delay for better UX.

## Key Concepts
*   **Self-Play**: The Agent trains by playing against a frozen copy of itself (`opponent_model`).
*   **Cyclic Decay**: Epsilon (Randomness) drops linearly, then resets every `N` cycles. This prevents getting stuck in local optima.
*   **Batched Inference**: Using `SyncVectorEnv` to run 512 games at once allows the GPU to process 512 moves in a single matrix operation, speeding up training by ~15x.
