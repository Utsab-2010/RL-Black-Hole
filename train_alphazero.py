
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from black_hole.game import BlackHoleGame
from black_hole.model import AlphaBH, preprocess_obs
from black_hole.mcts import AlphaMCTS

# Hyperparameters
NUM_ITERATIONS = 50   # Total loops
SELF_PLAY_EPISODES = 5 # Games per iteration
MCTS_SIMS = 50        # Simulations per move (Low for speed, increase later)
BATCH_SIZE = 64
EPOCHS = 4            # Training epochs per iteration
LEARNING_RATE = 0.001
BUFFER_SIZE = 5000
PLOT_WINDOW = 5

os.makedirs("models_alphazero", exist_ok=True)
LOG_FILE = "models_alphazero/training.log"

def setup_logging():
    with open(LOG_FILE, "w") as f:
        f.write("Iteration,Loss,AvgReward\n")

def log_metrics(iteration, loss, reward):
    with open(LOG_FILE, "a") as f:
        f.write(f"{iteration},{loss:.4f},{reward:.2f}\n")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)

def self_play(agent, mcts, episodes, buffer):
    rewards = []
    
    agent.eval()
    
    for ep in range(episodes):
        game = BlackHoleGame()
        states = []
        policies = []
        
        while not game.check_game_over():
            # Run MCTS
            # Use temp=1 for early game, temp=0 for late? Or always 1 for training?
            # AlphaZero uses 1 for first 30 moves, then 0.
            # Our game is short (20 moves). Use 1.
            
            mcts_probs = mcts.run(game, num_simulations=MCTS_SIMS, temperature=1.0)
            
            # Store state for training
            # Need to store canonical board relative to current player?
            # Creating canonical input helps network generalize.
            
            obs = {
               "board": game.board.copy(), 
               "current_tile": (game.tiles_placed // 2) + 1
            }
            # Canonicalize Observation for Storage
            # If P2 to move, flip board so network learns "My pieces are 1, Opponent is 2"
            if game.current_player == 2:
                 board = obs["board"]
                 p1 = (board[:, 0] == 1)
                 p2 = (board[:, 0] == 2)
                 board[p1, 0] = 2
                 board[p2, 0] = 1
                 obs["board"] = board
            
            states.append(obs)
            policies.append(mcts_probs)
            
            # Sample action from policy
            action = np.random.choice(len(mcts_probs), p=mcts_probs)
            
            # Make move
            tile_val = (game.tiles_placed // 2) + 1
            if tile_val > 10: tile_val = 10
            game.make_move(action, tile_val)
            
        # Game Over
        winner, _ = game.calculate_score()
        # Reward relative to Player 1?
        # Standard AlphaZero: Value z is outcome from perspective of current player AT THAT STEP.
        
        # If P1 wins (Winner=1):
        # Steps where Player=1 -> z=1
        # Steps where Player=2 -> z=-1
        
        # If P2 wins (Winner=2):
        # Steps where Player=1 -> z=-1
        # Steps where Player=2 -> z=1
        
        # Tie -> z=0
        
        if winner == 1:
            game_value = 1.0
        elif winner == 2:
            game_value = -1.0
        else:
            game_value = 0.0
            
        # Backfill buffer
        # We need to know who was 'current_player' for each stored state?
        # Since moves alternate: P1, P2, P1...
        # Index 0: P1, Index 1: P2...
        
        curr_player = 1
        for i in range(len(states)):
            val = game_value if curr_player == 1 else -game_value
            buffer.push(states[i], policies[i], val)
            curr_player = 3 - curr_player # 1->2, 2->1
            
        rewards.append(game_value) # P1 perspective reward
        print(f"Episode {ep+1}/{episodes} | Winner: {winner}")

    return np.mean(rewards)

def train(agent, optimizer, buffer, device):
    agent.train()
    running_loss = 0.0
    
    for _ in range(EPOCHS):
        if len(buffer) < BATCH_SIZE: break
        
        states, target_pis, target_vs = buffer.sample(BATCH_SIZE)
        
        # Convert to Tensor
        # Preprocess batch expects dict of lists...
        # But here we have list of dicts.
        # We need to stack them for `preprocess_batch`.
        
        batch_obs = {
            "board": np.stack([s["board"] for s in states]),
            "current_tile": np.array([s["current_tile"] for s in states])
        }
        
        input_tensor = preprocess_obs(batch_obs, device) # Wait, preprocess_obs is for single?
        # We should use batch version or ensure logic works.
        # Let's use custom collation here.
        
        boards = torch.tensor(batch_obs["board"], dtype=torch.float32, device=device)
        tiles = torch.tensor(batch_obs["current_tile"], dtype=torch.float32, device=device).unsqueeze(1)
        boards_flat = boards.view(BATCH_SIZE, -1)
        input_tensor = torch.cat([boards_flat, tiles], dim=1)
        
        target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32, device=device)
        target_vs = torch.tensor(np.array(target_vs), dtype=torch.float32, device=device).unsqueeze(1)
        
        # Forward
        out_pi, out_v = agent(input_tensor)
        
        # Loss
        # Value: MSE
        l_v = nn.MSELoss()(out_v, target_vs)
        
        # Policy: Cross Entropy
        # out_pi are logits. target_pis are probs.
        # CE = - sum(target * log_softmax(logits))
        log_probs = torch.log_softmax(out_pi, dim=1)
        l_pi = -(target_pis * log_probs).sum(dim=1).mean()
        
        loss = l_v + l_pi
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / EPOCHS

def moving_average(data, window_size=5):
    if len(data) < window_size:
        return np.mean(data) if data else 0
    return np.mean(data[-window_size:])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = AlphaBH().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    mcts = AlphaMCTS(agent, device)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    setup_logging()
    
    # Plotting Data
    train_losses = []
    avg_rewards = []
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    for iteration in range(NUM_ITERATIONS):
        print(f"--- Iteration {iteration+1} ---")
        
        # 1. Self Play
        print("Self Playing...")
        avg_reward = self_play(agent, mcts, SELF_PLAY_EPISODES, buffer)
        avg_rewards.append(avg_reward)
        
        # 2. Train
        print(f"Training (Buffer: {len(buffer)})...")
        if len(buffer) >= BATCH_SIZE:
            loss = train(agent, optimizer, buffer, device)
            train_losses.append(loss)
            
            # Log
            log_metrics(iteration, loss, avg_reward)
            print(f"Loss: {loss:.4f} | Avg Reward (P1): {avg_reward:.2f}")
            
            # Save Model
            torch.save(agent.state_dict(), f"models_alphazero/model_{iteration}.pth")
            
        else:
            print("Skipping training (buffer too small)")
            train_losses.append(0)

        # 3. Update Plots
        ax1.clear()
        ax2.clear()
        
        ax1.set_title("Training Loss")
        ax1.plot(train_losses, label="Loss")
        if len(train_losses) >= PLOT_WINDOW:
            ma_loss = [np.mean(train_losses[max(0, i-PLOT_WINDOW+1):i+1]) for i in range(len(train_losses))]
            ax1.plot(ma_loss, label=f"MA-{PLOT_WINDOW}", linestyle="--")
        ax1.legend()
        
        ax2.set_title("Avg Reward (Win Rate Proxy)")
        ax2.plot(avg_rewards, label="Reward")
        if len(avg_rewards) >= PLOT_WINDOW:
            ma_rew = [np.mean(avg_rewards[max(0, i-PLOT_WINDOW+1):i+1]) for i in range(len(avg_rewards))]
            ax2.plot(ma_rew, label=f"MA-{PLOT_WINDOW}", linestyle="--")
        ax2.legend()
                    
        plt.pause(0.1)
        plt.savefig("models_alphazero/training_plot.png")

if __name__ == "__main__":
    main()
