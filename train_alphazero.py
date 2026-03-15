
import os
import sys
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Headless backend — no GUI window, no freeze
import matplotlib.pyplot as plt
import math
from collections import deque
from black_hole.game import BlackHoleGame
from black_hole.model import AlphaBH, preprocess_obs, preprocess_batch
from black_hole.mcts import AlphaMCTS

# Hyperparameters
TRAINING_ITERATIONS = 5000       # Total training loops (Self-Play -> Train)
SELF_PLAY_EPISODES = 40         # Games played(parallelly) per iteration to generate data
MCTS_SIMS = 15                 # MCTS simulations per move (Teacher strength)
MCTS_SIMS_EVAL = 8            # MCTS simulations during eval vs random
BATCH_SIZE = 512                # Minibatch size for training
TRAINING_EPOCHS_PER_ITER = 8   # Passes through the buffer per iteration
LEARNING_RATE = 0.2     # Peak LR at start of first cycle
LR_MIN = 1e-5             # Floor LR (never goes below this)
LR_CYCLE_LEN = 100        # Iterations per cosine cycle (uniform)
LR_PEAK_DECAY = 0.85      # Each restart peak = 85% of previous peak
BUFFER_SIZE = 4096
PLOT_WINDOW = 25
EVAL_INTERVAL = 10       # Run eval every N iterations
EVAL_GAMES = 5        # Games per role vs random (x2 for P1+P2)
EVAL_UPGRADE_THRESHOLD = 0.6  # If vs-random score >= this, also run vs last checkpoint
EVAL_CHECKPOINT_GAMES = 5  # Games per role vs checkpoint (x2 for P1+P2)

LOG_LINES = ["Iteration,Loss,AvgReward\n"]

def log_metrics(iteration, loss, reward):
    LOG_LINES.append(f"{iteration},{loss:.4f},{reward:.2f}\n")

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
    
    # Initialize parallel games
    games = [BlackHoleGame() for _ in range(episodes)]
    active_games = list(range(episodes))
    
    # Histories for each game
    states_history = {i: [] for i in range(episodes)}
    policies_history = {i: [] for i in range(episodes)}
    
    move_count = 0
    total_moves = episodes * (games[0].num_hexes - 1)  # dynamic: 44 moves per game for L=9
    print(f"  [Self-Play] Starting {episodes} parallel games (MCTS sims: {MCTS_SIMS} per move)")
    
    while active_games:
        # Get active game instances
        current_boards = [games[i] for i in active_games]
        move_count += len(active_games)
        avg_tiles = sum(games[i].tiles_placed for i in active_games) / len(active_games)
        # print(f"  [Self-Play] Move batch | Active games: {len(active_games)} | Avg tiles placed: {avg_tiles:.1f}/20", flush=True)
        
        # Run MCTS centrally for all active games
        batched_probs = mcts.run_batched(current_boards, num_simulations=MCTS_SIMS, temperature=1.0)
        # print(f"  [Self-Play] MCTS done for this batch", flush=True)
        
        still_active = []
        for idx, probs in zip(active_games, batched_probs):
            game = games[idx]
            
            obs = {
               "board": game.board.copy(), 
               "current_tile": (game.tiles_placed // 2) + 1
            }
            
            # Canonicalize Observation for Storage
            if game.current_player == 2:
                 board = obs["board"]
                 p1 = (board[:, 0] == 1)
                 p2 = (board[:, 0] == 2)
                 board[p1, 0] = 2
                 board[p2, 0] = 1
                 obs["board"] = board
            
            states_history[idx].append(obs)
            policies_history[idx].append(probs)
            
            # Sample action
            action = np.random.choice(len(probs), p=probs)
            
            # Make move (tile_val is always (tiles_placed // 2) + 1, game enforces natural cap)
            tile_val = (game.tiles_placed // 2) + 1
            game.make_move(action, tile_val)
            
            if game.check_game_over():
                winner, _ = game.calculate_score()
                
                if winner == 1: game_value = 1.0
                elif winner == 2: game_value = -1.0
                else: game_value = 0.0
                
                # Backfill buffer
                curr_player = 1
                for i in range(len(states_history[idx])):
                    val = game_value if curr_player == 1 else -game_value
                    buffer.push(states_history[idx][i], policies_history[idx][i], val)
                    curr_player = 3 - curr_player
                    
                rewards.append(game_value)
                print(f"  [Self-Play] Game {idx+1} done | Winner: P{winner} | Reward: {game_value:+.0f} | Buffer: {len(buffer)}", flush=True)
            else:
                still_active.append(idx)
                
        active_games = still_active

    return np.mean(rewards)

def train(agent, optimizer, buffer, device):
    agent.train()
    running_loss = 0.0
    
    for _ in range(TRAINING_EPOCHS_PER_ITER):
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
        
        input_tensor = preprocess_batch(batch_obs, device) 
        
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
        print(l_pi.item(),l_v.item())
        loss = l_v + l_pi
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / TRAINING_EPOCHS_PER_ITER

def eval_vs_random(agent, mcts, device, games_per_role=20):
    """
    Plays agent (MCTS) vs a pure random opponent.
    Returns average reward from the agent's perspective.
    """
    agent.eval()
    rewards = []

    for role in [1, 2]:  # agent plays as P1 then P2
        for _ in range(games_per_role):
            game = BlackHoleGame()

            while not game.check_game_over():
                tile_val = (game.tiles_placed // 2) + 1

                if game.current_player == role:
                    probs = mcts.run(game, num_simulations=MCTS_SIMS_EVAL, temperature=0.0)
                    action = np.argmax(probs)
                else:
                    action = random.choice(game.get_valid_moves())

                game.make_move(action, tile_val)

            winner, _ = game.calculate_score()
            if winner == role:   rewards.append(1.0)
            elif winner == 0:    rewards.append(0.0)
            else:                rewards.append(-1.0)

    avg = float(np.mean(rewards))
    print(f"  [Eval vs Random] Avg reward over {games_per_role*2} games: {avg:+.3f} "
          f"(P1: {np.mean(rewards[:games_per_role]):+.2f}, P2: {np.mean(rewards[games_per_role:]):+.2f})",
          flush=True)
    return avg


def eval_vs_checkpoint(agent, mcts, device, checkpoint_path, games_per_role=10):
    """
    Plays current agent (MCTS) vs the last saved checkpoint model (also using MCTS).
    Only called when eval_vs_random >= EVAL_UPGRADE_THRESHOLD.
    Returns average reward from the current agent's perspective.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"  [Eval vs Checkpoint] No checkpoint found at {checkpoint_path}, skipping.")
        return None

    # Load old model
    old_agent = AlphaBH().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    old_agent.load_state_dict(state)
    old_agent.eval()
    old_mcts = AlphaMCTS(old_agent, device)

    agent.eval()
    rewards = []

    for role in [1, 2]:  # current agent plays as P1 then P2
        for _ in range(games_per_role):
            game = BlackHoleGame()

            while not game.check_game_over():
                tile_val = (game.tiles_placed // 2) + 1

                if game.current_player == role:
                    probs = mcts.run(game, num_simulations=MCTS_SIMS_EVAL, temperature=0.0)
                else:
                    probs = old_mcts.run(game, num_simulations=MCTS_SIMS_EVAL, temperature=0.0)

                probs_t = torch.tensor(probs, dtype=torch.float32)
                top_k = min(5, (probs_t > 0).sum().item())
                top_vals, top_idx = probs_t.topk(top_k)
                top_vals = top_vals / (top_vals.sum() + 1e-8)  # renormalize
                action = top_idx[torch.multinomial(top_vals, 1)].item()
                game.make_move(action, tile_val)

            winner, _ = game.calculate_score()
            if winner == role:   rewards.append(1.0)
            elif winner == 0:    rewards.append(0.0)
            else:                rewards.append(-1.0)

    avg = float(np.mean(rewards))
    print(f"  [Eval vs Checkpoint] Avg reward over {games_per_role*2} games: {avg:+.3f} "
          f"(P1: {np.mean(rewards[:games_per_role]):+.2f}, P2: {np.mean(rewards[games_per_role:]):+.2f})",
          flush=True)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint (model.pth) to resume training tightly from")
    parser.add_argument("--load", type=str, help="Path to checkpoint to start fresh training with pre-trained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = AlphaBH().to(device)
    optimizer = optim.SGD(agent.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Cosine annealing warm restart with geometrically decaying peaks
    # LR follows: peak * 0.5*(1 + cos(pi * t / T)) within each cycle
    # After each cycle, peak *= LR_PEAK_DECAY (down to 85% each time)
    def lr_lambda(iteration):
        cycle      = iteration // LR_CYCLE_LEN
        cycle_pos  = iteration  % LR_CYCLE_LEN
        peak       = LEARNING_RATE * (LR_PEAK_DECAY ** cycle)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / LR_CYCLE_LEN))
        lr         = LR_MIN + (peak - LR_MIN) * cos_factor
        return lr / LEARNING_RATE   # LambdaLR multiplies base_lr by this

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mcts = AlphaMCTS(agent, device)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    os.makedirs("trained_models", exist_ok=True)
    
    # Plotting Data
    train_losses = []
    avg_rewards = []
    eval_results = []   # (iteration, avg_reward) pairs

    start_iteration = 0

    # --- Loading Logic ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            if 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                agent.load_state_dict(checkpoint) # fallback

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            if 'iteration' in checkpoint:
                start_iteration = checkpoint['iteration']
                print(f"Resumed at Iteration: {start_iteration}")
                
            if 'train_losses' in checkpoint: train_losses = checkpoint['train_losses']
            if 'avg_rewards' in checkpoint: avg_rewards = checkpoint['avg_rewards']
            if 'eval_results' in checkpoint: eval_results = checkpoint['eval_results']
        else:
            print(f"Error: Checkpoint file not found: {args.resume}")
            return
            
    elif args.load:
        if os.path.isfile(args.load):
            print(f"Loading pretrained weights (Fine-Tuning): {args.load}")
            checkpoint = torch.load(args.load, map_location=device)
            if 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                agent.load_state_dict(checkpoint)
        else:
            print(f"Error: Model file not found: {args.load}")
            return
    
    # Auto-increment run number based on existing directories
    run_num = 1
    # If resuming, continue using the SAME directory instead of making a new one
    if args.resume:
        # assume the checkpoint is stored in the run_dir we want to use
        # e.g., trained_models/AlphaZero_DQN_run4/model.pth -> trained_models/AlphaZero_DQN_run4
        run_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"Resuming logging in existing directory: {run_dir}")
    else:
        while os.path.exists(f"trained_models/AlphaZero_DQN_run{run_num}"):
            run_num += 1
        run_dir = f"trained_models/AlphaZero_DQN_run{run_num}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving outputs to: {run_dir}")

    for iteration in range(start_iteration, TRAINING_ITERATIONS):
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
            
            # Step LR every iteration (cosine schedule tracks iteration count)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0] 
            
            # Log
            log_metrics(iteration, loss, avg_reward)
            print(f"Loss: {loss:.4f} | Avg Reward (P1): {avg_reward:.2f} | LR: {current_lr:.6f}")
        else:
            print("Skipping training (buffer too small)")
            train_losses.append(0)
            log_metrics(iteration, 0.0, avg_reward)

        # 2.5 Eval (every EVAL_INTERVAL iterations)
        if (iteration + 1) % EVAL_INTERVAL == 0:
            print(f"--- Eval vs Random (Iteration {iteration+1}) ---")
            eval_reward = eval_vs_random(agent, mcts, device, games_per_role=EVAL_GAMES)

            # If agent is crushing random (>= threshold), run harder eval vs last checkpoint
            checkpoint_path = os.path.join(run_dir, "model.pth")
            if eval_reward >= EVAL_UPGRADE_THRESHOLD:
                print(f"  >> Score {eval_reward:+.3f} >= {EVAL_UPGRADE_THRESHOLD}, "
                      f"running vs last checkpoint...")
                ckpt_reward = eval_vs_checkpoint(
                    agent, mcts, device, checkpoint_path,
                    games_per_role=EVAL_CHECKPOINT_GAMES
                )
                if ckpt_reward is not None:
                    eval_reward = ckpt_reward  # report the harder score
                    print(f"  >> Final reported eval (vs checkpoint): {eval_reward:+.3f}")

            eval_results.append((iteration + 1, eval_reward))

        # 3. Save updated plot each iteration (create fresh, close after)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.set_title("Training Loss")
        ax1.plot(train_losses, label="Loss")
        if len(train_losses) >= PLOT_WINDOW:
            ma_loss = [np.mean(train_losses[max(0, i-PLOT_WINDOW+1):i+1]) for i in range(len(train_losses))]
            ax1.plot(ma_loss, label=f"MA-{PLOT_WINDOW}", linestyle="--")
        ax1.legend()
        
        ax2.set_title("Self-Play Avg Reward (P1 perspective)")
        ax2.plot(avg_rewards, label="Reward")
        if len(avg_rewards) >= PLOT_WINDOW:
            ma_rew = [np.mean(avg_rewards[max(0, i-PLOT_WINDOW+1):i+1]) for i in range(len(avg_rewards))]
            ax2.plot(ma_rew, label=f"MA-{PLOT_WINDOW}", linestyle="--")
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax2.legend()

        ax3.set_title(f"Eval vs Random (every {EVAL_INTERVAL} iters)")
        if eval_results:
            xs, ys = zip(*eval_results)
            ax3.plot(xs, ys, marker='o', label="Eval Reward")
            ax3.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "training_plot.png"))
        plt.close(fig)  # Free memory, no GUI events needed
        
        # Save model checkpoint (overwrite same file each iteration)
        if len(buffer) >= BATCH_SIZE:
            save_dict = {
                'model_config': agent.config,
                'state_dict': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': iteration + 1,
                'train_losses': train_losses,
                'avg_rewards': avg_rewards,
                'eval_results': eval_results
            }
            torch.save(save_dict, os.path.join(run_dir, "model.pth"))
            
        with open(os.path.join(run_dir, "training.log"), "w") as f:
            f.writelines(LOG_LINES)

if __name__ == "__main__":
    main()
