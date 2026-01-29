import gymnasium as gym
import black_hole # Register env
from black_hole.wrappers import SelfPlayWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import copy

from black_hole.model import QNetwork, preprocess_obs, get_action_mask

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
BUFFER_SIZE = 2048
BATCH_SIZE = 256
TARGET_UPDATE = 1000
EVAL_INTERVAL = 100 # episodes
SELF_PLAY_UPDATE_THRESHOLD = 0.75 # Win rate > 55% to update opponent
NUM_EPISODES = 500000
OPPONENT_UPDATE_MIN_EPISODES = 3000 # Wait 1000 episodes
CHECKPOINT_INTERVAL = 5000 # Save checkpoint every N episodes
NUM_CYCLIC_DECAY_CYCLES = 50 # Number of restart cycles for epsilon
OPPONENT_UPDATE_REQ_STREAK = 3 # Require N consecutive wins > threshold to update


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definitions moved to black_hole/model.py



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        return state, action, reward, next_state, done, mask, next_mask

def evaluate_winrate(env, model, num_episodes=20):
    """
    Runs the agent using pure exploitation (argmax) against the current opponent in the env.
    Returns the win rate (0.0 to 1.0).
    """
    wins = 0
    model.eval() # Set to eval mode
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = get_action_mask(obs, device).unsqueeze(0)
            state_tensor = preprocess_obs(obs, device).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model(state_tensor)
                q_values[~mask] = -float('inf')
                action = q_values.argmax().item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done and reward > 0:
                wins += 1
                
    model.train() # Set back to train mode
    return wins / num_episodes

# --- Training Loop ---
def train():
    env_raw = gym.make("BlackHole-v0")
    
    # Model Config
    model_config = {
        'pos_dim': 4,
        'val_dim': 4,
        'player_dim': 3,
        'hidden_dims': [128, 64]
    }
    
    # Opponent Policies
    current_model = QNetwork(**model_config).to(device)
    target_model = QNetwork(**model_config).to(device)
    target_model.load_state_dict(current_model.state_dict())
    
    # "Best" model (Opponent)
    opponent_model = QNetwork(**model_config).to(device)
    opponent_model.load_state_dict(current_model.state_dict())
    opponent_model.eval()
    
    # --- Setup Directories ---
    base_dir = "trained_models"
    method = "DQN"
    game = "BlackHole"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    existing = [d for d in os.listdir(base_dir) if d.startswith(f"{game}_{method}_v")]
    versions = []
    for d in existing:
        try:
            v = int(d.split("_v")[-1])
            versions.append(v)
        except ValueError:
            pass
    next_version = max(versions) + 1 if versions else 1
    
    save_dir = os.path.join(base_dir, f"{game}_{method}_v{next_version}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Real-time Log File
    log_file_path = os.path.join(save_dir, "training_run.log")
    
    def log(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    log(f"Starting Training Run v{next_version}")
    log(f"Hyperparameters: LR={LEARNING_RATE}, Gamma={GAMMA}, Eps={EPSILON_START}->{EPSILON_END}")
    log(f"Model Architecture: {current_model}")
    
    def opponent_policy_fn(obs):
        # Input obs is canonical dictionary
        state = preprocess_obs(obs, device).unsqueeze(0)
        with torch.no_grad():
            q_values = opponent_model(state)
            # Mask invalid
            mask = get_action_mask(obs, device).unsqueeze(0)
            q_values[~mask] = -float('inf')
            action = q_values.argmax().item()
        return action

    env = SelfPlayWrapper(env_raw, opponent_policy=opponent_policy_fn)
    
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    steps = 0
    last_opponent_update_episode = 0
    consecutive_wins = 0
    epsilon = EPSILON_START
    
    # Metrics
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    avg_rewards_log = []
    win_rates = []

    log("Starting Training Loop...")
    
    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            # Cyclic Decay Schedule
            # Divide training into N cycles
            cycle_len = NUM_EPISODES / NUM_CYCLIC_DECAY_CYCLES
            current_cycle = int(episode / cycle_len)
            cycle_progress = (episode % cycle_len) / cycle_len
            
            # Max epsilon for this cycle decays from START to END across cycles
            cycle_max_eps = EPSILON_START - (current_cycle / NUM_CYCLIC_DECAY_CYCLES) * (EPSILON_START - EPSILON_END)
            cycle_max_eps = max(cycle_max_eps, EPSILON_END)
            
            # Decay within cycle (Linear)
            epsilon = EPSILON_END + (cycle_max_eps - EPSILON_END) * (1 - cycle_progress)
            
            while not done:
                state_tensor = preprocess_obs(obs, device).unsqueeze(0)
                mask = get_action_mask(obs, device).unsqueeze(0)
                
                # Epsilon Greedy
                if random.random() < epsilon:
                    # Random valid move
                    valid_indices = torch.nonzero(mask[0]).flatten().cpu().numpy()
                    action = random.choice(valid_indices)
                else:
                    current_model.eval()
                    with torch.no_grad():
                        q_values = current_model(state_tensor)
                        q_values[~mask] = -float('inf')
                        action = q_values.argmax().item()
                    current_model.train()
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                buffer.push(
                    preprocess_obs(obs, device).cpu().numpy(), 
                    action, 
                    reward, 
                    preprocess_obs(next_obs, device).cpu().numpy(), 
                    done,
                    get_action_mask(obs, device).cpu().numpy(),
                    get_action_mask(next_obs, device).cpu().numpy()
                )
                
                obs = next_obs
                episode_reward += reward
                steps += 1
                
                # Train Step
                if len(buffer.buffer) > BATCH_SIZE:
                    s_batch, a_batch, r_batch, ns_batch, d_batch, m_batch, nm_batch = buffer.sample(BATCH_SIZE)
                    
                    s_batch = torch.FloatTensor(np.array(s_batch)).to(device)
                    a_batch = torch.LongTensor(np.array(a_batch)).unsqueeze(1).to(device)
                    r_batch = torch.FloatTensor(np.array(r_batch)).unsqueeze(1).to(device)
                    ns_batch = torch.FloatTensor(np.array(ns_batch)).to(device)
                    d_batch = torch.FloatTensor(np.array(d_batch)).unsqueeze(1).to(device)
                    nm_batch = torch.BoolTensor(np.array(nm_batch)).to(device)
                    
                    # Q(s, a)
                    q_val = current_model(s_batch).gather(1, a_batch)
                    
                    # Q(s', a') - Double DQN or Standard? Standard for now.
                    with torch.no_grad():
                        next_q = target_model(ns_batch)
                        next_q[~nm_batch] = -float('inf')
                        max_next_q = next_q.max(1)[0].unsqueeze(1)
                        target = r_batch + (1 - d_batch) * GAMMA * max_next_q
                    
                    loss = nn.MSELoss()(q_val, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    optimizer.step()

                # Update Target
                if steps % TARGET_UPDATE == 0:
                    target_model.load_state_dict(current_model.state_dict())

            recent_rewards.append(episode_reward)
            # Track Win (1 if reward > 0, assumption based on BlackHole rules)
            recent_wins.append(1 if episode_reward > 0 else 0)

            # Evaluation & Self-Play Update
            if episode % EVAL_INTERVAL == 0 and episode > 0:
                avg_reward = np.mean(recent_rewards)
                train_win_rate = np.mean(recent_wins)
                
                # Pure Exploitation Evaluation
                eval_win_rate = evaluate_winrate(env, current_model, num_episodes=20)
                
                avg_rewards_log.append(avg_reward)
                win_rates.append(eval_win_rate) # Track Eval Win Rate
                
                log(f"Episode {episode} | AvgReward: {avg_reward:.2f} | TrainWR: {train_win_rate:.2f} | EvalWR: {eval_win_rate:.2f} | Eps: {epsilon:.2f}")
                
                # Streak Logic
                if eval_win_rate > SELF_PLAY_UPDATE_THRESHOLD:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 0
                
                episodes_since_update = episode - last_opponent_update_episode
                
                if consecutive_wins >= OPPONENT_UPDATE_REQ_STREAK and episodes_since_update >= OPPONENT_UPDATE_MIN_EPISODES:
                    log(f">>> PROMOTING MODEL: Updating Opponent (Streak: {consecutive_wins}, Eval WR: {eval_win_rate:.2f}) <<<")
                    opponent_model.load_state_dict(current_model.state_dict())
                    last_opponent_update_episode = episode
                    consecutive_wins = 0 # Reset streak after promotion
                elif consecutive_wins > 0:
                    log(f"Win Streak: {consecutive_wins}/{OPPONENT_UPDATE_REQ_STREAK} (WR: {eval_win_rate:.2f}) - Waiting for streak or min episodes ({episodes_since_update}/{OPPONENT_UPDATE_MIN_EPISODES})")
            
            # Checkpoint
            if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
                checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
                checkpoint = {
                    'model_config': current_model.config,
                    'state_dict': current_model.state_dict(),
                    'episode': episode
                }
                torch.save(checkpoint, checkpoint_path)
                log(f"Checkpoint saved to {checkpoint_path}")
                    
    except KeyboardInterrupt:
        log("\nTraining interrupted by user. Saving current model...")
        
    finally:
        # Save Model
        model_path = os.path.join(save_dir, "model.pth")
        
        checkpoint = {
            'model_config': current_model.config,
            'state_dict': current_model.state_dict(),
            'episode': NUM_EPISODES
        }
        
        torch.save(checkpoint, model_path)
        log(f"Model saved to {model_path}")
        
        # Save Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(avg_rewards_log)
        plt.title("Avg Training Reward (Last 100 Episodes)")
        plt.xlabel(f"Eval Interval ({EVAL_INTERVAL} eps)")
        
        plt.subplot(1, 2, 2)
        plt.plot(win_rates)
        plt.title(f"Win Rate vs Past Self (Update > {SELF_PLAY_UPDATE_THRESHOLD})")
        plt.xlabel(f"Eval Interval ({EVAL_INTERVAL} eps)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_log.png"))
        log("Training plot saved.")
        
        # Save Log Text (Summary)
        log_path = os.path.join(save_dir, "training_summary.txt")
        with open(log_path, "w") as f:
            f.write(f"Game: {game}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Version: {next_version}\n")
            f.write(f"Episodes: {NUM_EPISODES}\n")
            f.write(f"Buffer Size: {BUFFER_SIZE}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Learning Rate: {LEARNING_RATE}\n")
            f.write(f"Gamma: {GAMMA}\n")
            f.write(f"Self-Play Update Threshold: {SELF_PLAY_UPDATE_THRESHOLD}\n")
            f.write(f"Epsilon Decay: {EPSILON_DECAY}\n")
            f.write("-" * 20 + "\n")
            f.write("Model Architecture:\n")
            f.write(str(current_model) + "\n")
            f.write("-" * 20 + "\n")
            f.write(f"Final Win Rate (vs Opponent): {win_rates[-1] if win_rates else 'N/A'}\n")
        print(f"Training summary saved to {log_path}")
    
    if __name__ == "__main__":
        train()
