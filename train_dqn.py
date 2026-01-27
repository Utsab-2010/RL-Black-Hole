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

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 64
TARGET_UPDATE = 1000
EVAL_INTERVAL = 100 # episodes
SELF_PLAY_UPDATE_THRESHOLD = 0.75 # Win rate > 55% to update opponent
NUM_EPISODES = 30000
OPPONENT_UPDATE_MIN_EPISODES = 3000 # Wait 1000 episodes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class QNetwork(nn.Module):
    def __init__(self, pos_dim=4, val_dim=4, player_dim=3, hidden_dims=[128, 64]):
        super(QNetwork, self).__init__()
        self.config = {
            'pos_dim': pos_dim,
            'val_dim': val_dim,
            'player_dim': player_dim,
            'hidden_dims': hidden_dims
        }
        
        # Embeddings
        self.pos_emb = nn.Embedding(21, pos_dim)
        self.val_emb = nn.Embedding(11, val_dim)
        self.player_emb = nn.Embedding(3, player_dim)
        
        # Per Position Feature Size
        per_pos_feat = pos_dim + val_dim + player_dim
        # Total Board Flattened
        board_flat_dim = 21 * per_pos_feat
        # Current Tile Feature (uses val_emb)
        input_dim = board_flat_dim + val_dim
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 21)

    def forward(self, x):
        batch_size = x.shape[0]
        
        board_data = x[:, :42].view(batch_size, 21, 2)
        players = board_data[:, :, 0]
        values = board_data[:, :, 1]
        current_tile = x[:, 42]
        
        pos_indices = torch.arange(21, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        p_emb = self.player_emb(players)
        v_emb = self.val_emb(values)
        pos_emb = self.pos_emb(pos_indices)
        
        board_feat = torch.cat([p_emb, v_emb, pos_emb], dim=2)
        board_flat = board_feat.view(batch_size, -1)
        
        cur_feat = self.val_emb(current_tile)
        total_feat = torch.cat([board_flat, cur_feat], dim=1)
        
        feat = self.fc_layers(total_feat)
        return self.output_layer(feat)

def preprocess_obs(obs):
    # Convert dict obs to flattened LongTensor of indices
    # board: (21, 2)
    board = obs["board"].flatten()
    current_tile = np.array([obs["current_tile"]])
    
    # Concatenate to (43,) array
    features = np.concatenate([board, current_tile]).astype(int) # Ensure int for Embedding
    return torch.LongTensor(features).to(device)

def get_action_mask(obs):
    # Return bool tensor: True if action is valid
    # valid moves: where board[i, 0] == 0
    valid = (obs["board"][:, 0] == 0)
    return torch.BoolTensor(valid).to(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        return state, action, reward, next_state, done, mask, next_mask

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
    
    # --- Setup Directories (Moved to Start) ---
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
        state = preprocess_obs(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = opponent_model(state)
            # Mask invalid
            mask = get_action_mask(obs).unsqueeze(0)
            q_values[~mask] = -float('inf')
            action = q_values.argmax().item()
        return action

    env = SelfPlayWrapper(env_raw, opponent_policy=opponent_policy_fn)
    
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    steps = 0
    last_opponent_update_episode = 0
    epsilon = EPSILON_START
    
    # Metrics
    recent_rewards = deque(maxlen=100)
    avg_rewards_log = []
    win_rates = []

    log("Starting Training Loop...")
    
    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            # Linear Decay over episodes (Start of Episode)
            progress = episode / NUM_EPISODES
            epsilon = max(EPSILON_END, EPSILON_START - progress * (EPSILON_START - EPSILON_END))
            
            while not done:
                state_tensor = preprocess_obs(obs).unsqueeze(0)
                mask = get_action_mask(obs).unsqueeze(0)
                
                # Epsilon Greedy
                if random.random() < epsilon:
                    # Random valid move
                    valid_indices = torch.nonzero(mask[0]).flatten().cpu().numpy()
                    action = random.choice(valid_indices)
                else:
                    with torch.no_grad():
                        q_values = current_model(state_tensor)
                        q_values[~mask] = -float('inf')
                        action = q_values.argmax().item()
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                buffer.push(
                    preprocess_obs(obs).cpu().numpy(), 
                    action, 
                    reward, 
                    preprocess_obs(next_obs).cpu().numpy(), 
                    done,
                    get_action_mask(obs).cpu().numpy(),
                    get_action_mask(next_obs).cpu().numpy()
                )
                
                obs = next_obs
                episode_reward += reward
                steps += 1
                
                # Train Step
                if len(buffer.buffer) > BATCH_SIZE:
                    s_batch, a_batch, r_batch, ns_batch, d_batch, m_batch, nm_batch = buffer.sample(BATCH_SIZE)
                    
                    s_batch = torch.LongTensor(np.array(s_batch)).to(device)
                    a_batch = torch.LongTensor(np.array(a_batch)).unsqueeze(1).to(device)
                    r_batch = torch.FloatTensor(np.array(r_batch)).unsqueeze(1).to(device)
                    ns_batch = torch.LongTensor(np.array(ns_batch)).to(device)
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

            # Evaluation & Self-Play Update
            if episode % EVAL_INTERVAL == 0 and episode > 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_rewards_log.append(avg_reward)
                log(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
                
                # Eval against opponent
                wins = 0
                n_eval = 20
                for _ in range(n_eval):
                    # Deterministic Eval
                    e_obs, _ = env.reset()
                    e_done = False
                    while not e_done:
                        e_tens = preprocess_obs(e_obs).unsqueeze(0)
                        e_mask = get_action_mask(e_obs).unsqueeze(0)
                        with torch.no_grad():
                            q = current_model(e_tens)
                            q[~e_mask] = -float('inf')
                            act = q.argmax().item()
                        e_obs, r, t, tr, _ = env.step(act)
                        e_done = t or tr
                        if e_done and r == 1: # We (agent) won
                            wins += 1
                
                win_rate = wins / n_eval
                win_rates.append(win_rate)
                log(f"Eval Win Rate vs Opponent: {win_rate:.2f}")
                
                episodes_since_update = episode - last_opponent_update_episode
                if win_rate > SELF_PLAY_UPDATE_THRESHOLD and episodes_since_update >= OPPONENT_UPDATE_MIN_EPISODES:
                    log(">>> PROMOTING MODEL: Updating Opponent to Current Model <<<")
                    opponent_model.load_state_dict(current_model.state_dict())
                    last_opponent_update_episode = episode
                elif win_rate > SELF_PLAY_UPDATE_THRESHOLD:
                    log(f"Win rate good ({win_rate:.2f}) but waiting for min episodes ({episodes_since_update}/{OPPONENT_UPDATE_MIN_EPISODES})")
                    
    except KeyboardInterrupt:
        log("\nTraining interrupted by user. Saving current model...")
        
    finally:
        # Save Model
        model_path = os.path.join(save_dir, "model.pth")
        
        checkpoint = {
            'model_config': current_model.config,
            'state_dict': current_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
