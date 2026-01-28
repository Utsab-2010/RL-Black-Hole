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
BUFFER_SIZE = 2048
BATCH_SIZE = 512
TARGET_UPDATE = 1000
EVAL_INTERVAL = 100 # episodes
SELF_PLAY_UPDATE_THRESHOLD = 0.75 # Win rate > 55% to update opponent
NUM_EPISODES = 200000
OPPONENT_UPDATE_MIN_EPISODES = 3000 # Wait 1000 episodes
CHECKPOINT_INTERVAL = 5000 # Save checkpoint every N episodes
NUM_CYCLIC_DECAY_CYCLES = 50 # Number of restart cycles for epsilon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

def get_sinusoidal_embeddings(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # div_term = 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class QNetwork(nn.Module):
    def __init__(self, pos_dim=4, val_dim=4, player_dim=3, hidden_dims=[128, 64], start_val=-10):
        super(QNetwork, self).__init__()
        self.config = {
            'pos_dim': pos_dim,
            'val_dim': val_dim,
            'player_dim': player_dim,
            'hidden_dims': hidden_dims,
            'start_val': start_val
        }
        self.start_val = start_val 

        # ResNet Config (Downscaling)
        self.map_channels = 2 # Opponent, Player
        
        # Initial Conv (6x6)
        # 2 -> 32
        self.conv_in = nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Downscaling Blocks (ResNet-18 style expansion)
        self.layer1 = ResBlock(32, 64, stride=2)   # 6x6 -> 3x3
        self.layer2 = ResBlock(64, 128, stride=2)  # 3x3 -> 2x2
        self.layer3 = ResBlock(128, 256, stride=2) # 2x2 -> 1x1
        
        # Final flattened size: 256 * 1 * 1 = 256
        self.flattened_res_size = 256
        
        # Turn Embedding (Fixed Sinusoidal)
        self.turn_dim = 16
        # Register as buffer so it's saved with state_dict but not optimized
        self.register_buffer('turn_emb_table', get_sinusoidal_embeddings(22, self.turn_dim))
        
        # MLP Head
        mlp_input_dim = self.flattened_res_size + self.turn_dim
        
        layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 21) 

        # Indices for 21 positions in 6x6 grid
        self.pos_map = []
        idx = 0
        for r in range(6):
            for c in range(r + 1):
                self.pos_map.append((r, c))
                idx += 1
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Transform state to 6x6x2
        board_data = x[:, :42].view(batch_size, 21, 2)
        players = board_data[:, :, 0] # (Batch, 21)
        
        grid = torch.full((batch_size, 2, 6, 6), self.start_val, device=device, dtype=torch.float32)
        
        rows = torch.tensor([r for r, c in self.pos_map], device=device)
        cols = torch.tensor([c for r, c in self.pos_map], device=device)
        
        # Fill Lower Triangle
        grid[:, 0, rows, cols] = (players == 1).float() * 1.0
        grid[:, 1, rows, cols] = (players == 2).float() * 2.0
        
        # 2. ResNet Encoding
        out = self.conv_in(grid)
        out = self.bn_in(out)
        out = self.relu(out)
        
        out = self.layer1(out) # 32 -> 64, 3x3
        out = self.layer2(out) # 64 -> 128, 2x2
        out = self.layer3(out) # 128 -> 256, 1x1
        
        # Flatten
        out = out.view(batch_size, -1) # (B, 256)
        
        # 3. Turn Encoding
        turn_count = (players != 0).sum(dim=1) # (B,)
        turn_emb = self.turn_emb_table[turn_count] # (B, 16)
        
        # Concatenate
        combined = torch.cat([out, turn_emb], dim=1)
        
        # 4. MLP
        logits = self.mlp(combined) 
        output = self.output_head(logits) # (B, 21)
        
        # 5. Masking and Softmax
        filled_mask = (players != 0)
        output = output.masked_fill(filled_mask, -float('inf'))
        
        probs = torch.softmax(output, dim=1)
        final_values = probs * 100.0
        
        return final_values

def preprocess_obs(obs):
    # Convert dict obs to flattened FloatTensor of indices
    # board: (21, 2)
    board = obs["board"].flatten()
    current_tile = np.array([obs["current_tile"]])
    
    # Concatenate to (43,) array
    features = np.concatenate([board, current_tile])
    return torch.FloatTensor(features).to(device)

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
            mask = get_action_mask(obs).unsqueeze(0)
            state_tensor = preprocess_obs(obs).unsqueeze(0)
            
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
                state_tensor = preprocess_obs(obs).unsqueeze(0)
                mask = get_action_mask(obs).unsqueeze(0)
                
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
                
                episodes_since_update = episode - last_opponent_update_episode
                if eval_win_rate > SELF_PLAY_UPDATE_THRESHOLD and episodes_since_update >= OPPONENT_UPDATE_MIN_EPISODES:
                    log(f">>> PROMOTING MODEL: Updating Opponent (Eval WR: {eval_win_rate:.2f}) <<<")
                    opponent_model.load_state_dict(current_model.state_dict())
                    last_opponent_update_episode = episode
                elif eval_win_rate > SELF_PLAY_UPDATE_THRESHOLD:
                    log(f"Eval Win rate good ({eval_win_rate:.2f}) but waiting for min episodes ({episodes_since_update}/{OPPONENT_UPDATE_MIN_EPISODES})")
            
            # Checkpoint
            if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
                checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
                checkpoint = {
                    'model_config': current_model.config,
                    'state_dict': current_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
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
