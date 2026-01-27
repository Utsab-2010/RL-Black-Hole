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
OPPONENT_UPDATE_MIN_EPISODES = 1000 # Wait 1000 episodes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging

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
            layers.append(nn.Tanh()) # Using Tanh as requested previously
            prev_dim = h_dim
        
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 21)

    def forward(self, x):
        # x input is now expected to be LongTensor of shape (Batch, 43)
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

# --- Helper Functions ---
def preprocess_obs(obs):
    # obs is a dict from gymnasium env
    # board: (21, 2) array, col 0 is player, col 1 is value
    # current_tile: int
    
    board_data = obs["board"].flatten() # (42,)
    current_tile = np.array([obs["current_tile"]]) # (1,)
    
    # Concatenate and convert to LongTensor
    processed = np.concatenate((board_data, current_tile))
    return torch.LongTensor(processed).to(device)

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
    # --- Setup Directories & Logging (Start) ---
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
    
    # Configure Logging
    log_file = os.path.join(save_dir, "training_run.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting Training Run v{next_version}")
    logging.info(f"Hyperparameters: LR={LEARNING_RATE}, Gamma={GAMMA}, Eps={EPSILON_START}->{EPSILON_END}")
    
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
    
    opponent_model = QNetwork(**model_config).to(device)
    opponent_model.load_state_dict(current_model.state_dict())
    opponent_model.eval()
    
    logging.info(f"Model Architecture: {current_model}")

    def opponent_policy_fn(obs):
        # input obs is canonical dictionary
        print("DEBUG: Opponent Start", end=' ')
        state = preprocess_obs(obs).unsqueeze(0)
        with torch.no_grad():
            print("DEBUG: Opponent Forward", end=' ')
            q_values = opponent_model(state)
            mask = get_action_mask(obs).unsqueeze(0)
            q_values[~mask] = -float('inf')
            action = q_values.argmax().item()
        print("DEBUG: Opponent End", end=' ')
        return action

    env = SelfPlayWrapper(env_raw, opponent_policy=opponent_policy_fn)
    
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    steps = 0
    last_opponent_update_episode = 0
    epsilon = EPSILON_START
    
    recent_rewards = deque(maxlen=100)
    avg_rewards_log = []
    win_rates = []
    
    print("Starting Training Loop...")
    
    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            # Linear Decay over episodes
            # From 1.0 to 0.05 over 100% of episodes
            progress = episode / NUM_EPISODES
            epsilon = max(EPSILON_END, EPSILON_START - progress * (EPSILON_START - EPSILON_END))
            
            while not done:
                # Debug liveness
                print(f"DEBUG: Step {steps}, Eps {epsilon:.3f}", end='\r')

                print("DEBUG: Preprocess", end=' ')
                state_tensor = preprocess_obs(obs).unsqueeze(0)
                mask = get_action_mask(obs).unsqueeze(0)
                
                if random.random() < epsilon:
                    valid_indices = torch.nonzero(mask[0]).flatten().cpu().numpy()
                    action = random.choice(valid_indices)
                else:
                    with torch.no_grad():
                        print("DEBUG: Forward", end=' ')
                        q_values = current_model(state_tensor)
                        q_values[~mask] = -float('inf')
                        action = q_values.argmax().item()
                
                print("DEBUG: Step", end=' ')
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Note: preprocess_obs returns FloatTensor? 
            # In previous steps I fixed it to LongTensor in the buffer sampling, 
            # but preprocess_obs itself should return LongTensor.
            # I will assume preprocess_obs is correct (LongTensor) from previous edits.
            
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
            
            epsilon = max(EPSILON_END, EPSILON_START - (steps / EPSILON_DECAY))
            
            if len(buffer.buffer) > BATCH_SIZE:
                s_batch, a_batch, r_batch, ns_batch, d_batch, m_batch, nm_batch = buffer.sample(BATCH_SIZE)
                
                s_batch = torch.LongTensor(np.array(s_batch)).to(device)
                a_batch = torch.LongTensor(np.array(a_batch)).unsqueeze(1).to(device)
                r_batch = torch.FloatTensor(np.array(r_batch)).unsqueeze(1).to(device)
                ns_batch = torch.LongTensor(np.array(ns_batch)).to(device)
                d_batch = torch.FloatTensor(np.array(d_batch)).unsqueeze(1).to(device)
                nm_batch = torch.BoolTensor(np.array(nm_batch)).to(device)
                
                q_val = current_model(s_batch).gather(1, a_batch)
                
                with torch.no_grad():
                    next_q = target_model(ns_batch)
                    next_q[~nm_batch] = -float('inf')
                    max_next_q = next_q.max(1)[0].unsqueeze(1)
                    target = r_batch + (1 - d_batch) * GAMMA * max_next_q
                
                loss = nn.MSELoss()(q_val, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % TARGET_UPDATE == 0:
                target_model.load_state_dict(current_model.state_dict())

        recent_rewards.append(episode_reward)

        # Evaluation & Logging
        if episode % EVAL_INTERVAL == 0 and episode > 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_rewards_log.append(avg_reward)
            
            # Eval against opponent
            wins = 0
            n_eval = 20
            for _ in range(n_eval):
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
                    if e_done and r == 1: wins += 1
            
            win_rate = wins / n_eval
            win_rates.append(win_rate)
            
            log_msg = f"Episode {episode}: AvgReward={avg_reward:.2f}, WinRate={win_rate:.2f}, Eps={epsilon:.2f}"
            logging.info(log_msg)
            print(log_msg) # Force print to console
            
            episodes_since_update = episode - last_opponent_update_episode
            if win_rate > SELF_PLAY_UPDATE_THRESHOLD and episodes_since_update >= OPPONENT_UPDATE_MIN_EPISODES:
                msg = ">>> PROMOTING MODEL: Updating Opponent <<<"
                logging.info(msg)
                print(msg)
                opponent_model.load_state_dict(current_model.state_dict())
                last_opponent_update_episode = episode
            elif win_rate > SELF_PLAY_UPDATE_THRESHOLD:
                msg = f"Win rate {win_rate:.2f} but waiting ({episodes_since_update}/{OPPONENT_UPDATE_MIN_EPISODES} eps)"
                logging.info(msg)
                print(msg)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        logging.info("Training interrupted by KeyboardInterrupt.")

    finally:
        # Save logic (At end of loop or interrupt)
        # Save Model Checkpoint (Dict)
        checkpoint = {
            'model_config': current_model.config,
            'state_dict': current_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'episode': episode # variable might not exist if fail before loop, but ok for now
        }
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(checkpoint, model_path)
        logging.info(f"Model checkpoint saved to {model_path}")
        print(f"Model saved to {model_path}")
        
        # Save Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(avg_rewards_log)
        plt.title("Avg Training Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(win_rates)
        plt.title("Win Rate vs Past Self")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_log.png"))
        logging.info("Training plot saved.")
        
        logging.info("Training Complete.")

if __name__ == "__main__":
    train()
