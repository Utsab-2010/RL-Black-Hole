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

from black_hole.model import QNetwork, preprocess_batch, get_action_mask_batch, get_action_mask

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
BUFFER_SIZE = 5096
BATCH_SIZE = 2048
TARGET_UPDATE = 1000
EVAL_INTERVAL = 100 # episodes
SELF_PLAY_UPDATE_THRESHOLD = 0.75 # Win rate > 55% to update opponent
NUM_EPISODES = 500000
OPPONENT_UPDATE_MIN_EPISODES = 3000 # Wait 1000 episodes
CHECKPOINT_INTERVAL = 5000 # Save checkpoint every N episodes
NUM_CYCLIC_DECAY_CYCLES = 30 # Number of restart cycles for epsilon
NUM_ENVS = 512
OPPONENT_UPDATE_REQ_STREAK = 3 # Require N consecutive wins > threshold to update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definitions moved to black_hole/model.py

def evaluate_winrate(env, model, device, num_episodes=20):
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
            
            # Prepare batch of 1
            obs_batch = {
                "board": np.expand_dims(obs["board"], axis=0),
                "current_tile": np.expand_dims(obs["current_tile"], axis=0)
            }
            state_tensor = preprocess_batch(obs_batch, device)
            
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

def canonicalize_batch(obs, active_player_mask):
    # obs is a dict of numpy arrays (mutable). We modify in place or copy.
    # We want to flip the board for rows where active_player_mask is True (meaning it is P2's turn).
    # Since VectorEnv returns numpy arrays, we can work with numpy.
    
    # Copy to avoid corrupting original env state if needed (though usually fine to mutate for network input)
    # But wait, we need next_state to be correct (uncanonicalized) for the buffer if we store raw states?
    # Our buffer stores PREPROCESSED states (canonicalized). 
    
    # We will return a COPY of the components to be safe.
    
    board = obs["board"].copy() # (B, 21, 2)
    
    # active_player_mask: Boolean array (B,), True if we need to swap P1/P2
    
    # Swap 1 <-> 2 in board[active_player_mask]
    # Indices where P1(1) -> 2
    # Indices where P2(2) -> 1
    
    mask_flip = active_player_mask
    
    if np.any(mask_flip):
        sub_board = board[mask_flip]
        p1_mask = (sub_board[:, :, 0] == 1)
        p2_mask = (sub_board[:, :, 0] == 2)
        
        sub_board[:, :, 0][p1_mask] = 2
        sub_board[:, :, 0][p2_mask] = 1
        
        board[mask_flip] = sub_board
        
    return {
        "board": board,
        "current_tile": obs["current_tile"].copy()
    }

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

    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make("BlackHole-v0") for _ in range(NUM_ENVS)
    ])
    # Single env for evaluation
    eval_env_single = gym.make("BlackHole-v0")
    # Wrap it for Self-Play (standard wrapper) because evaluate_winrate expects simple step()
    # But wait, evaluate_winrate acts as Player 1.
    # We need an opponent.
    # The original evaluate_winrate assumed the env had an opponent policy? 
    # Or did we wrap it?
    # Original code: env = SelfPlayWrapper(env_raw, opponent_policy=opponent_policy_fn)
    # So we need to wrap the eval env too!
    
    def opponent_policy_fn(obs):
        # Input obs is canonical dictionary (single)
        obs_batch = {
            "board": np.expand_dims(obs["board"], axis=0),
            "current_tile": np.expand_dims(obs["current_tile"], axis=0)
        }
        state = preprocess_batch(obs_batch, device)
        with torch.no_grad():
            q_values = opponent_model(state)
            # Mask invalid
            mask = get_action_mask(obs, device).unsqueeze(0)
            q_values[~mask] = -float('inf')
            action = q_values.argmax().item()
        return action
        
    eval_env = SelfPlayWrapper(eval_env_single, opponent_policy=opponent_policy_fn)
    
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

    log(f"Starting Vectorized Training Run v{next_version} with {NUM_ENVS} envs")
    log(f"Hyperparameters: LR={LEARNING_RATE}, Gamma={GAMMA}, Eps={EPSILON_START}->{EPSILON_END}")
    
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    global_steps = 0 # Total environment steps across all envs
    episode_count = 0
    global_steps = 0 # Total environment steps across all envs
    episode_count = 0
    last_opponent_update_episode = 0
    consecutive_wins = 0
    
    # Metrics
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    avg_rewards_log = []
    win_rates = []

    log("Starting Training Loop...")
    
    # Initialize Environments
    obs, _ = envs.reset()
    
    # Track "pending" transitions for the Agent (P1)
    # We only store transitions for Player 1.
    # pending_obs[env_id] = torch_tensor_state
    # pending_action[env_id] = action
    # accumulated_reward[env_id] = reward_sum
    # active[env_id] = True/False (Is P1 waiting for P2 response?)
    
    pending_obs = [None] * NUM_ENVS
    pending_action = [None] * NUM_ENVS
    accumulated_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    
    try:
        while episode_count < NUM_EPISODES:
            # 1. Determine who needs to act in each env
            # obs["current_player"] is (B,) array of 1 or 2.
            current_players = obs["current_player"]
            
            # 2. Select Actions
            actions = np.zeros(NUM_ENVS, dtype=int)
            
            # --- AGENT (P1) ACTIONS ---
            p1_indices = np.where(current_players == 1)[0]
            if len(p1_indices) > 0:
                # Get Canonical Obs for P1 (No flip needed, already 1)
                # But we need to separate batch.
                # Construct sub-batch
                
                # We need to construct a "canonical" sub-batch for inference
                # Dict of arrays -> mask -> wrap in Tensor
                
                # Optimized: convert ALL to tensor first?
                # preprocess_batch does the conversion.
                
                # Slice the dict items
                obs_p1 = {k: v[p1_indices] for k, v in obs.items()}
                
                # Preprocess
                state_p1 = preprocess_batch(obs_p1, device)
                mask_p1 = get_action_mask_batch(obs_p1, device)
                
                # Epsilon calculation
                # Use global episode count for epsilon
                cycle_len = NUM_EPISODES / NUM_CYCLIC_DECAY_CYCLES
                current_cycle = int(episode_count / cycle_len)
                cycle_progress = (episode_count % cycle_len) / cycle_len
                cycle_max_eps = max(EPSILON_END, EPSILON_START - (current_cycle / NUM_CYCLIC_DECAY_CYCLES) * (EPSILON_START - EPSILON_END))
                epsilon = EPSILON_END + (cycle_max_eps - EPSILON_END) * (1 - cycle_progress)
                
                # Epsilon Greedy
                # Generate random numbers for each env in p1_indices
                rands = np.random.random(len(p1_indices))
                exploit_mask = rands >= epsilon
                
                # Random Actions
                # We need valid moves.
                # mask_p1 is (B, 21) bool
                
                p1_actions = np.zeros(len(p1_indices), dtype=int)
                
                # TODO: Vectorize random choice? 
                # Loop for randoms is fast enough for small batch
                for i, is_exploit in enumerate(exploit_mask):
                    valid = torch.nonzero(mask_p1[i]).cpu().flatten().numpy()
                    if len(valid) == 0:
                        p1_actions[i] = 0 # Should not happen if game valid
                    elif not is_exploit:
                        p1_actions[i] = np.random.choice(valid)
                    else:
                        # Exploit placeholder (will fill with model)
                        p1_actions[i] = -1 
                
                # Model Inference for exploiters
                exploit_indices = np.where(exploit_mask)[0]
                if len(exploit_indices) > 0:
                    model_input = state_p1[exploit_indices]
                    model_mask = mask_p1[exploit_indices]
                    
                    current_model.eval()
                    with torch.no_grad():
                        q_vals = current_model(model_input)
                        q_vals[~model_mask] = -float('inf')
                        best_acts = q_vals.argmax(dim=1).cpu().numpy()
                    current_model.train()
                    
                    p1_actions[exploit_indices] = best_acts
                
                # Store actions
                actions[p1_indices] = p1_actions
                
                # BUFFER LOGIC START (P1 just acted)
                # Store the state and action as "Pending"
                # But wait, we need to push the PREVIOUS transition first if it exists?
                # Rules:
                # P1 -> acts. (Pending: S, A). Reward=0 so far.
                # ... P2 acts ...
                # P1 -> turn again. 
                # This means (S, A) led to NewS. Reward accumulated.
                # So we push (PendingS, PendingA, AccReward, NewS, Done=False).
                
                for i, idx in enumerate(p1_indices):
                    # Convert Tensor state to numpy for buffer
                    s_numpy = state_p1[i].cpu().numpy()
                    a = p1_actions[i]
                    
                    if pending_obs[idx] is not None:
                        # Close previous transition
                        # The 'state_p1[i]' is the NEXT state for the previous action
                        prev_s = pending_obs[idx]
                        prev_a = pending_action[idx]
                        r = accumulated_rewards[idx]
                        
                        # Get masks for buffer
                        # We need re-computation of masks numpy-side or store them?
                        # buffer.push expects numpy arrays.
                        
                        # Re-calculate masks for stored states (expensive?)
                        # Optimize: Store masks in pending too.
                        # For now, quick recalculation
                        
                        # Hack: To get mask from flat features/numpy is hard if we don't have dict.
                        # Actually preprocess_batch returns flat features.
                        # get_action_mask_batch uses dict.
                        # We used dict to get state_p1.
                        # So we can get mask for current state from mask_p1[i].
                        
                        # What about PREV state mask? We should have stored it.
                        # Let's just store the minimal needed.
                        pass # Logic inside loop
                        
                        # To simplify: Let's assume we can push to buffer later.
                        
                        # Wait, we need mask of Prev S (to train valid moves) and Mask of Next S (target).
                        # We have Next S mask (mask_p1[i]).
                        # We did not store Prev S Mask.
                        # Fix: Store mask in pending.
                        
                    # We will handle Pushing strictly in the "Process Outcomes" phase after env.step?
                    # No, we need to know who acted.
                    
                    # Let's do it here:
                    if pending_obs[idx] is not None:
                        prev_s, prev_mask = pending_obs[idx]
                        prev_a = pending_action[idx]
                        rew = accumulated_rewards[idx]
                        
                        curr_s = s_numpy
                        curr_mask = mask_p1[i].cpu().numpy()
                        
                        buffer.push(prev_s, prev_a, rew, curr_s, False, prev_mask, curr_mask)
                    
                    # Reset accumulators for NEW pending
                    pending_obs[idx] = (s_numpy, mask_p1[i].cpu().numpy())
                    pending_action[idx] = a
                    accumulated_rewards[idx] = 0.0

            # --- OPPONENT (P2) ACTIONS ---
            p2_indices = np.where(current_players == 2)[0]
            if len(p2_indices) > 0:
                # Flip board for Opponent
                obs_p2_raw = {k: v[p2_indices] for k, v in obs.items()}
                
                # Canonicalize (Flip 1<->2)
                # Helper assumes dict of arrays
                # We need active mask. Here, ALL are P2, so flip all.
                mask_true = np.ones(len(p2_indices), dtype=bool)
                obs_p2_canon = canonicalize_batch(obs_p2_raw, mask_true)
                
                state_p2 = preprocess_batch(obs_p2_canon, device)
                mask_p2 = get_action_mask_batch(obs_p2_canon, device)
                
                # Model Inference
                with torch.no_grad():
                    q_vals = opponent_model(state_p2)
                    q_vals[~mask_p2] = -float('inf')
                    p2_actions = q_vals.argmax(dim=1).cpu().numpy()
                
                actions[p2_indices] = p2_actions

            # 3. Step Environments
            # actions is (N,) int array
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            # 4. Handle Rewards & Dones
            # Update accumulated rewards for EVERY environment
            # If P1 moved, reward is for P1.
            # If P2 moved, reward is for P2 (which is -Reward for P1).
            # env.step returns Reward for the Player who MOVED.
            # So if P1 moved, R = R_p1. accum += R.
            # If P2 moved, R = R_p2. accum -= R. (Since R_p2 = -R_p1 usually in zero sum)
            # BlackHoleEnv:
            # P1 places. Reward=0 usually.
            # Game End: Returns 1 (Win) or -1 (Loss).
            # If P2 moves and wins, R=1 (for P2). P1 reward should be -1.
            
            # Logic:
            # If index in p1_indices: accum += rewards[idx]
            # If index in p2_indices: accum -= rewards[idx]
            
            # Vectorized update
            accumulated_rewards[p1_indices] += rewards[p1_indices]
            accumulated_rewards[p2_indices] -= rewards[p2_indices]
            
            # 5. Handle Dones (Terminal States)
            dones = terminations | truncations
            done_indices = np.where(dones)[0]
            
            for idx in done_indices:
                if pending_obs[idx] is not None:
                    # Episode finished. Push terminal transition.
                    # Next State? It's the terminal state. 
                    # But VectorEnv resets automatically!
                    # next_obs[idx] is the INITIAL state of new episode.
                    # The terminal state is in infos.
                    
                    term_obs = next_obs # Fallback
                    # Try to get terminal observation from info
                    # Gymnasium VectorEnv: 'final_observation' in infos
                    # infos is dict? or list of dicts? or keys with stacked vals?
                    # SyncVectorEnv usually returns a dict with 'final_observation' array/list?
                    # Or `final_observation` key exists in the "infos" (which is a tuple/dict?)
                    # Wait, Gymnasium `vector_step` returns `infos` as a dictionary, 
                    # with `final_observation` containing an array of states for done envs.
                    # Or `_final_observation` boolean mask?
                    
                    # Let's ensure we get the terminal state.
                    # If we can't easily, using reset state (next_obs) as "Next State" with Done=True is OK-ish
                    # provided we don't bootstrap from it. Q-learning ignores Next S if Done.
                    # So Next S content doesn't matter much if Done=True.
                    # BUT mask matters? No, max_next_q is 0.
                    
                    prev_s, prev_mask = pending_obs[idx]
                    prev_a = pending_action[idx]
                    rew = accumulated_rewards[idx]
                    
                    # We just need a dummy next state and dummy mask
                    dummy_s = prev_s # Shape ok
                    dummy_mask = prev_mask # Shape ok
                    
                    buffer.push(prev_s, prev_a, rew, dummy_s, True, prev_s, prev_mask)
                    
                    # Metrics
                    episode_count += 1
                    recent_rewards.append(rew)
                    recent_wins.append(1 if rew > 0 else 0)
                    
                    # Logging
                    if episode_count % EVAL_INTERVAL == 0:
                        avg_r = np.mean(recent_rewards)
                        train_wr = np.mean(recent_wins)
                        
                        # Exploitation Eval
                        eval_wr = evaluate_winrate(eval_env, current_model, device, num_episodes=20)
                        
                        win_rates.append(eval_wr)
                        avg_rewards_log.append(avg_r)
                        
                        log(f"Episode {episode_count} | AvgReward: {avg_r:.2f} | TrainWR: {train_wr:.2f} | EvalWR: {eval_wr:.2f} | Eps: {epsilon:.2f} | Envs: {NUM_ENVS}")
                        
                        # Opponent Update
                        if eval_wr > SELF_PLAY_UPDATE_THRESHOLD:
                            consecutive_wins += 1
                        else:
                            consecutive_wins = 0

                        episodes_since_update = episode_count - last_opponent_update_episode
                        
                        if consecutive_wins >= OPPONENT_UPDATE_REQ_STREAK and episodes_since_update >= OPPONENT_UPDATE_MIN_EPISODES:
                            log(f">>> PROMOTING MODEL: (Streak: {consecutive_wins}, WR: {eval_wr:.2f}) <<<")
                            opponent_model.load_state_dict(current_model.state_dict())
                            last_opponent_update_episode = episode_count
                            consecutive_wins = 0 # Reset
                        elif consecutive_wins > 0:
                            log(f"Win Streak: {consecutive_wins}/{OPPONENT_UPDATE_REQ_STREAK} (WR: {eval_wr:.2f}) - Waiting for streak or min episodes ({episodes_since_update}/{OPPONENT_UPDATE_MIN_EPISODES})")
                        
                         # Checkpoint
                        if episode_count % CHECKPOINT_INTERVAL == 0:
                            path = os.path.join(save_dir, "checkpoint.pth")
                            save_dict = {
                                'model_config': current_model.config,
                                'state_dict': current_model.state_dict(),
                                'episode': episode_count
                            }
                            torch.save(save_dict, path)
                            log("Checkpoint saved.")
                        
                        # Save Plot Periodically
                        plt.figure(figsize=(12, 5))
                        plt.subplot(1, 2, 1)
                        plt.plot(avg_rewards_log, label='Avg Reward')
                        plt.legend()
                        plt.title("Training Reward")
                        
                        plt.subplot(1, 2, 2)
                        plt.plot(win_rates, label='Win Rate')
                        plt.legend()
                        plt.title("Win Rate vs Opponent")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, "training_plot.png"))
                        plt.close()

                    # Clear pending
                    pending_obs[idx] = None
                    pending_action[idx] = None
                    accumulated_rewards[idx] = 0.0

            
            # Upgrade obs
            obs = next_obs
            global_steps += NUM_ENVS
            
            # 6. Training Step
            # Train model every step? Or every K steps?
            # With 16 envs, we generate 16 transitions per step.
            # We can train multiple times or once.
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
                
                # Q(s', a')
                with torch.no_grad():
                    next_q = target_model(ns_batch)
                    next_q[~nm_batch] = -float('inf')
                    max_next_q = next_q.max(1)[0].unsqueeze(1)
                    target = r_batch + (1 - d_batch) * GAMMA * max_next_q
                
                loss = nn.MSELoss()(q_val, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update Target
                if global_steps % TARGET_UPDATE == 0:
                    target_model.load_state_dict(current_model.state_dict())
                    
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        log("Saving final model.")
        torch.save({
            'model_config': current_model.config,
            'state_dict': current_model.state_dict(),
            'episode': episode_count
        }, os.path.join(save_dir, "model.pth"))
        
        # Final Plot Save
        if avg_rewards_log:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(avg_rewards_log, label='Avg Reward')
            plt.legend()
            plt.title("Training Reward")
            
            plt.subplot(1, 2, 2)
            plt.plot(win_rates, label='Win Rate')
            plt.legend()
            plt.title("Win Rate vs Opponent")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "training_plot.png"))
            plt.close()
            log("Final training plot saved.")
            
        envs.close()

if __name__ == "__main__":
    train()
