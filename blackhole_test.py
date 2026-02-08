import pygame
import gymnasium as gym
import black_hole
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import traceback

from black_hole.model import QNetwork, preprocess_obs, get_action_mask

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (20, 20, 30)
CIRCLE_COLOR = (50, 50, 60)
HIGHLIGHT_COLOR = (80, 80, 100)
P1_COLOR = (200, 50, 50) # Red
P2_COLOR = (50, 200, 50) # Green
TEXT_COLOR = (255, 255, 255)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_latest_model(base_dir="trained_models"):
    if not os.path.exists(base_dir):
        return None
    
    candidates = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".pth"):
                path = os.path.join(root, f)
                candidates.append(path)
                
    if not candidates:
        return None
        
    latest = max(candidates, key=os.path.getmtime)
    return latest

def load_agent(model_path, device):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    print(f"Loading model from: {model_path}")
    try:
        # Load Checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's the new dictionary format or old state_dict
        if isinstance(checkpoint, dict):
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"Loaded config: {config}")
                agent = QNetwork(**config).to(device)
            else:
                print("Warning: 'model_config' not found. Using default architecture.")
                agent = QNetwork().to(device)
            
            if 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                try:
                    agent.load_state_dict(checkpoint)
                except RuntimeError:
                    print("Error: Could not load state_dict directly.")
                    raise
        else:
            # Fallback for old models
            print("Warning: Loading old format model. Assuming default architecture.")
            agent = QNetwork().to(device)
            agent.load_state_dict(checkpoint)
            
        agent.eval()
        return agent
    except Exception as e:
        traceback.print_exc()
        print(f"Error loading model: {e}")
        return None

def get_ai_move(agent, obs, device, flip_board=False, stochastic=False):
    # Prepare Observation
    if flip_board:
        # Flip Board: 1->2, 2->1 for Agent Input
        ai_board = obs["board"].copy()
        p1_mask = (ai_board[:, 0] == 1)
        p2_mask = (ai_board[:, 0] == 2)
        ai_board[p1_mask, 0] = 2
        ai_board[p2_mask, 0] = 1
        
        ai_obs = {
            "board": ai_board,
            "current_player": 1, # Canonical P1
            "current_tile": obs["current_tile"] # Is current tile specific to player? Usually 1-10 sequence.
        }
    else:
        ai_obs = obs

    state_tensor = preprocess_obs(ai_obs, device).unsqueeze(0)
    
    with torch.no_grad():
        q_values = agent(state_tensor)
        # Mask
        mask_tensor = get_action_mask(ai_obs, device)
        q_values[0, ~mask_tensor] = -float('inf')
        
        if stochastic:
            # Softmax + Multinomial
            probs = torch.softmax(q_values, dim=1) # (1, 21)
            # Ensure masked are 0
            probs[0, ~mask_tensor] = 0
            # Renormalize
            if probs.sum() > 0:
                probs = probs / probs.sum()
                action = torch.multinomial(probs, 1).item()
                # print(probs)
            else:
                # Fallback if numerical issues
                action = q_values.argmax().item()
        else:
            action = q_values.argmax().item()
        
    return action

def main():
    parser = argparse.ArgumentParser(description="Play Black Hole")
    parser.add_argument("--model", type=str, help="Path to opponent model (Human vs AI)", default=None)
    parser.add_argument("--p1", type=str, help="Path to P1 model (AI vs AI)")
    parser.add_argument("--p2", type=str, help="Path to P2 model (AI vs AI)")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2], help="Your Player ID (1 or 2) in Human vs AI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--sim", action="store_true", help="Run in headless simulation mode (AI vs AI only)")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to simulate in --sim mode")
    parser.add_argument("--stochastic-p1", action="store_true", help="Use probabilistic sampling for Player 1")
    parser.add_argument("--stochastic-p2", action="store_true", help="Use probabilistic sampling for Player 2")
    args = parser.parse_args()

    # Determine Mode
    mode = "HumanVsAi" # Default
    agent_p1 = None
    agent_p2 = None
    
    if args.p1 and args.p2:
        mode = "AiVsAi"
        print("Mode: AI vs AI")
        agent_p1 = load_agent(args.p1, DEVICE)
        agent_p2 = load_agent(args.p2, DEVICE)
        if not agent_p1 or not agent_p2: return
    else:
        # Default Loading logic
        # If simulation, we NEED two agents potentially? 
        # Or Sim can be Model vs Self?
        # Let's assume Sim is AI vs AI.
        
        path = args.model
        if not path:
             path = find_latest_model()
             
        if not path:
            print("No model found!")
            return
            
        print(f"Loading Model: {path}")
        model = load_agent(path, DEVICE)
        
        if args.sim:
            # If sim is active but only one model provided, it plays against itself
            mode = "AiVsAi"
            agent_p1 = model
            agent_p2 = model
            print("Simulation Mode: Self-Play (P1 = P2)")
        else:
            print("Mode: Human vs AI")
            human_player = args.player
            if human_player == 1:
                agent_p2 = model
            else:
                agent_p1 = model

    # --- SIMULATION MODE ---
    if args.sim:
        if not agent_p1 or not agent_p2:
            print("Error: For simulation, you must provide models (e.g. via --model for self-play or --p1/--p2)")
            return
            
        print(f"Starting Simulation of {args.num_games} games...")
        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        # Headless Loop
        env = gym.make("BlackHole-v0")
        
        for i in range(args.num_games):
            obs, _ = env.reset()
            done = False
            
            while not done:
                current_p = obs.get("current_player", 1)
                active_agent = agent_p1 if current_p == 1 else agent_p2
                
                # Inference
                flip = (current_p == 2)
                use_stochastic = args.stochastic_p1 if current_p == 1 else args.stochastic_p2
                action = get_ai_move(active_agent, obs, DEVICE, flip_board=flip, stochastic=use_stochastic)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if done:
                    if reward == 1: p1_wins += 1
                    elif reward == -1: p2_wins += 1
                    else: draws += 1
            
            if (i+1) % 10 == 0:
                print(f"Game {i+1}/{args.num_games} complete.")
                
        print("\n--- Simulation Results ---")
        print(f"Total Games: {args.num_games}")
        print(f"Player 1 Wins: {p1_wins} ({p1_wins/args.num_games*100:.1f}%)")
        print(f"Player 2 Wins: {p2_wins} ({p2_wins/args.num_games*100:.1f}%)")
        print(f"Draws:         {draws} ({draws/args.num_games*100:.1f}%)")
        return

    # Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Black Hole - " + mode)
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Init Env
    env = gym.make("BlackHole-v0")
    obs, info = env.reset()
    
    # Board Layout
    positions = []
    start_y = 100
    spacing_y = 70
    spacing_x = 70
    center_x = SCREEN_WIDTH // 2
    
    idx = 0
    for row in range(6): 
        count = row + 1
        row_width = (count - 1) * spacing_x
        start_x = center_x - row_width / 2
        for col in range(count):
            x = int(start_x + col * spacing_x)
            y = int(start_y + row * spacing_y)
            positions.append((x, y, 40)) 
            idx += 1
            if idx >= 21: break
            
    running = True
    clock = pygame.time.Clock()
    game_over = False
    result_text = ""

    while running:
        # Determine Status
        current_p = obs.get("current_player", 1) # Gym env might not have it in obs depending on version, usually in info? 
        # Actually BlackHoleEnv return obs as dict with 'current_player' (it should).
        # Assuming current_player is 1 or 2.
        
        is_human_turn = False
        if mode == "HumanVsAi" and not game_over:
            if current_p == args.player:
                is_human_turn = True
        
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over:
                mx, my = pygame.mouse.get_pos()
                valid_moves = info.get("valid_moves", []) # Helper or use mask
                
                # Check clicks
                clicked_idx = -1
                for i, (cx, cy, cr) in enumerate(positions):
                    dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
                    if dist < cr:
                        clicked_idx = i
                        break
                
                if clicked_idx != -1:
                    # Validate
                    # We can use get_action_mask if valid_moves is not in info
                    # But env.step handles invalid?
                    # Let's trust logic or mask.
                    mask = get_action_mask(obs, DEVICE) # Tensor bool
                    if mask[clicked_idx]:
                        obs, reward, terminated, truncated, info = env.step(clicked_idx)
                        if terminated or truncated:
                            game_over = True
                            if reward == 1: result_text = "Player 1 Wins!"
                            elif reward == -1: result_text = "Player 2 Wins!"
                            else: result_text = "Draw!"
                    else:
                        print("Invalid Move!")

        # AI Turn Logic
        if not is_human_turn and not game_over:
            # Determines which AI agent acts
            active_agent = agent_p1 if current_p == 1 else agent_p2
            
            if active_agent:
                # Inference
                # Flip = True if P2, False if P1
                flip = (current_p == 2)
                
                use_stochastic = args.stochastic_p1 if current_p == 1 else args.stochastic_p2
                action = get_ai_move(active_agent, obs, DEVICE, flip_board=flip, stochastic=use_stochastic)
                
                print(f"AI ({'P1' if current_p==1 else 'P2'}) plays {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    game_over = True
                    if reward == 1: result_text = "Player 1 Wins!"
                    elif reward == -1: result_text = "Player 2 Wins!"
                    else: result_text = "Draw!"

        # --- Drawing ---
        screen.fill(BG_COLOR)
        
        # Header
        if not game_over:
            if mode == "AiVsAi":
                turn_txt = f"AI P{current_p} Thinking..."
            else:
                turn_txt = "Your Turn" if is_human_turn else "AI Thinking..."
            
            s_surf = font.render(turn_txt, True, TEXT_COLOR)
            screen.blit(s_surf, (20, 20))
            
            # Roles Info
            r_surf = small_font.render(f"P1: {'Human' if (mode=='HumanVsAi' and args.player==1) else 'AI'} | P2: {'Human' if (mode=='HumanVsAi' and args.player==2) else 'AI'}", True, (150, 150, 150))
            screen.blit(r_surf, (20, SCREEN_HEIGHT - 30))

            # Tile Info
            # Approximate turn count from tiles placed
            board = obs["board"]
            tiles_placed = np.sum(board[:,0] != 0)
            tile_val = (tiles_placed // 2) + 1
            if tile_val > 10: tile_val = 10
            t_surf = font.render(f"Current Tile: {tile_val}", True, TEXT_COLOR)
            screen.blit(t_surf, (20, 60))

        else:
            res_surf = font.render(f"GAME OVER: {result_text}", True, (255, 200, 0))
            screen.blit(res_surf, (center_x - res_surf.get_width()//2, 50))

        # Board
        board = obs["board"]
        for i, (cx, cy, cr) in enumerate(positions):
            player, value = board[i]
            
            color = CIRCLE_COLOR
            if player == 1: color = P1_COLOR
            elif player == 2: color = P2_COLOR
            
            pygame.draw.circle(screen, color, (cx, cy), 30)
            pygame.draw.circle(screen, (200, 200, 200), (cx, cy), 30, 2) 
            
            if value > 0:
                v_surf = font.render(str(value), True, (255, 255, 255))
                screen.blit(v_surf, (cx - v_surf.get_width()//2, cy - v_surf.get_height()//2))
            # else:
            #     ii_surf = small_font.render(str(i), True, (100,100,100))
            #     screen.blit(ii_surf, (cx, cy))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
