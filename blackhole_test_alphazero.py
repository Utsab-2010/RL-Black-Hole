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

from black_hole.model import AlphaBH, preprocess_obs, get_action_mask
from black_hole.mcts import AlphaMCTS

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

    print(f"Loading AlphaZero model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent = AlphaBH().to(device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['state_dict'])
        else:
            try:
                agent.load_state_dict(checkpoint)
            except RuntimeError:
                print("Error loading raw state_dict. Model might be incompatible.")
                raise
            
        agent.eval()
        return agent
    except Exception as e:
        traceback.print_exc()
        print(f"Error loading model: {e}")
        return None

def get_action(game, role, obs, agent, device, use_mcts, mcts_sims):
    # Prepare Observation
    # If role is 2, the board needs to be flipped for the agent.
    flip_board = (role == 2)

    if flip_board:
        # Flip Board: 1->2, 2->1 for Agent Input
        ai_board = obs["board"].copy()
        p1_mask = (ai_board[:, 0] == 1)
        p2_mask = (ai_board[:, 0] == 2)
        ai_board[p1_mask, 0] = 2
        ai_board[p2_mask, 0] = 1
        
        # The model no longer needs current_tile, but preprocess_obs expects a dict
        # We will pass a dummy current_tile just to satisfy preprocess_obs padding logic
        # if it still looks for it, or just pass the board.
        ai_obs = {
            "board": ai_board,
            "current_player": 2, # Canonical player for flipped board
            "current_tile": 0, # Dummy value, ignored by network now
            "tiles_placed": np.sum(ai_board[:, 0] != 0)
        }
    else:
        ai_obs = obs.copy() # Make a copy to avoid modifying the original obs
        ai_obs["tiles_placed"] = np.sum(obs["board"][:, 0] != 0)

    # Use MCTS
    if use_mcts:
        # MCTS expects a canonical game state (player 1 always to move)
        # The `game` object passed here is already canonical for the MCTS run.
        mcts = AlphaMCTS(agent, device)
        # Use temp=0 for greedy play during test (deterministic based on N)
        probs = mcts.run(game, num_simulations=mcts_sims, temperature=0.0)
        return np.argmax(probs)

    # Fallback to pure Network Policy (No MCTS)
    state_tensor = preprocess_obs(ai_obs, device).unsqueeze(0)
    with torch.no_grad():
        policy_logits, _ = agent(state_tensor)
        q_values = policy_logits
        
        mask_tensor = get_action_mask(ai_obs, device)
        q_values[0, ~mask_tensor] = -float('inf')
        action = q_values.argmax().item()
        
    return action

def main():
    parser = argparse.ArgumentParser(description="Play Black Hole (AlphaZero Model)")
    parser.add_argument("--model", type=str, help="Path to opponent model (Human vs AI)", default=None)
    parser.add_argument("--p1", type=str, help="Path to P1 model (AI vs AI)")
    parser.add_argument("--p2", type=str, help="Path to P2 model (AI vs AI)")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2], help="Your Player ID (1 or 2) in Human vs AI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to simulate in --sim mode")
    parser.add_argument("--eval-sims", type=int, default=10, help="Number of MCTS simulations during testing")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode (AI vs AI)")
    args = parser.parse_args()

    # Determine Mode
    mode = "HumanVsAi" # Default
    agent_p1 = None
    agent_p2 = None
    use_mcts = True # Default to MCTS on
    mcts_sims = args.eval_sims
    
    if args.p1 and args.p2:
        mode = "AiVsAi"
        print("Mode: AI vs AI (AlphaZero)")
        agent_p1 = load_agent(args.p1, DEVICE)
        agent_p2 = load_agent(args.p2, DEVICE)
        if not agent_p1 or not agent_p2: return
    else:
        path = args.model
        if not path:
             path = find_latest_model()
             
        if not path:
            print("No AlphaZero model found in trained_models/!")
            return
            
        print(f"Loading Model: {path}")
        model = load_agent(path, DEVICE)
        if not model: return
        
        if args.sim:
            mode = "AiVsAi"
            agent_p1 = model
            agent_p2 = model
            print("Simulation Mode: Self-Play (P1 = P2)")
        else:
            print(f"Mode: Human vs AI (Playing as Player {args.player})")
            human_player = args.player
            if human_player == 1:
                agent_p2 = model
            else:
                agent_p1 = model

    # --- SIMULATION MODE ---
    if args.sim:
        print(f"Starting Simulation of {args.num_games} games...")
        p1_wins = 0
        p2_wins = 0
        draws = 0
        env = gym.make("BlackHole-v0")
        
        for i in range(args.num_games):
            obs, _ = env.reset()
            done = False
            while not done:
                current_p = obs.get("current_player", 1)
                
                # P1 Turn
                if current_p == 1:
                    action = get_action(env.unwrapped.game, 1, obs, agent_p1, DEVICE, use_mcts, mcts_sims)
                # P2 Turn
                else:
                    action = get_action(env.unwrapped.game, 2, obs, agent_p2, DEVICE, use_mcts, mcts_sims)
                
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
    pygame.display.set_caption("Black Hole AlphaZero Test - " + mode)
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Init Env
    env = gym.make("BlackHole-v0")
    obs, info = env.reset()
    
    # Board Layout — dynamically sized based on game layers
    num_layers = env.unwrapped.game.layers
    num_hexes = env.unwrapped.game.num_hexes
    positions = []
    # Scale spacing to fit the larger pyramid on screen
    spacing_y = max(40, 700 // num_layers)
    spacing_x = max(40, 700 // num_layers)
    circle_r = max(18, spacing_x // 2 - 4)
    center_x = SCREEN_WIDTH // 2
    start_y = max(40, (SCREEN_HEIGHT - spacing_y * num_layers) // 2)
    idx = 0
    for row in range(num_layers):
        count = row + 1
        row_width = (count - 1) * spacing_x
        start_x = center_x - row_width / 2
        for col in range(count):
            x = int(start_x + col * spacing_x)
            y = int(start_y + row * spacing_y)
            positions.append((x, y, circle_r))
            idx += 1
            if idx >= num_hexes: break
            
    running = True
    clock = pygame.time.Clock()
    game_over = False
    result_text = ""

    while running:
        current_p = obs.get("current_player", 1)
        
        is_human_turn = False
        if mode == "HumanVsAi" and not game_over:
            if current_p == args.player:
                is_human_turn = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over:
                mx, my = pygame.mouse.get_pos()
                clicked_idx = -1
                for i, (cx, cy, cr) in enumerate(positions):
                    dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
                    if dist < cr:
                        clicked_idx = i
                        break
                
                if clicked_idx != -1:
                    mask = get_action_mask(obs, DEVICE)
                    if mask[clicked_idx]:
                        obs, reward, terminated, truncated, info = env.step(clicked_idx)
                        if terminated or truncated:
                            game_over = True
                            if reward == 1: result_text = "Player 1 Wins!"
                            elif reward == -1: result_text = "Player 2 Wins!"
                            else: result_text = "Draw!"
                    else:
                        print("Invalid Move!")

        # AI Turn
        if not is_human_turn and not game_over:
            active_agent = agent_p1 if current_p == 1 else agent_p2
            if active_agent:
                # Get the current game state from the environment for MCTS
                game_state = env.unwrapped.game
                
                action = get_action(game_state, current_p, obs, active_agent, DEVICE, use_mcts, mcts_sims)
                
                print(f"AlphaZero ({'P1' if current_p==1 else 'P2'}) plays {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    game_over = True
                    if reward == 1: result_text = "Player 1 Wins!"
                    elif reward == -1: result_text = "Player 2 Wins!"
                    else: result_text = "Draw!"

        # Drawing
        screen.fill(BG_COLOR)
        
        if not game_over:
            turn_txt = f"AlphaZero {current_p} Thinking..." if mode == "AiVsAi" else ("Your Turn" if is_human_turn else "AlphaZero Thinking...")
            s_surf = font.render(turn_txt, True, TEXT_COLOR)
            screen.blit(s_surf, (20, 20))
            
            board = obs["board"]
            tiles_placed = np.sum(board[:,0] != 0)
            tile_val = (tiles_placed // 2) + 1
            if tile_val > 10: tile_val = 10
            t_surf = font.render(f"Current Tile: {tile_val}", True, TEXT_COLOR)
            screen.blit(t_surf, (20, 60))
            
            mcts_status = "ON" if use_mcts else "OFF"
            m_surf = small_font.render(f"MCTS: {mcts_status} (Sims: {args.sim})", True, (150, 150, 150))
            screen.blit(m_surf, (20, SCREEN_HEIGHT - 60))
        else:
            res_surf = font.render(f"GAME OVER: {result_text}", True, (255, 200, 0))
            screen.blit(res_surf, (center_x - res_surf.get_width()//2, 50))

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

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
