import pygame
import gymnasium as gym
import black_hole
import torch
import torch.nn as nn
import numpy as np
import os
import sys

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

def main():
    # Parse Args
    import argparse
    parser = argparse.ArgumentParser(description="Play Black Hole against AI")
    parser.add_argument("--model", type=str, help="Path to model .pth file", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    try:
        if args.model:
            model_path = args.model
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return
        else:
            model_path = find_latest_model()
            
        if not model_path:
            print("No model found in trained_models/. Please run train_dqn.py or train_dqn_vector.py first.")
            return

        print(f"Loading model from: {model_path}")
        
        # Load Checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Check if it's the new dictionary format or old state_dict
        if isinstance(checkpoint, dict):
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"Loaded config from checkpoint: {config}")
                agent = QNetwork(**config).to(DEVICE)
            else:
                print("Warning: 'model_config' not found in checkpoint. Using default architecture.")
                agent = QNetwork().to(DEVICE)
            
            if 'state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['state_dict'])
            else:
                # If it's a dict but has no state_dict key, maybe it IS the state_dict?
                # Unlikely if it has other keys, but for safety:
                try:
                    agent.load_state_dict(checkpoint)
                except RuntimeError:
                    print("Error: Could not load state_dict directly.")
                    raise
        else:
            # Fallback for old models (entire object or pure state_dict)
            print("Warning: Loading old format model. Assuming default architecture.")
            agent = QNetwork().to(DEVICE)
            agent.load_state_dict(checkpoint)
            
        agent.eval()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading model: {e}")
        return

    # Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Black Hole - Player vs AI")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Init Env
    env = gym.make("BlackHole-v0")
    obs, info = env.reset()
    
    positions = []
    start_y = 100
    spacing_y = 70
    spacing_x = 70
    center_x = SCREEN_WIDTH // 2
    
    idx = 0
    for row in range(6): # 1+2+3+4+5+6 = 21
        count = row + 1
        row_width = (count - 1) * spacing_x
        start_x = center_x - row_width / 2
        
        for col in range(count):
            x = int(start_x + col * spacing_x)
            y = int(start_y + row * spacing_y)
            positions.append((x, y, 40)) # x, y, radius
            idx += 1
            if idx >= 21: break
            
    running = True
    clock = pygame.time.Clock()
    
    game_over = False
    result_text = ""

    while running:
        human_turn = (obs["current_player"] == 1) # Assume Human is Player 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and human_turn and not game_over:
                mx, my = pygame.mouse.get_pos()
                
                # Check clicks
                valid_moves = info["valid_moves"]
                clicked_idx = -1
                
                for i, (cx, cy, cr) in enumerate(positions):
                    dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
                    if dist < cr:
                        clicked_idx = i
                        break
                
                if clicked_idx != -1:
                    if clicked_idx in valid_moves:
                        # Make Move
                        obs, reward, terminated, truncated, info = env.step(clicked_idx)
                        human_turn = False # Handled instantly but logic persists
                        
                        if terminated or truncated:
                            game_over = True
                            if reward == 1: result_text = "You Won!"
                            elif reward == -1: result_text = "AI Won!"
                            else: result_text = "Draw!"
                    else:
                        print("Invalid Move!")

        # AI Turn
        if not human_turn and not game_over:
            # AI is Player 2.
            # But the agent expects "Canonical Player 1" view.
            # See wrappers.py logic.
            # We need to flip the obs for the AI.
            
            # --- Canonicalize for AI ---
            # Flip Board: 1->2, 2->1.
            ai_board = obs["board"].copy()
            p1_mask = (ai_board[:, 0] == 1)
            p2_mask = (ai_board[:, 0] == 2)
            ai_board[p1_mask, 0] = 2
            ai_board[p2_mask, 0] = 1
            
            ai_obs = {
                "board": ai_board,
                "current_player": 1, # Fake P1
                "current_tile": obs["current_tile"]
            }
            
            # Use updated helpers with DEVICE
            state_tensor = preprocess_obs(ai_obs, DEVICE).unsqueeze(0) # Returns (1, 43)
            
            with torch.no_grad():
                q_values = agent(state_tensor)
                # Mask
                # Use get_action_mask with DEVICE
                mask_tensor = get_action_mask(ai_obs, DEVICE) # Returns (21,) bool tensor
                
                q_values[0, ~mask_tensor] = -float('inf')
                action = q_values.argmax().item()
                
            # Execute AI Move
            print(f"AI plays at {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                game_over = True
                if reward == 1: result_text = "You Won!" # P1 win
                elif reward == -1: result_text = "AI Won!" # P2 win
                else: result_text = "Draw!"


        # --- Drawing ---
        screen.fill(BG_COLOR)
        
        # Draw status
        if not game_over:
            turn_txt = "Your Turn" if human_turn else "AI Thinking..."
            s_surf = font.render(f"P1 (Red) vs AI (Green) | {turn_txt}", True, TEXT_COLOR)
            screen.blit(s_surf, (20, 20))
            
            # Show current tile to place
            # logic: tiles placed = count non-zeros
            tc = (obs["game_turn"] if "game_turn" in obs else info.get("turn", 0)) 
            # Note: gym 'info' has 'turn'.
            tile_val = (tc // 2) + 1
            if tile_val > 10: tile_val = 10
            
            t_surf = font.render(f"Current Tile: {tile_val}", True, TEXT_COLOR)
            screen.blit(t_surf, (20, 60))
        else:
            res_surf = font.render(f"GAME OVER: {result_text}", True, (255, 200, 0))
            screen.blit(res_surf, (center_x - res_surf.get_width()//2, 50))

        # Draw Board
        board = obs["board"]
        for i, (cx, cy, cr) in enumerate(positions):
            player, value = board[i]
            
            color = CIRCLE_COLOR
            if player == 1: color = P1_COLOR
            elif player == 2: color = P2_COLOR
            
            pygame.draw.circle(screen, color, (cx, cy), 30)
            pygame.draw.circle(screen, (200, 200, 200), (cx, cy), 30, 2) # outline
            
            if value > 0:
                v_surf = font.render(str(value), True, (255, 255, 255))
                screen.blit(v_surf, (cx - v_surf.get_width()//2, cy - v_surf.get_height()//2))
            else:
                # Show index small?
                i_surf = small_font.render(str(i), True, (100, 100, 100))
                # screen.blit(i_surf, (cx, cy + 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
