import gymnasium as gym
import black_hole
from gymnasium.utils.env_checker import check_env

def test_random_agent():
    print("Initializing BlackHole-v0...")
    env = gym.make("BlackHole-v0", render_mode="ansi")
    
    print("Checking environment conformance with Gymnasium...")
    check_env(env.unwrapped)
    print("Environment conformance check passed!")

    print("\nStarting Random Agent Loop...")
    obs, info = env.reset()
    print("Initial Board:\n")
    env.render()
    
    terminated = False
    truncated = False
    step = 0
    
    while not (terminated or truncated):
        step += 1
        # Random valid action
        valid_moves = info["valid_moves"]
        if not valid_moves:
            print("No valid moves but game not over?!")
            break
            
        action = valid_moves[0] # Just pick first valid move for determinism in this simple test, or random
        import random
        action = random.choice(valid_moves)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step}: Player {3 - obs['current_player']} placed tile {obs['current_tile']-1 if obs['current_tile']>1 else 1} at {action}")
        env.render()
        
        if terminated:
            print(f"\nGame Over! Reward: {reward}")
            if reward == 1: print("Player 1 Won!")
            elif reward == -1: print("Player 2 Won!")
            else: print("Draw!")

if __name__ == "__main__":
    test_random_agent()
