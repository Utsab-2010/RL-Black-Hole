import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game import BlackHoleGame

class BlackHoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        self.game = BlackHoleGame()
        self.render_mode = render_mode
        
        # Action space: 0-20 representing the 21 board positions
        self.action_space = spaces.Discrete(21)
        
        # Observation space:
        # board: 21x2 (player, value)
        # current_player: scalar 1 or 2
        # current_tile: scalar 1-10
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=10, shape=(21, 2), dtype=int),
            "current_player": spaces.Discrete(3), # 0 (unused), 1, 2
            "current_tile": spaces.Discrete(11) # 0 (unused), 1-10
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        # Determine value to place based on history/turn
        # Rule: Players place 1 to 10 in order.
        # Turn 0: P1 places 1
        # Turn 1: P2 places 1
        # Turn 2: P1 places 2
        # Turn 3: P2 places 2
        # ...
        # Tile value = (Turn // 2) + 1
        
        current_turn_idx = self.game.tiles_placed
        tile_value = (current_turn_idx // 2) + 1
        
        terminated = False
        truncated = False
        reward = 0
        
        # Validate Action
        valid_moves = self.game.get_valid_moves()
        if action not in valid_moves:
            # Invalid move! 
            # In some RL settings we terminate with heavy penalty
            # Or just ignore it (no-op) but that stalls training.
            # Here we will terminate with penalty to force learning valid moves.
            terminated = True
            reward = -10 # Invalid move penalty
            
            # For debugging/info
            info = self._get_info()
            info["error"] = "Invalid Move"
            return self._get_obs(), reward, terminated, truncated, info

        # Execute Move
        is_game_over = self.game.make_move(action, tile_value)
        
        if is_game_over:
            terminated = True
            # Calculate winner
            winner, reason = self.game.calculate_score()
            
            # Reward shaping:
            # We are training... whom?
            # Usually env is generic. 
            # If we train Player 1: +1 if P1 wins, -1 if P2 wins.
            # But the agent might be controlling P2 on P2's turn?
            # Standard single-agent env usually fixes one player or controls both.
            # Let's assume this env controls the *current player*.
            # But that's non-stationary.
            # For simplicity in this base env:
            # Return result from perspective of Player 1?
            # Or standard: 1=win, -1=loss, 0=draw.
            
            if winner == 1:
                reward = 1
            elif winner == 2:
                reward = -1
            else:
                reward = 0
            
            # Note: If the agent is self-play, it usually needs the reward 
            # relative to the player who just moved.
            # But this is a simple Gym wrapper.
            
        else:
            reward = 0 # Continue playing

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        # Calculate current tile for the *next* move (which is what the agent needs to know)
        current_turn_idx = self.game.tiles_placed
        if current_turn_idx < 20:
            tile_value = (current_turn_idx // 2) + 1
        else:
            tile_value = 0 # Game over state

        return {
            "board": self.game.board.astype(int),
            "current_player": self.game.current_player,
            "current_tile": tile_value
        }

    def _get_info(self):
        return {
            "valid_moves": self.game.get_valid_moves(),
            "turn": self.game.tiles_placed
        }

    def render(self):
        if self.render_mode == "ansi" or self.render_mode == "human":
            self.game.render_ascii()

    def close(self):
        pass
