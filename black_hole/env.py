import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game import BlackHoleGame, DEFAULT_LAYERS

class BlackHoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None, layers=DEFAULT_LAYERS):
        self.game = BlackHoleGame(layers=layers)
        self.render_mode = render_mode
        
        N = self.game.num_hexes  # 45 for L=9
        T = self.game.tiles_per_player  # 22 for L=9

        # Action space: 0 to N-1 representing the N board positions
        self.action_space = spaces.Discrete(N)
        
        # Observation space:
        # board: Nx2 (player, value)
        # current_player: scalar 1 or 2
        # current_tile: scalar 1 to T
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=T, shape=(N, 2), dtype=int),
            "current_player": spaces.Discrete(3),  # 0 (unused), 1, 2
            "current_tile": spaces.Discrete(T + 1)  # 0 (unused), 1-T
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        # Tile value = (Turn // 2) + 1
        # Turn 0: P1 places 1, Turn 1: P2 places 1, Turn 2: P1 places 2, etc.
        current_turn_idx = self.game.tiles_placed
        tile_value = (current_turn_idx // 2) + 1
        
        terminated = False
        truncated = False
        reward = 0
        
        # Validate Action
        valid_moves = self.game.get_valid_moves()
        if action not in valid_moves:
            terminated = True
            reward = -10  # Invalid move penalty
            info = self._get_info()
            info["error"] = "Invalid Move"
            return self._get_obs(), reward, terminated, truncated, info

        # Execute Move
        is_game_over = self.game.make_move(action, tile_value)
        
        if is_game_over:
            terminated = True
            winner, reason = self.game.calculate_score()
            
            if winner == 1:
                reward = 1
            elif winner == 2:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        current_turn_idx = self.game.tiles_placed
        max_turns = self.game.num_hexes - 1  # 44 for L=9

        if current_turn_idx < max_turns:
            tile_value = (current_turn_idx // 2) + 1
        else:
            tile_value = 0  # Game over state

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
