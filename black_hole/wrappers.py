import gymnasium as gym
import numpy as np
import random

class SelfPlayWrapper(gym.Wrapper):
    """
    Wraps the BlackHoleEnv to allow a single agent to play against a fixed opponent policy.
    The agent always sees itself as Player 1 (canonical view).
    If it is Player 2's turn in the real env, this wrapper:
    1. Swaps the board perspective (1 <-> 2).
    2. Queries the opponent policy.
    3. Executes opponent move.
    4. Swaps back or returns the resulting state for Player 1.
    """
    def __init__(self, env, opponent_policy=None):
        super().__init__(env)
        self.opponent_policy = opponent_policy
        self.training_as_player_2 = False # Flag if we want to occasionally force being P2?
        # Actually, best practice for self-play is to always have the network output for "Canonical Player".
        # So if we are P2, we flip board, network says "Place at X", we place at X.
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Randomly decide who starts 
        # (Though the game rules say P1 starts, we want to learn to play as P2 too).
        # But in this game, P1 always moves first?
        # Game logic: current_player = 1.
        # If we want to be P2, we let the opponent move first.
        
        self.i_am_player_1 = random.choice([True, False])
        
        if not self.i_am_player_1:
            # Opponent is Player 1.
            # Opponent moves first.
            obs = self._opponent_turn(obs)
            
            # Now it should be P2's turn (which is us).
            # But the observation says current_player=2.
            # We need to flip it to look like P1 for our agent.
            obs = self._canonicalize_obs(obs)
            
        return obs, info

    def step(self, action):
        # We are the agent.
        # Action is for the canonical board.
        # If we are effectively Player 2, the board we saw was flipped.
        # But the move "index" is isometric? Yes, the triangle is symmetric?
        # No, the triangle has specific adjacency.
        # However, the board indices 0-20 are fixed.
        # Does flipping players change the meaning of position "0" (top)?
        # No. Position 0 is always Top.
        # So we don't need to flip ACTION, only BOARD VALUES (who owns what).
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # Game ended on our move.
            # Calculate reward.
            # env.step returns:
            # 1 (P1 wins), -1 (P2 wins).
            # If we are P1, Reward = r.
            # If we are P2, Reward = -r.
            if not self.i_am_player_1:
                reward = -reward
            return self._canonicalize_obs(obs), reward, terminated, truncated, info

        # Now it is Opponent's turn.
        # Opponent sees board.
        # If we are P1, Opponent is P2.
        # If we are P2, Opponent is P1.
        
        # We need to give opponent the correct view.
        # If opponent is P2 and we are P1. Obs is P2 view?
        # env.step returns obs with current_player=2.
        # Opponent policy expects P1 view (canonical).
        # So we canonicalize for opponent.
        
        # NOTE: _opponent_turn handles the canonicalization for the opponent internally
        # OR we pass canonical obs.
        
        # obs = self._opponent_turn(obs) # REMOVED: Caused double step
        # Check if opponent ended game
        
        # Check current status from env
        # If _opponent_turn returns, game might be over or it's our turn.
        
        # wait _opponent_turn calls step.
        # We need to capture the return from _opponent_turn.
        
        # Wait, if _opponent_turn calls step, it returns the tuple.
        # But I need to handle it carefully.
        
        # Let's refactor _opponent_turn to be part of step logic here for clarity.
        pass # placeholder to continue below
    
        # Logic is:
        # Agent Moved. Game NOT over.
        # Now Opponent Move.
        
        # Get Obs for opponent (Canonical)
        # If we are P1, obs is P2 turn. Flip for opponent.
        opp_obs = self._canonicalize_obs(obs)
        
        # Opponent picks action
        if self.opponent_policy:
            opp_action = self.opponent_policy(opp_obs)
        else:
            # Random policy fallback
            valid = info["valid_moves"]
            opp_action = random.choice(valid)
            
        obs, reward, terminated, truncated, info = self.env.step(opp_action)
        
        if terminated or truncated:
            # Game ended on opponent move.
            # Reward is 1 (P1 win), -1 (P2 win).
            # If we are P1, reward is r.
            # If we are P2, reward is -r.
            if not self.i_am_player_1:
                reward = -reward
            return self._canonicalize_obs(obs), reward, terminated, truncated, info
            
        # It is now our turn again.
        # If we are P2, obs is P1 turn? No, it's P2 turn.
        # Wait.
        # Start: P1 turn.
        # A (P1) moves. -> P2 turn.
        # B (P2) moves. -> P1 turn.
        
        # If we are P2. 
        # Opponent (P1) moved first. -> P2 turn.
        # We (P2) moved (Step called). -> P1 turn.
        
        # Opponent (P1) moves. -> P2 turn.
        # We return obs.
        # BUT obs is P2 turn.
        # We need to canonicalize it to look like P1 turn for the agent.
        
        return self._canonicalize_obs(obs), reward, terminated, truncated, info

    def _opponent_turn(self, obs):
        # This helper is used mainly in reset() to do the first move if needed.
        # Similar logic to step.
        
        # Opponent sees canonical obs.
        opp_obs = self._canonicalize_obs(obs)
        
        if self.opponent_policy:
            action = self.opponent_policy(opp_obs)
        else:
            valid = [i for i, x in enumerate(obs["board"]) if x[0] == 0]
            action = random.choice(valid) # Simplified validity check
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # We process reward/done in the caller usually, but for reset()
        # we mostly assume game doesn't end on turn 1.
        return obs

    def _canonicalize_obs(self, obs):
        # If we need the perspective of the "Other" player as "Player 1".
        # We swap the 1s and 2s in the board.
        # AND we swap current_player 1 and 2.
        
        # Copy to avoid mutating original
        new_obs = {
            "board": obs["board"].copy(),
            "current_player": obs["current_player"], # scalar
            "current_tile": obs["current_tile"]
        }
        
        # If current_player is 2, make it 1.
        # If current_player is 1, make it 2.
        # In this wrapper, we usually call this when we want to flip the view.
        # e.g., It is P2's turn. P2 wants to see itself as P1.
        
        cp = new_obs["current_player"]
        if cp == 1:
            new_top = 2
        else:
            new_top = 1
        new_obs["current_player"] = 1 # Always pretend to be P1 in canonical view
        
        # Swap board values
        # 1 -> 2, 2 -> 1, 0 -> 0
        board = new_obs["board"]
        p1_mask = (board[:, 0] == 1)
        p2_mask = (board[:, 0] == 2)
        
        board[p1_mask, 0] = 2
        board[p2_mask, 0] = 1
        
        return new_obs
