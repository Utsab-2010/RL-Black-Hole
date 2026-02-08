
import math
import random
import copy
import numpy as np
import torch
from black_hole.game import BlackHoleGame
from black_hole.model import preprocess_obs, get_action_mask

class MCTSNode:
    def __init__(self, state, parent=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        
        self.visits = 0
        self.total_value = 0.0
        self.mean_value = 0.0  # Q-value
        self.prior_prob = prior_prob  # P(s, a) from network
        
        self.is_expanded = False

    def value(self):
        return self.mean_value

class AlphaMCTS:
    def __init__(self, model, device, cpu_count=1.41, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.device = device
        self.c_puct = cpu_count
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps

    def run(self, root_state, num_simulations=800, temperature=1.0):
        # 1. Create Root
        # We clone state to be safe
        root = MCTSNode(copy.deepcopy(root_state))
        
        # 2. Expand Root (Add noise)
        self._expand_node(root)
        self._add_dirichlet_noise(root)
        
        # 3. Simulations
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded and node.children:
                action, node = self._select_child(node)
                search_path.append(node)
                
            # Evaluation & Expansion
            value = self._evaluate(node)
            
            # If node was terminal, value is actual game result
            if node.state.check_game_over():
                winner, _ = node.state.calculate_score()
                # Value for current player perspective
                # winner 1 => Val=1. 
                # If current_player (who just moved to get here) is 1, return 1.
                # If current_player is 2, return -1.
                # Usually network outputs V for current player.
                # And calculate_score returns winner ID.
                
                # Logic:
                # Node state has 'current_player' who is about to move.
                # BUT the game is over.
                # The winner is absolute.
                # If winner == 1: V = 1.
                # If winner == 2: V = -1.
                # If winner == 0: V = 0.
                # BUT we need V relative to whoever's turn it was supposed to be (or previous player)?
                # Standard AlphaZero: V is always from perspective of current player state.
                # If Game Over, V is ground truth.
                
                if winner == 1:
                    value = 1.0
                elif winner == 2:
                    value = -1.0
                else:
                    value = 0.0
                    
                # If it is P2's turn in this state (meaning P1 moved to get here),
                # and P1 won (value=1), then for P2 this state is BAD (-1).
                # Wait, standard:
                # Score is always relative to 'current_player' of the state.
                # If current_player is 2, and winner is 1, return -1.
                
                cp = node.state.current_player
                if cp == 2: value = -value 
                
            else:
                self._expand_node(node)
            
            # Backpropagation
            self._backpropagate(search_path, value, node.state.current_player)

        # 4. Generate Policy
        counts = {a: child.visits for a, child in root.children.items()}
        total = sum(counts.values())
        
        probs = np.zeros(21)
        
        if temperature == 0:
            best_a = max(counts, key=counts.get)
            probs[best_a] = 1.0
        else:
            # Apply temperature (exponentiate counts)
            # pi ~ N^(1/tau)
            denom = sum(n ** (1/temperature) for n in counts.values())
            for a, n in counts.items():
                probs[a] = (n ** (1/temperature)) / denom
                
        return probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            # PUCT Score
            # Q(s,a) + U(s,a)
            # U = c * P * sqrt(sum(N)) / (1 + N)
            
            q_val = child.mean_value
            # AlphaZero usually flips Q-value perspective naturally during backprop.
            # Child's Q is from child's player perspective (Next Player).
            # So for current player, expected value is -Q.
            # Standard implementation stores Q relative to taking the action.
            
            # My backprop logic:
            # Value V is for current player.
            # Parent (who moves) wants to MAXIMIZE value.
            # So Q should be positive.
            # Let's ensure backprop handles this.
            
            prior = child.prior_prob
            u_val = self.c_puct * prior * math.sqrt(node.visits) / (1 + child.visits)
            
            # Q usually in range [-1, 1].
            # If child has 0 visits, Q=0.
            
            score = q_val + u_val
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def _expand_node(self, node):
        if node.is_expanded or node.state.check_game_over():
            return
            
        # Get Valid Moves
        valid_moves = node.state.get_valid_moves()
        
        # We need Policy Priors from Network if we are evaluating NEW usage.
        # But _evaluate is creating children? No.
        # Strict AlphaZero:
        # Leaf evaluation returns (P, v).
        # We use P to initialize children priors.
        # We use v to backpropagate.
        
        # So expand needs P.
        # Evaluate needs to run first?
        # Standard: Select -> Leaf.
        # Evaluate(Leaf) -> (P, v).
        # Expand Leaf using P.
        # Backup v.
        
        # My loop call evaluate AFTER loop.
        # Correction:
        # Leaf is node.
        # Evaluate(node) -> P, v.
        # Create children using P.
        
        # I did _evaluate inside proper flow.
        pass # Handling in Evaluate

    def _evaluate(self, node):
        # Prepare Input
        # Need to convert game state to tensor
        # BlackHoleGame state structure:
        obs = {
            "board": node.state.board, # (21, 2)
            "current_tile": (node.state.tiles_placed // 2) + 1
        }
        
        # Canonicalize if Player 2
        # AlphaZero network expects canonical input (always Player 1 perspective).
        if node.state.current_player == 2:
            board = obs["board"].copy()
            # Flip 1 <-> 2
            p1 = (board[:, 0] == 1)
            p2 = (board[:, 0] == 2)
            board[p1, 0] = 2
            board[p2, 0] = 1
            obs["board"] = board
            
        state_tensor = preprocess_obs(obs, self.device).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item() # Scalar [-1, 1]
        
        # Expand children here
        valid_moves = node.state.get_valid_moves()
        
        # Renormalize priors over valid moves
        valid_probs = policy_probs[valid_moves]
        sum_probs = valid_probs.sum()
        if sum_probs > 0:
            valid_probs /= sum_probs
        else:
            # Uniform if network creates 0 prob for valid moves (should not happen with softmax)
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
            
        for i, action in enumerate(valid_moves):
            # Create Child
            next_state = copy.deepcopy(node.state)
            
            # Determine tile value logic
            tile_val = (next_state.tiles_placed // 2) + 1
            if tile_val > 10: tile_val = 10
            
            next_state.make_move(action, tile_val)
            
            child = MCTSNode(next_state, parent=node, prior_prob=valid_probs[i])
            node.children[action] = child
            
        node.is_expanded = True
        return value

    def _backpropagate(self, path, value, leaf_player):
        # value is V(s_leaf) from perspective of leaf_player.
        # We walk UP the path.
        # Nodes alternate players.
        # If node.player == leaf_player, value is positive.
        # If node.player != leaf_player, value is negative.
        
        # More robust:
        # value is always relative to "player who just moved" or "player to move"?
        # Network V(s) is "Expected outcome for player s.current_player".
        # So if s.current_player wins, V=1.
        
        current_val_perspective = value 
        # But wait, parent of leaf (who played to get to leaf) wants to MAXIMIZE Val.
        # If V(leaf) = 1 (Leaf Player Wins), then Parent (Opponent) loses. V should be -1 for Parent.
        # So we negate value every step up.
        
        for node in reversed(path):
            node.visits += 1
            node.total_value += current_val_perspective
            node.mean_value = node.total_value / node.visits
            
            current_val_perspective = -current_val_perspective

    def _add_dirichlet_noise(self, node):
        valid_actions = list(node.children.keys())
        if not valid_actions: return
        
        noise = np.random.dirichlet([self.dir_alpha] * len(valid_actions))
        
        for i, action in enumerate(valid_actions):
            child = node.children[action]
            child.prior_prob = (1 - self.dir_eps) * child.prior_prob + self.dir_eps * noise[i]
